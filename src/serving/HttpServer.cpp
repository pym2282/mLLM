// src/serving/HttpServer.cpp

#include <httplib/httplib.h>  // must be first on Windows (winsock2 before winsock1)

#include "serving/HttpServer.h"
#include "serving/Scheduler.h"
#include "serving/GenerationRequest.h"
#include "serving/RequestQueue.h"
#include "tokenizer/ITokenizer.h"
#include "models/base/GenerateResult.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mllm
{
    // ----------------------------------------------------------------
    // Token pipe: bridges scheduler thread → httplib content-provider thread
    // Diffs are pre-decoded strings (scheduler thread calls Decode).
    // ----------------------------------------------------------------
    struct TokenPipe
    {
        std::mutex              mu;
        std::condition_variable cv;
        std::deque<std::string> diffs;
        std::atomic<int>        completion_toks{0};
        bool                    finished     = false;
        FinishReason            finish_reason = FinishReason::EOS;
    };

    // Per-request state carried across repeated content-provider callbacks
    struct StreamState
    {
        bool header_sent = false;
        bool done_sent   = false;
    };

    // Mutable decode state used inside on_token (scheduler thread only)
    struct DecodeState
    {
        std::vector<int64_t> acc;
        std::string          prev;
    };

    struct HttpServer::Impl
    {
        httplib::Server svr;
        Scheduler&      scheduler;
        ITokenizer&     tokenizer;
        int             port;

        Impl(Scheduler& sched, ITokenizer& tok, int p)
            : scheduler(sched), tokenizer(tok), port(p) {}

        void HandleGenerate(const httplib::Request& req, httplib::Response& res);
        void HandleChatCompletions(const httplib::Request& req, httplib::Response& res);
    };

    // ----------------------------------------------------------------
    // helpers
    // ----------------------------------------------------------------

    static nlohmann::json ErrorJson(const std::string& msg)
    {
        return nlohmann::json{{"error", msg}};
    }

    static std::string FinishReasonStr(FinishReason r)
    {
        switch (r)
        {
        case FinishReason::EOS:    return "stop";   // OpenAI compat
        case FinishReason::Stop:   return "stop";
        case FinishReason::Length: return "length";
        }
        return "length";
    }

    static std::string SseDelta(
        const std::string&    request_id,
        const std::string&    model_name,
        long long             created,
        const nlohmann::json& delta,
        const std::string&    finish_reason = "")
    {
        nlohmann::json choice = {
            {"index", 0},
            {"delta", delta}
        };
        choice["finish_reason"] = finish_reason.empty()
            ? nlohmann::json(nullptr)
            : nlohmann::json(finish_reason);

        nlohmann::json obj = {
            {"id",      "chatcmpl-" + request_id},
            {"object",  "chat.completion.chunk"},
            {"created", created},
            {"model",   model_name},
            {"choices", nlohmann::json::array({choice})}
        };
        return "data: " + obj.dump() + "\n\n";
    }

    // ----------------------------------------------------------------
    // POST /v1/generate
    // ----------------------------------------------------------------
    void HttpServer::Impl::HandleGenerate(
        const httplib::Request& req,
        httplib::Response&      res)
    {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (const std::exception& e)
        {
            res.status = 400;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        if (!body.contains("prompt") || !body["prompt"].is_string())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("'prompt' (string) is required").dump(),
                "application/json");
            return;
        }

        const std::string prompt = body["prompt"].get<std::string>();

        GenerateOptions opts;
        if (body.contains("max_new_tokens") && body["max_new_tokens"].is_number())
            opts.max_new_tokens = body["max_new_tokens"].get<int>();
        if (body.contains("temperature") && body["temperature"].is_number())
            opts.temperature = body["temperature"].get<float>();
        if (body.contains("top_k") && body["top_k"].is_number())
            opts.top_k = body["top_k"].get<int>();
        if (body.contains("top_p") && body["top_p"].is_number())
            opts.top_p = body["top_p"].get<float>();
        if (body.contains("use_greedy") && body["use_greedy"].is_boolean())
            opts.use_greedy = body["use_greedy"].get<bool>();
        if (body.contains("repetition_penalty") && body["repetition_penalty"].is_number())
            opts.repetition_penalty = body["repetition_penalty"].get<float>();
        opts.enable_thinking = tokenizer.SupportsThinking();
        if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
            opts.enable_thinking = body["enable_thinking"].get<bool>();
        opts.eos_token_id = tokenizer.GetEOSTokenId();

        if (body.contains("stop") && body["stop"].is_array())
        {
            for (const auto& s : body["stop"])
            {
                if (!s.is_string()) continue;
                const auto ids = tokenizer.Encode(s.get<std::string>());
                if (!ids.empty())
                    opts.stop_sequence_ids.push_back(ids);
            }
        }

        auto prompt_ids = tokenizer.Encode(prompt);
        if (prompt_ids.empty())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("tokenizer produced empty sequence for prompt").dump(),
                "application/json");
            return;
        }

        auto gen_req = std::make_shared<GenerationRequest>();
        gen_req->request_id    = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        gen_req->prompt_tokens = std::move(prompt_ids);
        gen_req->options       = opts;

        auto future = gen_req->result_promise.get_future();

        try { scheduler.GetQueue().Push(std::move(gen_req)); }
        catch (const std::exception& e)
        {
            res.status = 503;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        GenerateResult result;
        try { result = future.get(); }
        catch (const std::exception& e)
        {
            res.status = 500;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        const std::string text = tokenizer.Decode(result.tokens);

        nlohmann::json response{
            {"text",          text},
            {"finish_reason", FinishReasonStr(result.finish_reason)}
        };
        res.set_content(response.dump(), "application/json");
    }

    // ----------------------------------------------------------------
    // POST /v1/chat/completions  (OpenAI-compatible, streaming + non-streaming)
    // ----------------------------------------------------------------
    void HttpServer::Impl::HandleChatCompletions(
        const httplib::Request& req,
        httplib::Response&      res)
    {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (const std::exception& e)
        {
            res.status = 400;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("'messages' (array) is required").dump(),
                "application/json");
            return;
        }

        const bool        do_stream  = body.value("stream", false);
        const std::string model_name = body.value("model", "mllm");

        std::vector<mllm::Message> messages;
        for (const auto& m : body["messages"])
        {
            if (!m.contains("role") || !m.contains("content")) continue;
            if (!m["role"].is_string() || !m["content"].is_string()) continue;
            messages.push_back({m["role"].get<std::string>(),
                                 m["content"].get<std::string>()});
        }

        if (messages.empty())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("messages array is empty or malformed").dump(),
                "application/json");
            return;
        }

        GenerateOptions opts;
        if (body.contains("max_tokens") && body["max_tokens"].is_number())
            opts.max_new_tokens = body["max_tokens"].get<int>();
        if (body.contains("temperature") && body["temperature"].is_number())
            opts.temperature = body["temperature"].get<float>();
        if (body.contains("top_p") && body["top_p"].is_number())
            opts.top_p = body["top_p"].get<float>();
        opts.enable_thinking = tokenizer.SupportsThinking();
        if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
            opts.enable_thinking = body["enable_thinking"].get<bool>();
        opts.eos_token_id = tokenizer.GetEOSTokenId();

        const std::string prompt =
            tokenizer.BuildPromptFromMessages(messages, opts.enable_thinking);

        auto prompt_ids = tokenizer.Encode(prompt);
        if (prompt_ids.empty())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("tokenizer produced empty sequence").dump(),
                "application/json");
            return;
        }

        const int         prompt_token_count = static_cast<int>(prompt_ids.size());
        const std::string request_id = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const long long created =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        // ---- STREAMING PATH ----
        if (do_stream)
        {
            auto pipe  = std::make_shared<TokenPipe>();
            auto state = std::make_shared<StreamState>();
            auto ds    = std::make_shared<DecodeState>();

            const int64_t eos_id = opts.eos_token_id;
            ITokenizer&   tok    = tokenizer;

            // on_token: runs on scheduler thread; decodes diff, pushes to pipe
            opts.on_token = [pipe, ds, &tok, eos_id](int64_t token) -> bool {
                pipe->completion_toks.fetch_add(1, std::memory_order_relaxed);

                if (token != eos_id)
                {
                    ds->acc.push_back(token);
                    std::string new_dec = tok.Decode(ds->acc);
                    std::string diff    = new_dec.substr(ds->prev.size());
                    ds->prev            = std::move(new_dec);

                    if (!diff.empty())
                    {
                        {
                            std::lock_guard<std::mutex> lk(pipe->mu);
                            pipe->diffs.push_back(std::move(diff));
                        }
                        pipe->cv.notify_one();
                    }
                }
                return true;
            };

            auto gen_req = std::make_shared<GenerationRequest>();
            gen_req->request_id    = request_id;
            gen_req->prompt_tokens = std::move(prompt_ids);
            gen_req->options       = opts;

            auto future = gen_req->result_promise.get_future();

            try { scheduler.GetQueue().Push(std::move(gen_req)); }
            catch (const std::exception& e)
            {
                res.status = 503;
                res.set_content(ErrorJson(e.what()).dump(), "application/json");
                return;
            }

            // Watcher thread: marks pipe finished after generation completes
            std::thread([pipe, f = std::move(future)]() mutable {
                FinishReason fr = FinishReason::EOS;
                try { fr = f.get().finish_reason; } catch (...) {}
                {
                    std::lock_guard<std::mutex> lk(pipe->mu);
                    pipe->finish_reason = fr;
                    pipe->finished      = true;
                }
                pipe->cv.notify_one();
            }).detach();

            const std::string rid    = request_id;
            const std::string mname  = model_name;
            const long long   ts     = created;
            const int         ptoks  = prompt_token_count;

            res.set_chunked_content_provider(
                "text/event-stream",
                [pipe, state, rid, mname, ts, ptoks](
                    size_t /*offset*/,
                    httplib::DataSink& sink) -> bool
                {
                    if (state->done_sent)
                        return false;

                    // Role header — sent once before any content
                    if (!state->header_sent)
                    {
                        state->header_sent = true;
                        const std::string hdr = SseDelta(
                            rid, mname, ts, {{"role", "assistant"}});
                        if (!sink.write(hdr.data(), hdr.size()))
                            return false;
                    }

                    // Wait for next diff or finished signal
                    std::string  diff;
                    bool         pipe_done = false;
                    FinishReason fr        = FinishReason::EOS;

                    {
                        std::unique_lock<std::mutex> lk(pipe->mu);
                        pipe->cv.wait(lk, [&pipe] {
                            return !pipe->diffs.empty() || pipe->finished;
                        });

                        if (!pipe->diffs.empty())
                        {
                            diff = std::move(pipe->diffs.front());
                            pipe->diffs.pop_front();
                        }
                        else
                        {
                            pipe_done = true;
                            fr        = pipe->finish_reason;
                        }
                    }

                    if (!diff.empty())
                    {
                        const std::string chunk = SseDelta(
                            rid, mname, ts, {{"content", diff}});
                        return sink.write(chunk.data(), chunk.size());
                    }

                    if (pipe_done)
                    {
                        // Final chunk with finish_reason
                        const std::string fin = SseDelta(
                            rid, mname, ts,
                            nlohmann::json::object(),
                            FinishReasonStr(fr));
                        sink.write(fin.data(), fin.size());

                        // Usage chunk
                        const int ctoks = pipe->completion_toks.load();
                        nlohmann::json usage_obj = {
                            {"id",      "chatcmpl-" + rid},
                            {"object",  "chat.completion.chunk"},
                            {"created", ts},
                            {"model",   mname},
                            {"choices", nlohmann::json::array()},
                            {"usage",   {
                                {"prompt_tokens",     ptoks},
                                {"completion_tokens", ctoks},
                                {"total_tokens",      ptoks + ctoks}
                            }}
                        };
                        const std::string usage_str =
                            "data: " + usage_obj.dump() + "\n\n";
                        sink.write(usage_str.data(), usage_str.size());

                        // SSE done sentinel
                        const std::string done_str = "data: [DONE]\n\n";
                        sink.write(done_str.data(), done_str.size());
                        sink.done();

                        state->done_sent = true;
                        return false;
                    }

                    return true;
                }
            );

            return;
        }

        // ---- NON-STREAMING PATH ----
        auto gen_req = std::make_shared<GenerationRequest>();
        gen_req->request_id    = request_id;
        gen_req->prompt_tokens = std::move(prompt_ids);
        gen_req->options       = opts;

        auto future = gen_req->result_promise.get_future();

        try { scheduler.GetQueue().Push(std::move(gen_req)); }
        catch (const std::exception& e)
        {
            res.status = 503;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        GenerateResult result;
        try { result = future.get(); }
        catch (const std::exception& e)
        {
            res.status = 500;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        const std::string text = tokenizer.Decode(result.tokens);

        std::cout
            << "[HttpServer] Response (" << text.size() << " chars): "
            << text.substr(0, 200) << std::endl;

        const int ctoks = static_cast<int>(result.tokens.size());
        nlohmann::json response = {
            {"id",      "chatcmpl-" + request_id},
            {"object",  "chat.completion"},
            {"created", created},
            {"model",   model_name},
            {"choices", nlohmann::json::array({
                {
                    {"index", 0},
                    {"message", {
                        {"role",    "assistant"},
                        {"content", text}
                    }},
                    {"finish_reason", FinishReasonStr(result.finish_reason)}
                }
            })},
            {"usage", {
                {"prompt_tokens",     prompt_token_count},
                {"completion_tokens", ctoks},
                {"total_tokens",      prompt_token_count + ctoks}
            }}
        };
        res.set_content(response.dump(), "application/json");
    }

    // ----------------------------------------------------------------
    // construction / Run / Stop
    // ----------------------------------------------------------------

    HttpServer::HttpServer(
        Scheduler&  scheduler,
        ITokenizer& tokenizer,
        int         port)
        : impl_(std::make_unique<Impl>(scheduler, tokenizer, port))
    {
        impl_->svr.Post("/v1/generate",
            [this](const httplib::Request& req, httplib::Response& res)
            { impl_->HandleGenerate(req, res); });

        impl_->svr.Post("/v1/chat/completions",
            [this](const httplib::Request& req, httplib::Response& res)
            { impl_->HandleChatCompletions(req, res); });

        impl_->svr.Get("/health",
            [](const httplib::Request&, httplib::Response& res)
            { res.set_content(R"({"status":"ok"})", "application/json"); });
    }

    HttpServer::~HttpServer() = default;

    void HttpServer::Run()
    {
        std::cout << "[HttpServer] Listening on port " << impl_->port << std::endl;
        impl_->svr.listen("0.0.0.0", impl_->port);
    }

    void HttpServer::Stop()
    {
        impl_->svr.stop();
    }
}
