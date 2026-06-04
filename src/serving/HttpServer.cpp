// src/serving/HttpServer.cpp

#include <httplib/httplib.h>  // must be first on Windows (winsock2 before winsock1)

#include "serving/HttpServer.h"
#include "serving/Scheduler.h"
#include "serving/GenerationRequest.h"
#include "serving/RequestQueue.h"
#include "tokenizer/ITokenizer.h"
#include "models/base/GenerateResult.h"

#include <nlohmann/json.hpp>
#include "core/Logger.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
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

    // Returns the byte length of the longest valid UTF-8 prefix of s.
    // BPE byte-level tokens can produce partial multi-byte sequences; this
    // function ensures we never emit a diff that cuts inside a codepoint.
    static size_t Utf8SafeLen(const std::string& s)
    {
        if (s.empty()) return 0;
        const size_t n = s.size();
        // Walk backwards past continuation bytes (10xxxxxx).
        size_t back = 0;
        while (back < 3 && back < n &&
               (static_cast<unsigned char>(s[n - 1 - back]) & 0xC0) == 0x80)
            ++back;
        if (back >= n) return 0;
        const unsigned char lead = static_cast<unsigned char>(s[n - 1 - back]);
        size_t seq_len;
        if      ((lead & 0x80) == 0x00) seq_len = 1;
        else if ((lead & 0xE0) == 0xC0) seq_len = 2;
        else if ((lead & 0xF0) == 0xE0) seq_len = 3;
        else if ((lead & 0xF8) == 0xF0) seq_len = 4;
        else return n - 1 - back; // invalid lead byte — truncate before it
        // If we have all bytes of this sequence, string is complete.
        return (back + 1 == seq_len) ? n : n - 1 - back;
    }

    // Mutable decode state used inside on_token (scheduler thread only)
    struct DecodeState
    {
        std::vector<int64_t> acc;
        size_t               emitted = 0;  // bytes sent to pipe, always at UTF-8 boundary
    };

    struct HttpServer::Impl
    {
        httplib::Server svr;
        Scheduler&      scheduler;
        ITokenizer&     tokenizer;
        int             port;

        // Watcher futures: owned here so Stop() can wait for them to finish.
        std::mutex                   watcher_mu;
        std::vector<std::future<void>> watchers;

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
        return "data: " + obj.dump(-1, ' ', true) + "\n\n";
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
            res.set_content(ErrorJson(e.what()).dump(-1, ' ', true), "application/json; charset=utf-8");
            return;
        }

        if (!body.contains("prompt") || !body["prompt"].is_string())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("'prompt' (string) is required").dump(-1, ' ', true),
                "application/json; charset=utf-8");
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
        opts.enable_thinking = false;
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
                ErrorJson("tokenizer produced empty sequence for prompt").dump(-1, ' ', true),
                "application/json; charset=utf-8");
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
            res.set_content(ErrorJson(e.what()).dump(-1, ' ', true), "application/json; charset=utf-8");
            return;
        }

        GenerateResult result;
        try
        {
            if (future.wait_for(std::chrono::seconds(300)) != std::future_status::ready)
            {
                res.status = 504;
                res.set_content(ErrorJson("generation timeout").dump(-1, ' ', true),
                                "application/json; charset=utf-8");
                return;
            }
            result = future.get();
        }
        catch (const std::exception& e)
        {
            MLLM_ERROR("HttpServer", "generate error: " + std::string(e.what()));
            res.status = 500;
            res.set_content(ErrorJson("internal server error").dump(-1, ' ', true),
                            "application/json; charset=utf-8");
            return;
        }

        const std::string text = tokenizer.Decode(result.tokens);

        nlohmann::json response{
            {"text",          text},
            {"finish_reason", FinishReasonStr(result.finish_reason)}
        };
        res.set_content(response.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace), "application/json; charset=utf-8");
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
            res.set_content(ErrorJson(e.what()).dump(-1, ' ', true), "application/json; charset=utf-8");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("'messages' (array) is required").dump(-1, ' ', true),
                "application/json; charset=utf-8");
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
                ErrorJson("messages array is empty or malformed").dump(-1, ' ', true),
                "application/json; charset=utf-8");
            return;
        }

        GenerateOptions opts;
        if (body.contains("max_tokens") && body["max_tokens"].is_number())
            opts.max_new_tokens = body["max_tokens"].get<int>();
        if (body.contains("temperature") && body["temperature"].is_number())
            opts.temperature = body["temperature"].get<float>();
        if (body.contains("top_p") && body["top_p"].is_number())
            opts.top_p = body["top_p"].get<float>();
        opts.enable_thinking = false;
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
                ErrorJson("tokenizer produced empty sequence").dump(-1, ' ', true),
                "application/json; charset=utf-8");
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

            const int64_t eos_id  = opts.eos_token_id;
            ITokenizer*   tok_ptr = &tokenizer;  // raw ptr, safe: tokenizer outlives server

            opts.on_token = [pipe, ds, tok_ptr, eos_id](int64_t token) -> bool {
                pipe->completion_toks.fetch_add(1, std::memory_order_relaxed);
                if (token != eos_id) {
                    ds->acc.push_back(token);
                    const std::string new_dec = tok_ptr->Decode(ds->acc);
                    const size_t safe = Utf8SafeLen(new_dec);
                    if (safe > ds->emitted) {
                        std::string diff = new_dec.substr(ds->emitted, safe - ds->emitted);
                        ds->emitted = safe;
                        { std::lock_guard<std::mutex> lk(pipe->mu); pipe->diffs.push_back(std::move(diff)); }
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
                res.set_content(ErrorJson(e.what()).dump(-1, ' ', true), "application/json; charset=utf-8");
                return;
            }

            // Watcher: waits for generation result and signals the pipe.
            // Stored in this->watchers so Stop() can join them on shutdown.
            {
                auto w = std::async(std::launch::async,
                    [pipe, gen_fut = std::move(future), self = this]() mutable {
                        FinishReason fr = FinishReason::EOS;
                        try { fr = gen_fut.get().finish_reason; } catch (...) {}
                        {
                            std::lock_guard<std::mutex> lk(pipe->mu);
                            pipe->finish_reason = fr;
                            pipe->finished      = true;
                        }
                        pipe->cv.notify_one();
                        // Prune already-finished watchers from the list.
                        std::lock_guard<std::mutex> wlk(self->watcher_mu);
                        auto& v = self->watchers;
                        v.erase(
                            std::remove_if(v.begin(), v.end(),
                                [](const std::future<void>& fv) {
                                    return fv.wait_for(std::chrono::seconds(0))
                                           == std::future_status::ready;
                                }),
                            v.end());
                    });
                std::lock_guard<std::mutex> wlk(watcher_mu);
                watchers.push_back(std::move(w));
            }

            res.set_chunked_content_provider(
                "text/event-stream",
                [pipe, state, request_id, model_name, created, prompt_token_count](
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
                            request_id, model_name, created, {{"role", "assistant"}});
                        if (!sink.write(hdr.data(), hdr.size()))
                            return false;
                    }

                    // Wait for next diff or finished signal
                    std::string  diff;
                    bool         pipe_done = false;
                    FinishReason fr        = FinishReason::EOS;

                    {
                        std::unique_lock<std::mutex> lk(pipe->mu);
                        if (!pipe->cv.wait_for(lk, std::chrono::seconds(300), [&pipe] {
                                return !pipe->diffs.empty() || pipe->finished;
                            }))
                        {
                            // Timed out waiting for next token — close the stream.
                            sink.done();
                            state->done_sent = true;
                            return false;
                        }

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
                            request_id, model_name, created, {{"content", diff}});
                        return sink.write(chunk.data(), chunk.size());
                    }

                    if (pipe_done)
                    {
                        // Final chunk with finish_reason
                        const std::string fin = SseDelta(
                            request_id, model_name, created,
                            nlohmann::json::object(),
                            FinishReasonStr(fr));
                        sink.write(fin.data(), fin.size());

                        // Usage chunk
                        const int ctoks = pipe->completion_toks.load();
                        nlohmann::json usage_obj = {
                            {"id",      "chatcmpl-" + request_id},
                            {"object",  "chat.completion.chunk"},
                            {"created", created},
                            {"model",   model_name},
                            {"choices", nlohmann::json::array()},
                            {"usage",   {
                                {"prompt_tokens",     prompt_token_count},
                                {"completion_tokens", ctoks},
                                {"total_tokens",      prompt_token_count + ctoks}
                            }}
                        };
                        const std::string usage_str =
                            "data: " + usage_obj.dump(-1, ' ', true) + "\n\n";
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
            res.set_content(ErrorJson(e.what()).dump(-1, ' ', true), "application/json; charset=utf-8");
            return;
        }

        GenerateResult result;
        try
        {
            if (future.wait_for(std::chrono::seconds(300)) != std::future_status::ready)
            {
                res.status = 504;
                res.set_content(ErrorJson("generation timeout").dump(-1, ' ', true),
                                "application/json; charset=utf-8");
                return;
            }
            result = future.get();
        }
        catch (const std::exception& e)
        {
            res.status = 500;
            res.set_content(ErrorJson(e.what()).dump(-1, ' ', true), "application/json; charset=utf-8");
            return;
        }

        const std::string text  = tokenizer.Decode(result.tokens);
        const int         ctoks = static_cast<int>(result.tokens.size());
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
        res.set_content(response.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace), "application/json; charset=utf-8");
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
            { res.set_content(R"({"status":"ok"})", "application/json; charset=utf-8"); });
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
        // Wait for any in-flight watcher futures to finish.
        std::lock_guard<std::mutex> wlk(impl_->watcher_mu);
        for (auto& w : impl_->watchers)
            if (w.valid()) w.wait();
        impl_->watchers.clear();
    }

}
