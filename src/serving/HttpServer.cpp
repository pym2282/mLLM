// src/serving/HttpServer.cpp

#include "serving/HttpServer.h"
#include "serving/Scheduler.h"
#include "serving/GenerationRequest.h"
#include "serving/RequestQueue.h"
#include "tokenizer/ITokenizer.h"
#include "models/base/GenerateResult.h"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mllm
{
    struct HttpServer::Impl
    {
        httplib::Server svr;
        Scheduler&      scheduler;
        ITokenizer&     tokenizer;
        int             port;

        Impl(Scheduler& sched, ITokenizer& tok, int p)
            : scheduler(sched), tokenizer(tok), port(p) {}
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
        case FinishReason::EOS:    return "eos";
        case FinishReason::Stop:   return "stop";
        case FinishReason::Length: return "length";
        }
        return "length";
    }

    // ----------------------------------------------------------------
    // POST /v1/generate
    //
    // Request JSON:
    //   {
    //     "prompt"             : "Hello",          // required
    //     "max_new_tokens"     : 256,              // optional
    //     "temperature"        : 0.7,              // optional
    //     "top_k"              : 40,               // optional
    //     "top_p"              : 0.9,              // optional
    //     "use_greedy"         : false,            // optional
    //     "repetition_penalty" : 1.1,              // optional
    //     "stop"               : ["<|im_end|>"]    // optional string list
    //   }
    //
    // Response JSON:
    //   { "text": "...", "finish_reason": "eos|stop|length" }
    // ----------------------------------------------------------------
    static void HandleGenerate(
        const httplib::Request& req,
        httplib::Response&      res,
        HttpServer::Impl&       impl)
    {
        nlohmann::json body;
        try
        {
            body = nlohmann::json::parse(req.body);
        }
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
        if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
            opts.enable_thinking = body["enable_thinking"].get<bool>();

        opts.eos_token_id = impl.tokenizer.GetEOSTokenId();

        // Encode stop strings into token-ID sequences
        if (body.contains("stop") && body["stop"].is_array())
        {
            for (const auto& s : body["stop"])
            {
                if (!s.is_string()) continue;
                const auto ids = impl.tokenizer.Encode(s.get<std::string>());
                if (!ids.empty())
                    opts.stop_sequence_ids.push_back(ids);
            }
        }

        auto prompt_ids = impl.tokenizer.Encode(prompt);
        if (prompt_ids.empty())
        {
            res.status = 400;
            res.set_content(
                ErrorJson("tokenizer produced empty sequence for prompt").dump(),
                "application/json");
            return;
        }

        // Build and enqueue request
        auto gen_req = std::make_shared<GenerationRequest>();
        gen_req->request_id   = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        gen_req->prompt_tokens = std::move(prompt_ids);
        gen_req->options       = opts;

        auto future = gen_req->result_promise.get_future();

        try
        {
            impl.scheduler.GetQueue().Push(std::move(gen_req));
        }
        catch (const std::exception& e)
        {
            res.status = 503;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        // Block until generation completes
        GenerateResult result;
        try
        {
            result = future.get();
        }
        catch (const std::exception& e)
        {
            res.status = 500;
            res.set_content(ErrorJson(e.what()).dump(), "application/json");
            return;
        }

        const std::string text = impl.tokenizer.Decode(result.tokens);

        nlohmann::json response{
            {"text",          text},
            {"finish_reason", FinishReasonStr(result.finish_reason)}
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
            {
                HandleGenerate(req, res, *impl_);
            });

        impl_->svr.Get("/health",
            [](const httplib::Request&, httplib::Response& res)
            {
                res.set_content(R"({"status":"ok"})", "application/json");
            });
    }

    HttpServer::~HttpServer() = default;

    void HttpServer::Run()
    {
        std::cout
            << "[HttpServer] Listening on port "
            << impl_->port
            << std::endl;

        impl_->svr.listen("0.0.0.0", impl_->port);
    }

    void HttpServer::Stop()
    {
        impl_->svr.stop();
    }
}
