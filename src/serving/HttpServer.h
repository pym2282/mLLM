// src/serving/HttpServer.h

#pragma once

#include <memory>
#include <string>

namespace mllm
{
    class ITokenizer;
    class Scheduler;

    // HTTP inference server (POST /v1/generate, GET /health).
    // Uses PIMPL so httplib.h is only compiled once in HttpServer.cpp.
    class HttpServer
    {
    public:
        HttpServer(
            Scheduler&   scheduler,
            ITokenizer&  tokenizer,
            int          port = 8080);

        ~HttpServer();

        // Blocks until Stop() is called from another thread.
        void Run();

        void Stop();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
