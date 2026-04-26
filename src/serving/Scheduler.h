// src/serving/Scheduler.h

#pragma once

#include "serving/RequestQueue.h"
#include "models/base/IModelRunner.h"
#include <thread>

namespace mllm
{
    // Sequential request scheduler.
    //
    // Owns a single worker thread that pops requests from RequestQueue
    // and processes them one at a time via IModelRunner::Generate().
    //
    // NOTE: LlamaRunner::Generate() mutates internal kv_caches_.
    // Only one Generate() may run at a time — never call runner_ concurrently.
    class Scheduler
    {
    public:
        explicit Scheduler(IModelRunner& runner);
        ~Scheduler();

        void Start();
        void Stop();

        RequestQueue& GetQueue();

    private:
        void RunLoop();

        IModelRunner& runner_;
        RequestQueue  queue_;
        std::thread   worker_;
        bool          running_ = false;
    };
}
