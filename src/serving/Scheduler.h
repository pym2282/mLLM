// src/serving/Scheduler.h

#pragma once

#include "serving/RequestQueue.h"
#include "serving/PrefixCache.h"
#include "models/base/IModelRunner.h"
#include <thread>

namespace mllm
{
    // Sequential request scheduler with prefix caching.
    //
    // Maintains a PrefixCacheManager: after each Generate(), the prompt's
    // KV state is saved. On the next request with a matching prefix, the
    // KV state is restored and the runner skips recomputing the cached prefix.
    class Scheduler
    {
    public:
        explicit Scheduler(IModelRunner& runner);
        ~Scheduler();

        void Start();
        void Stop();

        RequestQueue& GetQueue();

        // For inspection / testing
        const PrefixCacheManager& GetPrefixCache() const { return prefix_cache_; }

    private:
        void RunLoop();

        IModelRunner&      runner_;
        RequestQueue       queue_;
        PrefixCacheManager prefix_cache_;
        std::thread        worker_;
        bool               running_ = false;
    };
}
