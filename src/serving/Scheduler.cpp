// src/serving/Scheduler.cpp

#include "serving/Scheduler.h"
#include "models/base/GenerateResult.h"
#include <iostream>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace mllm
{
    Scheduler::Scheduler(IModelRunner& runner)
        : runner_(runner)
    {}

    Scheduler::~Scheduler()
    {
        Stop();
    }

    void Scheduler::Start()
    {
        if (running_)
            return;

        running_ = true;
        worker_ = std::thread(&Scheduler::RunLoop, this);
    }

    void Scheduler::Stop()
    {
        if (!running_)
            return;

        queue_.Shutdown();

        if (worker_.joinable())
            worker_.join();

        running_ = false;
    }

    RequestQueue& Scheduler::GetQueue()
    {
        return queue_;
    }

    void Scheduler::RunLoop()
    {
        c10::cuda::set_device(0);
        (void)c10::cuda::getCurrentCUDAStream(0);
        {
            auto warmup = torch::empty(
                {1},
                torch::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(torch::kCUDA)
            );
        }

        while (true)
        {
            auto req = queue_.Pop();
            if (!req) break;

            req->status = RequestStatus::Running;

            std::cerr << "[Scheduler] Processing: " << req->request_id << "\n";

            try
            {
                // Prefix cache lookup: 기존 KV 재사용 가능한지 확인
                KVSnapshot cached_snap;
                const int64_t prefix_hit = prefix_cache_.Lookup(
                    req->prompt_tokens, cached_snap);

                if (prefix_hit > 0)
                {
                    runner_.SetKVSnapshot(cached_snap);
                    req->options.prefix_kv_len = prefix_hit;
                    std::cerr << "[Scheduler] Prefix cache hit: " << prefix_hit
                              << " tokens skipped\n";
                }

                GenerateResult output = runner_.Generate(
                    req->prompt_tokens,
                    req->options
                );

                // Generate 완료 후 prompt KV를 캐시에 저장
                const int64_t prompt_len =
                    static_cast<int64_t>(req->prompt_tokens.size());
                if (prompt_len >= PrefixCacheManager::BLOCK_SIZE)
                {
                    KVSnapshot snap = runner_.GetKVSnapshot(prompt_len);
                    if (!snap.empty())
                        prefix_cache_.Store(req->prompt_tokens, std::move(snap));
                }

                req->status = RequestStatus::Done;
                req->result_promise.set_value(std::move(output));
            }
            catch (...)
            {
                req->status = RequestStatus::Failed;
                req->result_promise.set_exception(std::current_exception());
            }
        }
    }
}
