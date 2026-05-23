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

            std::cerr
                << "[Scheduler] Processing: "
                << req->request_id
                << std::endl;

            try
            {
                GenerateResult output = runner_.Generate(
                    req->prompt_tokens,
                    req->options
                );

                req->status = RequestStatus::Done;
                req->result_promise.set_value(std::move(output));
            }
            catch (...)
            {
                req->status = RequestStatus::Failed;
                req->result_promise.set_exception(
                    std::current_exception()
                );
            }
        }
    }
}
