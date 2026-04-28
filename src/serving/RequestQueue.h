// src/serving/RequestQueue.h

#pragma once

#include "serving/GenerationRequest.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace mllm
{
    class RequestQueue
    {
    public:
        void Push(std::shared_ptr<GenerationRequest> req)
        {
            if (!req)
            {
                throw std::invalid_argument("RequestQueue cannot push null request.");
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (shutdown_)
                {
                    throw std::runtime_error("RequestQueue is shut down.");
                }
                queue_.push(std::move(req));
            }
            cv_.notify_one();
        }

        // Blocks until a request is available or Shutdown() is called.
        // Returns nullptr on shutdown with an empty queue.
        std::shared_ptr<GenerationRequest> Pop()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !queue_.empty() || shutdown_;
            });

            if (shutdown_ && queue_.empty())
                return nullptr;

            auto req = std::move(queue_.front());
            queue_.pop();
            return req;
        }

        void Shutdown()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                shutdown_ = true;
            }
            cv_.notify_all();
        }

        size_t Size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

    private:
        std::queue<std::shared_ptr<GenerationRequest>> queue_;
        mutable std::mutex                             mutex_;
        std::condition_variable                        cv_;
        bool                                           shutdown_ = false;
    };
}
