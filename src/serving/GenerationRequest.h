// src/serving/GenerationRequest.h

#pragma once

#include "models/base/GenerateOptions.h"
#include <atomic>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

namespace mllm
{
    enum class RequestStatus
    {
        Pending,
        Running,
        Done,
        Failed
    };

    struct GenerationRequest
    {
        std::string                        request_id;
        std::vector<int64_t>               prompt_tokens;
        GenerateOptions                    options;
        std::atomic<RequestStatus>         status = RequestStatus::Pending;
        std::promise<std::vector<int64_t>> result_promise;
    };
}
