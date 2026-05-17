// src/models/base/GenerateResult.h

#pragma once

#include <cstdint>
#include <vector>

namespace mllm
{
    enum class FinishReason
    {
        EOS,    // eos_token_id was generated
        Length, // max_new_tokens reached or context window full
        Stop,   // matched a stop_sequence_ids entry
    };

    struct GenerateResult
    {
        std::vector<int64_t> tokens;
        FinishReason finish_reason = FinishReason::Length;
    };
}
