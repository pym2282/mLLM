// src/models/base/GenerateOptions.h

#pragma once

namespace mllm
{
    struct GenerateOptions
    {
        int max_new_tokens = 32;

        float temperature = 1.0f;

        int top_k = 40;

        // NEW
        // nucleus sampling
        // usually 0.8 ~ 0.95
        // 1.0 = disabled
        float top_p = 0.9f;

        bool use_greedy = false;

        float repetition_penalty = 1.1f;

        int64_t eos_token_id = 2;
    };
}