#pragma once

namespace mllm
{
    struct GenerateOptions
    {
        int max_new_tokens = 32;

        float temperature = 1.0f;

        int top_k = 40;

        bool use_greedy = false;

        // repetition penalty
        // 1.0 = disabled
        // 1.05 ~ 1.2 추천
        float repetition_penalty = 1.1f;
    };
}