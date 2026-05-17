// src/models/base/GenerateOptions.h

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace mllm
{
    struct GenerateOptions
    {
        int max_new_tokens = 512;

        float temperature = 1.0f;

        int top_k = 40;

        // nucleus sampling — 0.8~0.95 typical; 1.0 = disabled
        float top_p = 0.9f;

        bool use_greedy = false;

        float repetition_penalty = 1.1f;

        int64_t eos_token_id = 2;

        // each inner vector is one stop sequence; generation halts on any match
        std::vector<std::vector<int64_t>> stop_sequence_ids;

        // Qwen3: false inserts <think></think> prefix so model skips reasoning
        bool enable_thinking = false;

        // streaming: called per token; return false to abort generation
        std::function<bool(int64_t)> on_token;
    };
}