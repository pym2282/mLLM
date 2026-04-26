// src/core/runtime/Sampler.h

#pragma once

#include <torch/torch.h>
#include <vector>
#include <cstdint>

namespace mllm
{
    class Sampler
    {
    public:
        static int64_t Sample(
            torch::Tensor logits,
            float temperature,
            int top_k,
            float top_p, // NEW
            bool use_greedy,
            const std::vector<int64_t>& previous_tokens,
            float repetition_penalty
        );

    private:
        static void ApplyRepetitionPenalty(
            torch::Tensor& logits,
            const std::vector<int64_t>& previous_tokens,
            float penalty
        );
    };
}