// src/core/runtime/Sampler.cpp

#include "Sampler.h"

#include <stdexcept>

namespace mllm
{
    void Sampler::ApplyRepetitionPenalty(
        torch::Tensor& logits,
        const std::vector<int64_t>& previous_tokens,
        float penalty
    )
    {
        if (penalty <= 1.0f)
        {
            return;
        }

        for (auto token_id : previous_tokens)
        {
            if (token_id < 0 || token_id >= logits.size(0))
            {
                continue;
            }

            float value =
                logits[token_id].item<float>();

            if (value < 0.0f)
            {
                logits[token_id] =
                    value * penalty;
            }
            else
            {
                logits[token_id] =
                    value / penalty;
            }
        }
    }

    int64_t Sampler::Sample(
        torch::Tensor logits,
        float temperature,
        int top_k,
        float top_p,
        bool use_greedy,
        const std::vector<int64_t>& previous_tokens,
        float repetition_penalty
    )
    {
        if (logits.dim() != 1)
        {
            throw std::runtime_error(
                "Sampler expects 1D logits tensor."
            );
        }

        // repetition penalty 먼저 적용
        ApplyRepetitionPenalty(
            logits,
            previous_tokens,
            repetition_penalty
        );

        // greedy mode
        if (use_greedy || temperature <= 0.0f)
        {
            return torch::argmax(
                logits,
                -1
            ).item<int64_t>();
        }

        auto probs =
            torch::softmax(
                logits / temperature,
                -1
            );

        // top-k filtering
        if (top_k > 0)
        {
            auto topk =
                torch::topk(
                    probs,
                    top_k,
                    -1
                );

            auto filtered_probs =
                torch::zeros_like(probs);

            filtered_probs.scatter_(
                -1,
                std::get<1>(topk),
                std::get<0>(topk)
            );

            probs =
                filtered_probs /
                filtered_probs.sum();
        }

        // --------------------------------
        // top-p (nucleus sampling)
        // --------------------------------

        if (top_p < 1.0f)
        {
            auto sorted = torch::sort(
                probs,
                -1,
                /*descending=*/true
            );

            auto sorted_probs = std::get<0>(sorted);
            auto sorted_indices = std::get<1>(sorted);

            auto cumulative_probs =
                torch::cumsum(
                    sorted_probs,
                    -1
                );

            auto remove_mask =
                cumulative_probs > top_p;

            // 첫 token은 항상 유지
            remove_mask[0] = false;

            sorted_probs.masked_fill_(
                remove_mask,
                0
            );

            auto filtered_probs =
                torch::zeros_like(probs);

            filtered_probs.scatter_(
                -1,
                sorted_indices,
                sorted_probs
            );

            probs =
                filtered_probs /
                filtered_probs.sum();
        }

        auto next_token =
            torch::multinomial(
                probs,
                1
            );

        return next_token.item<int64_t>();
    }
}