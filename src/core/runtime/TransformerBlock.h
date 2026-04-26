// src/core/runtime/TransformerBlock.h

#pragma once

#include <torch/torch.h>

#include "core/runtime/KVCache.h"
#include "core/runtime/RMSNorm.h"
#include "core/runtime/Attention.h"
#include "core/runtime/MLP.h"

namespace mllm
{
    struct LayerWeights
    {
        torch::Tensor input_layernorm;
        torch::Tensor post_attention_layernorm;

        // Attention
        torch::Tensor w_q;
        torch::Tensor w_k;
        torch::Tensor w_v;
        torch::Tensor w_o;

        // MLP (SwiGLU)
        torch::Tensor w_gate;
        torch::Tensor w_up;
        torch::Tensor w_down;
    };

    class TransformerBlock
    {
    public:
        static torch::Tensor Forward(
            const torch::Tensor& hidden,
            const LayerWeights& lw,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            double rope_theta,
            double rms_norm_eps,
            const torch::Tensor& position_ids,
            KVCache* kv_cache
        )
        {
            // -----------------------------
            // Attention block
            // -----------------------------
            auto residual = hidden;

            auto h = RMSNorm::Forward(
                hidden,
                lw.input_layernorm,
                rms_norm_eps
            );

            h = Attention::Forward(
                h,
                lw.w_q,
                lw.w_k,
                lw.w_v,
                lw.w_o,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_theta,
                position_ids,
                kv_cache
            );

            h = residual + h;

            // -----------------------------
            // MLP block
            // -----------------------------
            residual = h;

            h = RMSNorm::Forward(
                h,
                lw.post_attention_layernorm,
                rms_norm_eps
            );

            h = MLP::Forward(
                h,
                lw.w_gate,
                lw.w_up,
                lw.w_down
            );

            h = residual + h;

            return h;
        }
    };
}