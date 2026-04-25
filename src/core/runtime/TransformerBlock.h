#pragma once

#include <torch/torch.h>

#include "core/runtime/RMSNorm.h"
#include "core/runtime/Attention.h"
#include "core/runtime/MLP.h"

namespace mllm
{
    // Tensor handles for a single decoder layer (pre-norm Llama style).
    // Holds torch::Tensor by value — torch::Tensor is a reference-counted
    // handle, so copies share storage and are cheap.
    struct LayerWeights
    {
        torch::Tensor input_layernorm;            // [hidden]
        torch::Tensor post_attention_layernorm;   // [hidden]

        // Attention projections
        torch::Tensor w_q;   // [num_heads * head_dim, hidden]
        torch::Tensor w_k;   // [num_kv_heads * head_dim, hidden]
        torch::Tensor w_v;   // [num_kv_heads * head_dim, hidden]
        torch::Tensor w_o;   // [hidden, num_heads * head_dim]

        // MLP (SwiGLU)
        torch::Tensor w_gate;  // [intermediate, hidden]
        torch::Tensor w_up;    // [intermediate, hidden]
        torch::Tensor w_down;  // [hidden, intermediate]
    };

    // One pre-norm Llama decoder layer:
    //
    //   residual = x
    //   x = RMSNorm(x, input_layernorm)
    //   x = Attention(x)
    //   x = residual + x
    //
    //   residual = x
    //   x = RMSNorm(x, post_attention_layernorm)
    //   x = MLP(x)
    //   x = residual + x
    //
    // Stateless. Purely consumes a LayerWeights view — keeps the block
    // architecture independent of how weights were loaded or named,
    // which lets Qwen / Mistral reuse this unchanged.
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
            const torch::Tensor& position_ids)
        {
            // Attention sub-block
            auto residual = hidden;
            auto h = RMSNorm::Forward(
                hidden, lw.input_layernorm, rms_norm_eps);
            h = Attention::Forward(
                h,
                lw.w_q, lw.w_k, lw.w_v, lw.w_o,
                num_heads, num_kv_heads, head_dim,
                rope_theta, position_ids);
            h = residual + h;

            // MLP sub-block
            residual = h;
            h = RMSNorm::Forward(
                h, lw.post_attention_layernorm, rms_norm_eps);
            h = MLP::Forward(h, lw.w_gate, lw.w_up, lw.w_down);
            h = residual + h;

            return h;
        }
    };
}
