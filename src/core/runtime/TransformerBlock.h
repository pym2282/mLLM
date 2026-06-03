#pragma once

#include <torch/torch.h>

#include "core/runtime/KVCache.h"
#include "core/runtime/RMSNorm.h"
#include "core/runtime/Attention.h"
#include "core/runtime/MLP.h"
#include "core/runtime/Linear.h"

namespace mllm
{
    // Weights for one Gated DeltaNet (linear-attention) layer in Qwen3.5
    struct LinearAttnWeights
    {
        torch::Tensor in_proj_qkv;  // [conv_dim, hidden]
        torch::Tensor in_proj_z;    // [value_dim, hidden]
        torch::Tensor in_proj_a;    // [num_v_heads, hidden]
        torch::Tensor in_proj_b;    // [num_v_heads, hidden]
        torch::Tensor conv1d;       // [conv_dim, kernel_size]  (depthwise weights)
        torch::Tensor dt_bias;      // [num_v_heads]
        torch::Tensor A_log;        // [num_v_heads] GGUF ssm_a: converted negative decay coefficient
        torch::Tensor norm;         // [head_v_dim]  (gated RMSNorm weight)
        torch::Tensor out_proj;     // [hidden, value_dim]
    };

    // Rolling state cache for one Gated DeltaNet layer
    struct SSMCache
    {
        torch::Tensor conv_state;       // [B, conv_dim, kernel_size]
        torch::Tensor recurrent_state;  // [B, num_v_heads, head_k_dim, head_v_dim]  (float32)
        int len = 0;

        bool IsInitialized() const
        {
            return conv_state.defined() && recurrent_state.defined();
        }

        void Clear()
        {
            len = 0;
            if (conv_state.defined())
                conv_state.zero_();
            if (recurrent_state.defined())
                recurrent_state.zero_();
        }

        void Allocate(int batch_size, int conv_dim, int kernel_size,
                      int num_v_heads, int head_k_dim, int head_v_dim,
                      torch::Device device, torch::ScalarType dtype)
        {
            conv_state = torch::zeros(
                {batch_size, conv_dim, kernel_size},
                torch::TensorOptions().device(device).dtype(dtype));
            recurrent_state = torch::zeros(
                {batch_size, num_v_heads, head_k_dim, head_v_dim},
                torch::TensorOptions().device(device).dtype(torch::kFloat32));
        }
    };

    struct LayerWeights
    {
        torch::Tensor input_layernorm;
        torch::Tensor post_attention_layernorm;

        // Gemma 4: post-attention norm (applied to attention output before residual add)
        torch::Tensor post_attn_norm;
        // Gemma 4: post-FFN norm (applied to FFN output before residual add)
        torch::Tensor post_ffn_norm;
        // Gemma 4: Per-Layer Input (AltUP)
        torch::Tensor w_per_layer_inp_gate;  // [D_ple, H]
        torch::Tensor w_per_layer_proj;      // [H, D_ple]
        torch::Tensor per_layer_post_norm;   // [H]
        torch::Tensor layer_scalar;          // [1] scalar multiplier

        // -----------------------------
        // Attention
        // -----------------------------
        torch::Tensor w_q;
        torch::Tensor w_k;
        torch::Tensor w_v;
        torch::Tensor w_o;

        // Qwen3 QK-Norm (undefined for Qwen2/Llama)
        torch::Tensor w_q_norm;
        torch::Tensor w_k_norm;

        // Qwen2/2.5 attention projection biases (undefined for Qwen3/Llama)
        torch::Tensor b_q;
        torch::Tensor b_k;
        torch::Tensor b_v;

        // -----------------------------
        // MLP (SwiGLU)
        // -----------------------------
        torch::Tensor w_gate;
        torch::Tensor w_up;
        torch::Tensor w_down;

        // FP8 scale tensors (undefined for FP16/BF16 models)
        torch::Tensor w_q_scale;
        torch::Tensor w_k_scale;
        torch::Tensor w_v_scale;
        torch::Tensor w_o_scale;
        torch::Tensor w_gate_scale;
        torch::Tensor w_up_scale;
        torch::Tensor w_down_scale;
    };

    class TransformerBlock
    {
    public:
        static __declspec(noinline) torch::Tensor Forward(
            const torch::Tensor& hidden,
            const LayerWeights& lw,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            double rope_theta,
            double rms_norm_eps,
            bool use_qk_norm,
            const torch::Tensor& position_ids,
            KVCache* kv_cache,
            int rope_dim = 0
        )
        {
            // =====================================================
            // Attention Block
            // =====================================================

            auto residual = hidden.clone();

            auto h = RMSNorm::Forward(
                hidden,
                lw.input_layernorm,
                rms_norm_eps
            );

            // Qwen3 path
            if (use_qk_norm)
            {
                auto q_raw = Linear::Forward(h, lw.w_q);
                const auto B   = q_raw.size(0);
                const auto Sq  = q_raw.size(1);  // seq len from [B, S, proj_dim]
                const int64_t q_dim = static_cast<int64_t>(num_heads) * head_dim;

                torch::Tensor q, out_gate;

                if (q_raw.size(2) == 2 * q_dim)
                {
                    // Qwen3.5 stores [q, gate] per head, not [all q, all gate].
                    auto qg = q_raw.view({B, Sq, num_heads, 2 * head_dim});
                    q = qg.narrow(3, 0, head_dim).transpose(1, 2);
                    out_gate = qg.narrow(3, head_dim, head_dim)
                                   .reshape({B, Sq, q_dim});
                }
                else
                {
                    q = q_raw.view({B, Sq, num_heads, head_dim}).transpose(1, 2);
                }

                // RMSNorm must happen AFTER reshape (weight shape == [head_dim])
                q = RMSNorm::Forward(q, lw.w_q_norm, rms_norm_eps);

                auto k = Linear::Forward(h, lw.w_k);
                k = k.view({B, Sq, num_kv_heads, head_dim}).transpose(1, 2);
                k = RMSNorm::Forward(k, lw.w_k_norm, rms_norm_eps);

                auto v = Linear::Forward(h, lw.w_v);
                v = v.view({B, Sq, num_kv_heads, head_dim}).transpose(1, 2);

                h = Attention::ForwardProjected(
                    q,
                    k,
                    v,
                    lw.w_o,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rope_theta,
                    position_ids,
                    kv_cache,
                    out_gate,
                    rope_dim
                );
            }
            else
            {
                // Llama / Qwen2 path (optional biases on q/k/v)
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
                    kv_cache,
                    lw.b_q,
                    lw.b_k,
                    lw.b_v,
                    rope_dim
                );
            }
            h = residual + h;

            // =====================================================
            // MLP Block
            // =====================================================

            residual = h.clone();

            h = RMSNorm::Forward(h, lw.post_attention_layernorm, rms_norm_eps);

            h = MLP::Forward(h, lw.w_gate, lw.w_up, lw.w_down);

            return residual + h;
        }
    };
}
