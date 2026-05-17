#pragma once

#include <torch/torch.h>

#include "core/runtime/KVCache.h"
#include "core/runtime/RMSNorm.h"
#include "core/runtime/Attention.h"
#include "core/runtime/MLP.h"
#include "core/runtime/Linear.h"

namespace mllm
{
    struct LayerWeights
    {
        torch::Tensor input_layernorm;
        torch::Tensor post_attention_layernorm;

        // -----------------------------
        // Attention
        // -----------------------------
        torch::Tensor w_q;
        torch::Tensor w_k;
        torch::Tensor w_v;
        torch::Tensor w_o;

        // Qwen3 QK-Norm
        // Llama에서는 비어있음
        torch::Tensor w_q_norm;
        torch::Tensor w_k_norm;

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
        static torch::Tensor Forward(
            const torch::Tensor& hidden,
            const LayerWeights& lw,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            double rope_theta,
            double rms_norm_eps,
            bool use_qk_norm,
            const torch::Tensor& position_ids,
            KVCache* kv_cache
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
                // --------------------------------
                // Q projection
                // --------------------------------
                auto q = Linear::Forward(
                    h,
                    lw.w_q
                );

                // q:
                // [B, S, hidden]
                // ->
                // [B, num_heads, S, head_dim]
                q = q.view({
                    q.size(0),
                    q.size(1),
                    num_heads,
                    head_dim
                }).transpose(1, 2);

                // IMPORTANT:
                // Qwen q_norm weight shape == [head_dim]
                // not [hidden_size]
                //
                // so RMSNorm must happen AFTER reshape
                q = RMSNorm::Forward(
                    q,
                    lw.w_q_norm,
                    rms_norm_eps
                );

                // --------------------------------
                // K projection
                // --------------------------------
                auto k = Linear::Forward(
                    h,
                    lw.w_k
                );

                // k:
                // [B, S, kv_hidden]
                // ->
                // [B, num_kv_heads, S, head_dim]
                k = k.view({
                    k.size(0),
                    k.size(1),
                    num_kv_heads,
                    head_dim
                }).transpose(1, 2);

                // IMPORTANT:
                // same for k_norm
                k = RMSNorm::Forward(
                    k,
                    lw.w_k_norm,
                    rms_norm_eps
                );

                // --------------------------------
                // V projection
                // --------------------------------
                auto v = Linear::Forward(
                    h,
                    lw.w_v
                );

                // v:
                // [B, S, kv_hidden]
                // ->
                // [B, num_kv_heads, S, head_dim]
                v = v.view({
                    v.size(0),
                    v.size(1),
                    num_kv_heads,
                    head_dim
                }).transpose(1, 2);

                // --------------------------------
                // ForwardProjected expects:
                //
                // q: [B, H, S, D]
                // k: [B, KV, S, D]
                // v: [B, KV, S, D]
                //
                // so reshape again inside
                // should NOT happen there
                // (must be adjusted in Attention.h)
                // --------------------------------
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
                    kv_cache
                );
            }
            else
            {
                // Original Llama path
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
            }
            h = residual + h;

            // =====================================================
            // MLP Block
            // =====================================================

            residual = h.clone();

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