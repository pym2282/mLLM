#pragma once

#include <torch/torch.h>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "KVCache.h"
#include "core/runtime/Linear.h"
#include "core/runtime/RoPE.h"

namespace mllm
{
    class Attention
    {
    public:
        // =========================================================
        // Original path (Llama)
        //
        // hidden
        // -> q_proj / k_proj / v_proj
        // -> RoPE
        // -> KV cache
        // -> GQA
        // -> Attention
        // -> o_proj
        // =========================================================

        static torch::Tensor Forward(
            const torch::Tensor& hidden,
            const torch::Tensor& w_q,
            const torch::Tensor& w_k,
            const torch::Tensor& w_v,
            const torch::Tensor& w_o,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            double rope_theta,
            const torch::Tensor& position_ids,
            KVCache* kv_cache)
        {
            if (hidden.dim() != 3)
            {
                throw std::runtime_error(
                    "Attention: hidden must be [B, S, H]."
                );
            }

            auto q = Linear::Forward(hidden, w_q);
            auto k = Linear::Forward(hidden, w_k);
            auto v = Linear::Forward(hidden, w_v);

            return ForwardProjected(
                q,
                k,
                v,
                w_o,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_theta,
                position_ids,
                kv_cache
            );
        }

        // =========================================================
        // Qwen3 path
        //
        // projected q/k/v already prepared:
        //
        // q_proj -> q_norm
        // k_proj -> k_norm
        // v_proj
        //
        // TransformerBlock에서 호출
        // =========================================================

        static torch::Tensor ForwardProjected(
            torch::Tensor q,
            torch::Tensor k,
            torch::Tensor v,
            const torch::Tensor& w_o,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            double rope_theta,
            const torch::Tensor& position_ids,
            KVCache* kv_cache)
        {
            // =========================================================
            // Expected input shapes (already projected + reshaped)
            //
            // q : [B, num_heads,     S, head_dim]
            // k : [B, num_kv_heads,  S, head_dim]
            // v : [B, num_kv_heads,  S, head_dim]
            //
            // NOTE:
            // Qwen3 path already does:
            //
            // q_proj -> reshape -> q_norm
            // k_proj -> reshape -> k_norm
            // v_proj -> reshape
            //
            // So ForwardProjected must NOT reshape again.
            // =========================================================

            if (q.dim() != 4 ||
                k.dim() != 4 ||
                v.dim() != 4)
            {
                throw std::runtime_error(
                    "Attention::ForwardProjected: "
                    "q/k/v must be rank-4 tensors."
                );
            }

            if (num_heads % num_kv_heads != 0)
            {
                throw std::runtime_error(
                    "Attention: num_heads must be divisible by num_kv_heads."
                );
            }

            const auto B = q.size(0);
            const auto S = q.size(2); // IMPORTANT: [B, H, S, D]
            const auto in_dtype = q.scalar_type();

            const int n_rep =
                num_heads / num_kv_heads;

            // =====================================================
            // RoPE
            // =====================================================

            auto cs =
                RoPE::BuildCosSin(
                    position_ids,
                    head_dim,
                    rope_theta
                );

            auto& rope_cos = cs.first;
            auto& rope_sin = cs.second;

            q = RoPE::Apply(
                q,
                rope_cos,
                rope_sin
            );

            k = RoPE::Apply(
                k,
                rope_cos,
                rope_sin
            );

            // =====================================================
            // KV Cache
            // =====================================================

            if (kv_cache != nullptr)
            {
                if (kv_cache->IsInitialized())
                {
                    // concat on sequence dim
                    // shape: [B, H, S, D]
                    k = torch::cat(
                        {
                            kv_cache->key,
                            k
                        },
                        2
                    );

                    v = torch::cat(
                        {
                            kv_cache->value,
                            v
                        },
                        2
                    );
                }

                kv_cache->key = k;
                kv_cache->value = v;
            }

            // =====================================================
            // GQA
            //
            // Expand KV heads:
            //
            // [B, KV, S, D]
            // ->
            // [B, H, S, D]
            // =====================================================

            if (n_rep > 1)
            {
                k = k.repeat_interleave(
                    n_rep,
                    1
                );

                v = v.repeat_interleave(
                    n_rep,
                    1
                );
            }

            // =====================================================
            // Scaled Dot Product Attention
            // =====================================================

            auto q_f32 =
                q.to(torch::kFloat32);

            auto k_f32 =
                k.to(torch::kFloat32);

            const double scale =
                1.0 /
                std::sqrt(
                    static_cast<double>(
                        head_dim
                    )
                );

            auto scores =
                torch::matmul(
                    q_f32,
                    k_f32.transpose(-2, -1)
                ) * scale;

            // =====================================================
            // Causal Mask
            // =====================================================

            const auto total_seq =
                k.size(2);

            auto mask_opts =
                torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(q.device());

            auto mask =
                torch::triu(
                    torch::full(
                        {
                            S,
                            total_seq
                        },
                        -std::numeric_limits<float>::infinity(),
                        mask_opts
                    ),
                    1 + (total_seq - S)
                );

            scores =
                scores + mask;

            auto attn =
                torch::softmax(
                    scores,
                    -1
                ).to(in_dtype);

            // =====================================================
            // Weighted Sum
            // =====================================================

            auto out =
                torch::matmul(
                    attn,
                    v
                );

            // =====================================================
            // Merge heads
            //
            // [B, H, S, D]
            // ->
            // [B, S, H * D]
            // =====================================================

            out =
                out.transpose(1, 2)
                   .contiguous()
                   .view({
                        B,
                        S,
                        num_heads * head_dim
                    });

            // =====================================================
            // Output projection
            // =====================================================

            return Linear::Forward(
                out,
                w_o
            );
        }
    };
}