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
    // Llama-family Grouped-Query Attention (GQA).
    //
    //   hidden -> Q_proj, K_proj, V_proj
    //          -> reshape to [B, H, S, D]
    //          -> RoPE(Q), RoPE(K)
    //          -> repeat_interleave K/V along head dim (GQA)
    //          -> scores = Q @ K^T / sqrt(D)
    //          -> add causal mask (-inf on strict upper triangle)
    //          -> softmax (fp32, then cast back)
    //          -> attn @ V
    //          -> merge heads, O_proj
    //
    // dtype policy:
    //   - projections in input dtype (HF parity)
    //   - softmax in fp32, cast back to input dtype
    //
    // GQA note:
    //   num_kv_heads may be smaller than num_heads (e.g. TinyLlama 32/4).
    //   Use repeat_interleave along dim=1 so that K/V heads align 1:1 with
    //   attention heads. `repeat` would shuffle heads the wrong way — silent
    //   correctness bug.
    //
    // bias:
    //   Llama projections are bias-free. Qwen2 has bias on Q/K/V — Linear's
    //   optional bias arg can be threaded through here later if needed.
    //
    // No KV cache yet. This is the full-prefill path; KV cache integration
    // will extend the position_ids contract + split prefill vs decode.
    class Attention
    {
    public:
        // hidden:        [B, S, hidden_size]
        // w_q:           [num_heads    * head_dim, hidden_size]
        // w_k:           [num_kv_heads * head_dim, hidden_size]
        // w_v:           [num_kv_heads * head_dim, hidden_size]
        // w_o:           [hidden_size, num_heads * head_dim]
        // position_ids:  [S] int64
        // out:           [B, S, hidden_size]
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
                throw std::runtime_error("Attention: hidden must be [B, S, H].");
            }
            if (num_heads % num_kv_heads != 0)
            {
                throw std::runtime_error(
                    "Attention: num_heads must be divisible by num_kv_heads.");
            }

            const auto B = hidden.size(0);
            const auto S = hidden.size(1);
            const auto in_dtype = hidden.scalar_type();
            const int n_rep = num_heads / num_kv_heads;

            // ---- Projections -------------------------------------------------
            auto q = Linear::Forward(hidden, w_q);  // [B, S, H*D]
            auto k = Linear::Forward(hidden, w_k);  // [B, S, Hkv*D]
            auto v = Linear::Forward(hidden, w_v);  // [B, S, Hkv*D]

            // ---- Reshape to [B, H, S, D] ------------------------------------
            q = q.view({B, S, num_heads,    head_dim}).transpose(1, 2);
            k = k.view({B, S, num_kv_heads, head_dim}).transpose(1, 2);
            v = v.view({B, S, num_kv_heads, head_dim}).transpose(1, 2);

            // ---- RoPE on Q, K -----------------------------------------------
            auto cs = RoPE::BuildCosSin(position_ids, head_dim, rope_theta);
            auto& rope_cos = cs.first;
            auto& rope_sin = cs.second;
            q = RoPE::Apply(q, rope_cos, rope_sin);
            k = RoPE::Apply(k, rope_cos, rope_sin);

            // --------------------------------
            // KV Cache append + reuse
            // --------------------------------
            if (kv_cache != nullptr)
            {
                if (kv_cache->IsInitialized())
                {
                    // 기존 cache + 새 token concat
                    k = torch::cat(
                        {
                            kv_cache->key,
                            k
                        },
                        2 // seq_len dimension
                    );

                    v = torch::cat(
                        {
                            kv_cache->value,
                            v
                        },
                        2
                    );
                }

                // 최신 cache 저장
                kv_cache->key = k;
                kv_cache->value = v;
            }

            // ---- GQA: expand K, V to match num_heads ------------------------
            if (n_rep > 1)
            {
                k = k.repeat_interleave(n_rep, /*dim=*/1);
                v = v.repeat_interleave(n_rep, /*dim=*/1);
            }

            // ---- Scaled dot-product, causal mask, fp32 softmax -------------
            auto q_f32 = q.to(torch::kFloat32);
            auto k_f32 = k.to(torch::kFloat32);

            const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
            // scores: [B, H, S, S]
            auto scores = torch::matmul(q_f32, k_f32.transpose(-2, -1)) * scale;

            // Causal mask: -inf on strict upper triangle (j > i)
            auto mask_opts = torch::TensorOptions()
                .dtype(torch::kFloat32).device(hidden.device());
            auto mask = torch::triu(
                torch::full({S, S},
                    -std::numeric_limits<float>::infinity(), mask_opts),
                /*diagonal=*/1);
            scores = scores + mask;  // broadcasts over [B, H]

            auto attn = torch::softmax(scores, -1).to(in_dtype);

            // ---- Weighted sum of V ------------------------------------------
            auto out = torch::matmul(attn, v);  // [B, H, S, D]

            // ---- Merge heads and output projection --------------------------
            out = out.transpose(1, 2).contiguous()
                     .view({B, S, num_heads * head_dim});
            return Linear::Forward(out, w_o);
        }
    };
}
