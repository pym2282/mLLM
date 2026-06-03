#pragma once

#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mllm
{
    // Llama-style Rotary Positional Embedding (NeoX / HF convention).
    //
    // Conventions (MUST match HF transformers LlamaRotaryEmbedding):
    //   - rotate_half: split last dim into two halves (first / second),
    //     NOT interleaved pairs. rotate_half(x) = cat([-x2, x1], -1).
    //   - cos/sin layout: freqs = outer(pos, inv_freq)   -> [S, D/2]
    //                     emb   = cat([freqs, freqs], -1) -> [S, D]
    //     (concat-doubled, not interleaved).
    //   - inv_freq = 1 / theta ^ (arange(0, D, 2) / D)
    //   - cos/sin computed in fp32, cast to q/k dtype at apply time.
    //
    // Stateless — reusable across Llama / Qwen / Mistral (same rotation style).
    class RoPE
    {
    public:
        // positions:  [S] int64 (or any int type castable to fp32)
        // rope_dim:   number of dims to rotate (must be even; < head_dim for partial RoPE)
        // rope_theta: base frequency (e.g. 10000.0 for Llama)
        // Returns (cos, sin) both shape [S, rope_dim], fp32, on same device as positions.
        static std::pair<torch::Tensor, torch::Tensor> BuildCosSin(
            const torch::Tensor& positions,
            int rope_dim,
            double rope_theta)
        {
            if (rope_dim % 2 != 0)
            {
                throw std::runtime_error("RoPE: rope_dim must be even.");
            }

            const auto device = positions.device();
            auto fopts = torch::TensorOptions()
                .dtype(torch::kFloat32).device(device);

            // idx: [rope_dim/2] = 0, 2, 4, ..., rope_dim-2
            auto idx = torch::arange(0, rope_dim, 2, fopts);
            // inv_freq = 1 / theta^(idx/rope_dim) — divisor must be rope_dim
            const double log_theta = std::log(rope_theta);
            auto inv_freq =
                (idx / static_cast<double>(rope_dim) * log_theta).neg().exp();

            auto pos_f32 = positions.to(torch::kFloat32);
            // freqs: [S, rope_dim/2]
            auto freqs = pos_f32.unsqueeze(-1) * inv_freq.unsqueeze(0);
            // emb: [S, rope_dim]
            auto emb = torch::cat({freqs, freqs}, -1);

            return { emb.cos(), emb.sin() };
        }

        // Apply rotary to x (partial or full RoPE).
        //   x:   [B, H, S, D]
        //   cos: [S, rope_dim]  fp32   (rope_dim <= D)
        //   sin: [S, rope_dim]  fp32
        //   out: [B, H, S, D]  same dtype as x
        //
        // The first rope_dim dimensions are rotated; the remaining D-rope_dim are
        // passed through unchanged (Qwen3.5 partial RoPE: rope_dim=64, D=256).
        // Computes in fp32 for numerical parity with HF, casts back at end.
        static torch::Tensor Apply(
            const torch::Tensor& x,
            const torch::Tensor& cos,
            const torch::Tensor& sin)
        {
            const auto in_dtype = x.scalar_type();
            const auto D       = x.size(-1);
            const auto rope_dim = cos.size(-1);  // auto-detected from cos shape
            const auto half    = rope_dim / 2;

            auto x_f32 = x.to(torch::kFloat32);
            // [S, rope_dim] -> [1, 1, S, rope_dim] for (B, H) broadcast
            auto cos_b = cos.unsqueeze(0).unsqueeze(0);
            auto sin_b = sin.unsqueeze(0).unsqueeze(0);

            // Slice rotated portion and pass-through portion
            auto x_rot  = x_f32.slice(-1, 0, rope_dim);
            auto x_pass = x_f32.slice(-1, rope_dim, D);

            // rotate_half: cat([-x2, x1], -1)
            auto x1 = x_rot.slice(-1, 0, half);
            auto x2 = x_rot.slice(-1, half, rope_dim);
            auto rotated = torch::cat({-x2, x1}, -1);

            auto x_rot_out = x_rot * cos_b + rotated * sin_b;
            return torch::cat({x_rot_out, x_pass}, -1).to(in_dtype);
        }
    };
}
