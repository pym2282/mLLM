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
        // positions: [S] int64 (or any int type castable to fp32)
        // head_dim:  D, must be even
        // rope_theta: base frequency (e.g. 10000.0 for Llama)
        // Returns (cos, sin) both shape [S, D], fp32, on same device as positions.
        static std::pair<torch::Tensor, torch::Tensor> BuildCosSin(
            const torch::Tensor& positions,
            int head_dim,
            double rope_theta)
        {
            if (head_dim % 2 != 0)
            {
                throw std::runtime_error("RoPE: head_dim must be even.");
            }

            const auto device = positions.device();
            auto fopts = torch::TensorOptions()
                .dtype(torch::kFloat32).device(device);

            // idx: [D/2] = 0, 2, 4, ..., D-2
            auto idx = torch::arange(0, head_dim, 2, fopts);
            // inv_freq = 1 / theta^(idx/D) = exp(-idx/D * log(theta))
            // exp/log form avoids torch::pow(Scalar, Tensor) overload
            // resolution issues across libtorch versions.
            const double log_theta = std::log(rope_theta);
            auto inv_freq =
                (idx / static_cast<double>(head_dim) * log_theta).neg().exp();

            auto pos_f32 = positions.to(torch::kFloat32);
            // freqs: [S, D/2] = outer(pos, inv_freq) via broadcast
            auto freqs = pos_f32.unsqueeze(-1) * inv_freq.unsqueeze(0);
            // emb: [S, D]
            auto emb = torch::cat({freqs, freqs}, -1);

            return { emb.cos(), emb.sin() };
        }

        // Apply rotary to x.
        //   x:   [B, H, S, D]
        //   cos: [S, D]  fp32
        //   sin: [S, D]  fp32
        //   out: [B, H, S, D]  same dtype as x
        //
        // Computes in fp32 for numerical parity with HF, casts back at end.
        static torch::Tensor Apply(
            const torch::Tensor& x,
            const torch::Tensor& cos,
            const torch::Tensor& sin)
        {
            const auto in_dtype = x.scalar_type();
            const auto D = x.size(-1);
            const auto half = D / 2;

            auto x_f32 = x.to(torch::kFloat32);
            // [S, D] -> [1, 1, S, D] so it broadcasts across (B, H).
            auto cos_b = cos.unsqueeze(0).unsqueeze(0);
            auto sin_b = sin.unsqueeze(0).unsqueeze(0);

            // rotate_half(x) = cat([-x2, x1], -1)
            auto x1 = x_f32.slice(-1, 0, half);
            auto x2 = x_f32.slice(-1, half, D);
            auto rotated = torch::cat({-x2, x1}, -1);

            auto out = x_f32 * cos_b + rotated * sin_b;
            return out.to(in_dtype);
        }
    };
}
