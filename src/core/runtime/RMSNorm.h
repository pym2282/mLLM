#pragma once

#include <torch/torch.h>
#include <stdexcept>

namespace mllm
{
    // Llama-style RMSNorm.
    //
    //   y = x * weight / sqrt(mean(x^2) + eps)
    //
    // Reference (HF transformers, LlamaRMSNorm):
    //   variance = x.pow(2).mean(-1, keepdim=True)
    //   x = x * rsqrt(variance + eps)
    //   return (weight * x).to(input_dtype)
    //
    // Stateless pure function — reusable across Llama / Qwen / Mistral etc.
    // Internal computation runs in float32 for numerical parity with HF,
    // then casts back to the input dtype.
    class RMSNorm
    {
    public:
        // x:      [..., hidden_size]         any float dtype (bf16/f16/f32)
        // weight: [hidden_size]              any float dtype
        // eps:    variance epsilon (from config.rms_norm_eps)
        // out:    [..., hidden_size]         same dtype as x
        static torch::Tensor Forward(
            const torch::Tensor& x,
            const torch::Tensor& weight,
            double eps)
        {
            if (!x.defined() || !weight.defined())
            {
                throw std::runtime_error("RMSNorm: input or weight undefined.");
            }
            if (weight.dim() != 1)
            {
                throw std::runtime_error("RMSNorm: weight must be 1D.");
            }
            if (x.size(-1) != weight.size(0))
            {
                throw std::runtime_error(
                    "RMSNorm: last dim of x must match weight length.");
            }

            const auto input_dtype = x.scalar_type();

            auto x_f32 = x.to(torch::kFloat32);
            auto w_f32 = weight.to(torch::kFloat32);

            auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
            auto x_norm = x_f32 * torch::rsqrt(variance + eps);

            auto out = w_f32 * x_norm;

            return out.to(input_dtype);
        }
    };
}
