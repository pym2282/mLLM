#pragma once

#include <torch/torch.h>
#include <stdexcept>

namespace mllm
{
    // Stateless Linear projection.
    //
    //   y = x @ weight.T + bias
    //
    // Matches HF safetensors layout where Linear weights are stored as
    // [out_features, in_features]. The transpose is handled internally by
    // torch::nn::functional::linear — do not transpose manually.
    //
    // dtype policy:
    //   No fp32 upcast. HF runs Linear matmuls in the input dtype (bf16/f16),
    //   so upcasting here would diverge from HF numerical parity. The fp32
    //   cast in RMSNorm is specific to variance reduction stability and
    //   should not be copied into projection ops.
    //
    // bias:
    //   Optional. Llama-family projections are bias-free (pass {}), but
    //   Qwen2 adds bias on Q/K/V — the parameter is kept so Attention.h can
    //   serve both architectures without modification.
    class Linear
    {
    public:
        // x:      [..., in_features]
        // weight: [out_features, in_features]
        // bias:   [out_features] or undefined
        // out:    [..., out_features]
        static torch::Tensor Forward(
            const torch::Tensor& x,
            const torch::Tensor& weight,
            const torch::Tensor& bias = {})
        {
            if (!x.defined() || !weight.defined())
            {
                throw std::runtime_error("Linear: input or weight undefined.");
            }
            if (weight.dim() != 2)
            {
                throw std::runtime_error("Linear: weight must be 2D.");
            }
            if (x.size(-1) != weight.size(1))
            {
                throw std::runtime_error(
                    "Linear: last dim of x must match weight in_features.");
            }
            if (bias.defined())
            {
                if (bias.dim() != 1 || bias.size(0) != weight.size(0))
                {
                    throw std::runtime_error(
                        "Linear: bias must be 1D with length == out_features.");
                }
            }

            return torch::nn::functional::linear(x, weight, bias);
        }
    };
}
