#pragma once

#include <torch/torch.h>

#include "core/runtime/Linear.h"

namespace mllm
{
    // Llama-family SwiGLU MLP.
    //
    //   y = down_proj( silu(gate_proj(x)) * up_proj(x) )
    //
    // Reference (HF transformers, LlamaMLP):
    //   self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    //   where act_fn = nn.SiLU().
    //
    // dtype policy: run in input dtype (bf16/f16). SiLU and elementwise mul
    // do not require fp32 promotion for HF parity.
    //
    // All three projections are bias-free in Llama / Qwen / Mistral MLP, so
    // bias is omitted from the signature. Reintroduce via Linear::Forward's
    // optional bias argument if a future arch needs it.
    class MLP
    {
    public:
        // x:       [B, S, hidden_size]
        // w_gate:  [intermediate_size, hidden_size]
        // w_up:    [intermediate_size, hidden_size]
        // w_down:  [hidden_size, intermediate_size]
        // out:     [B, S, hidden_size]
        static torch::Tensor Forward(
            const torch::Tensor& x,
            const torch::Tensor& w_gate,
            const torch::Tensor& w_up,
            const torch::Tensor& w_down)
        {
            auto gate = Linear::Forward(x, w_gate);
            auto up   = Linear::Forward(x, w_up);
            // SiLU = x * sigmoid(x). Using primitives instead of torch::silu
            // for libtorch version portability.
            auto act  = gate * torch::sigmoid(gate) * up;
            return Linear::Forward(act, w_down);
        }
    };
}
