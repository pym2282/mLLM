#pragma once

#include <torch/torch.h>
#include <cmath>

namespace mllm
{
    // Gemma 1/2-style RMSNorm: y = x * (1 + w) / sqrt(mean(x^2) + eps)
    // Gemma 4-style RMSNorm: y = x * w / sqrt(mean(x^2) + eps)  (plain_weight=true)
    inline torch::Tensor GemmaRMSNorm(
        const torch::Tensor& x,
        const torch::Tensor& w,
        double eps,
        bool plain_weight = false)  // true for Gemma 4 (Gemma4RMSNorm: w not 1+w)
    {
        const auto dtype = x.scalar_type();
        const auto device = x.device();
        auto xf = x.to(torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto wf = w.to(torch::TensorOptions().dtype(torch::kFloat32).device(device));

        auto rms = torch::rsqrt(xf.pow(2).mean(-1, /*keepdim=*/true) + eps);
        if (plain_weight)
            return (xf * rms * wf).to(dtype);
        return (xf * rms * (1.0f + wf)).to(dtype);
    }

    // GeGLU: y = gelu(x @ w_gate.T) * (x @ w_up.T), then y @ w_down.T
    inline torch::Tensor GeGLU(
        const torch::Tensor& x,
        const torch::Tensor& w_gate,
        const torch::Tensor& w_up,
        const torch::Tensor& w_down)
    {
        auto gate = torch::nn::functional::gelu(
            torch::nn::functional::linear(x, w_gate),
            torch::nn::functional::GELUFuncOptions().approximate("tanh"));
        auto up   = torch::nn::functional::linear(x, w_up);
        return torch::nn::functional::linear(gate * up, w_down);
    }

    // Gemma 4 Per-Layer Input (AltUP) step — runs after attention+FFN each layer.
    // per_layer_emb: [B, S, D_ple] (already sliced and normalized for this layer)
    inline torch::Tensor PerLayerInputForward(
        const torch::Tensor& hidden,
        const torch::Tensor& per_layer_emb,  // [B, S, D_ple]
        const torch::Tensor& w_inp_gate,
        const torch::Tensor& w_proj,
        const torch::Tensor& post_norm_w,
        const torch::Tensor& layer_scalar,
        double eps,
        bool norm_plain_weight = false)
    {
        auto gate = torch::nn::functional::gelu(
            torch::nn::functional::linear(hidden, w_inp_gate),
            torch::nn::functional::GELUFuncOptions().approximate("tanh"));
        gate = gate * per_layer_emb;
        auto out = torch::nn::functional::linear(gate, w_proj);
        if (post_norm_w.defined())
            out = GemmaRMSNorm(out, post_norm_w, eps, norm_plain_weight);
        return hidden + out;
        (void)layer_scalar;
    }

    // Sliding window causal attention bias for prefill.
    // Position i can attend to positions [max(0, i - window + 1), i].
    // Returns [1, 1, S, S] additive bias (-inf for masked positions, 0 otherwise).
    inline torch::Tensor SlidingWindowMask(int64_t S, int window, torch::Device device)
    {
        // Causal mask base
        auto mask = torch::full(
            {S, S},
            -std::numeric_limits<float>::infinity(),
            torch::TensorOptions().dtype(torch::kFloat32).device(device));

        for (int64_t i = 0; i < S; ++i)
        {
            int64_t start = std::max(int64_t{0}, i - window + 1);
            mask.slice(0, i, i + 1).slice(1, start, i + 1).fill_(0.0f);
        }

        return mask.unsqueeze(0).unsqueeze(0); // [1, 1, S, S]
    }
}
