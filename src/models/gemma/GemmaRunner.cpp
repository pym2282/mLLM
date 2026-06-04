// src/models/gemma/GemmaRunner.cpp

#include "models/gemma/GemmaRunner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "core/Logger.h"
#include "core/MllmException.h"

#include "models/base/GenerateResult.h"
#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"
#include "models/base/GgufLoader.h"
#include "models/base/GenerateOptions.h"
#include "models/base/BaseModelRunner.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/Sampler.h"
#include "core/runtime/Linear.h"
#include "core/runtime/Attention.h"
#include "core/runtime/RoPE.h"
#include "core/runtime/RMSNorm.h"
#include "core/runtime/GemmaOps.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace mllm
{
    bool GemmaRunner::Load(const std::string& model_path)
    {
        try
        {
            model_path_ = model_path;

            if (GgufLoader::IsGguf(model_path))
                return LoadGguf(model_path);

            if (!LoadConfig(model_path + "/config.json"))
            {
                MLLM_ERROR("GemmaRunner", "Failed to load config");
                return false;
            }

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
            {
                MLLM_ERROR("GemmaRunner", "Failed to parse safetensors header");
                return false;
            }

            LoadAllWeights();

            kv_caches_.clear();
            kv_caches_.resize(config_.num_layers);
            is_loaded_ = true;

            MLLM_INFO("GemmaRunner", "Loaded (safetensors) layers=" + std::to_string(config_.num_layers)
                + " window=" + std::to_string(config_.sliding_window_size));
            return true;
        }
        catch (const std::exception& e)
        {
            MLLM_ERROR("GemmaRunner", "Load failed: " + std::string(e.what()));
            is_loaded_ = false;
            return false;
        }
    }

    bool GemmaRunner::LoadGguf(const std::string& gguf_path)
    {
        config_  = GgufLoader::ReadConfig(gguf_path);
        weights_ = GgufLoader::Load(gguf_path);

        // Gemma 4: attention.key_length is the RoPE dimension, not the actual head_dim.
        // Derive actual head_dim from Q weight shape: head_dim = q_out_dim / num_heads
        {
            const std::string qkey = "model.layers.0.self_attn.q_proj.weight";
            auto it = weights_.find(qkey);
            if (it != weights_.end() && config_.num_attention_heads > 0)
            {
                const int q_out = static_cast<int>(it->second.size(0));
                const int derived = q_out / config_.num_attention_heads;
                if (derived != config_.head_dim)
                {
                    MLLM_WARN("GemmaRunner", "head_dim override: " + std::to_string(config_.head_dim)
                        + " → " + std::to_string(derived)
                        + " (from Q weight shape " + std::to_string(q_out) + "/" + std::to_string(config_.num_attention_heads) + ")");
                    config_.head_dim = derived;
                    // rope_dim cannot exceed head_dim
                    if (config_.rope_dim > config_.head_dim)
                        config_.rope_dim = config_.head_dim;
                }
            }
        }

        LoadAllWeights();

        kv_caches_.clear();
        kv_caches_.resize(config_.num_layers);

        // Gemma 4 uses Gemma4RMSNorm (output = norm(x)*w, not (1+w))
        // Detect by AltUP presence (hidden_size_per_layer_input > 0)
        norm_plain_weight_ = (config_.hidden_size_per_layer_input > 0);

        is_loaded_ = true;

        MLLM_INFO("GemmaRunner", "Loaded (GGUF) layers=" + std::to_string(config_.num_layers)
            + " window=" + std::to_string(config_.sliding_window_size)
            + " global_every=" + std::to_string(config_.full_attention_interval));
        return true;
    }


    void GemmaRunner::LoadLayerWeights()
    {
        const bool has_altup = (config_.hidden_size_per_layer_input > 0);

        layer_weights_.clear();
        layer_weights_.reserve(config_.num_layers);

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string p = "model.layers." + std::to_string(i);
            LayerWeights lw;

            lw.input_layernorm          = weights_.at(p + ".input_layernorm.weight");
            lw.post_attn_norm           = weights_.at(p + ".post_attn_norm.weight");
            lw.post_attention_layernorm = weights_.at(p + ".post_attention_layernorm.weight");
            lw.post_ffn_norm            = weights_.at(p + ".post_ffn_norm.weight");

            lw.w_q      = weights_.at(p + ".self_attn.q_proj.weight");
            lw.w_k      = weights_.at(p + ".self_attn.k_proj.weight");
            lw.w_v      = weights_.at(p + ".self_attn.v_proj.weight");
            lw.w_o      = weights_.at(p + ".self_attn.o_proj.weight");
            lw.w_q_norm = weights_.at(p + ".self_attn.q_norm.weight");
            lw.w_k_norm = weights_.at(p + ".self_attn.k_norm.weight");

            lw.w_gate = weights_.at(p + ".mlp.gate_proj.weight");
            lw.w_up   = weights_.at(p + ".mlp.up_proj.weight");
            lw.w_down = weights_.at(p + ".mlp.down_proj.weight");

            if (has_altup)
            {
                // Some layers (global attention) may not have AltUP weights — use find()
                auto try_get = [&](const std::string& k) -> torch::Tensor {
                    auto it = weights_.find(k);
                    return (it != weights_.end()) ? it->second : torch::Tensor{};
                };
                lw.w_per_layer_inp_gate = try_get(p + ".per_layer_inp_gate.weight");
                lw.w_per_layer_proj     = try_get(p + ".per_layer_proj.weight");
                lw.per_layer_post_norm  = try_get(p + ".per_layer_post_norm.weight");
                lw.layer_scalar         = try_get(p + ".layer_scalar.weight");
                if (!lw.layer_scalar.defined())
                    lw.layer_scalar = try_get(p + ".layer_scalar");  // no .weight suffix variant
            }

            layer_weights_.push_back(std::move(lw));
        }
    }

    void GemmaRunner::LoadAllWeights()
    {
        LoadWeight("model.embed_tokens.weight");
        LoadWeight("model.norm.weight");
        if (!config_.tie_word_embeddings)
            LoadWeight("lm_head.weight");

        // Gemma 4 Per-Layer Input: token embedding [vocab, num_layers * D_ple]
        {
            auto it = weights_.find("per_layer_token_embd.weight");
            if (it != weights_.end())
            {
                per_layer_token_embd_ = it->second;
                // D_ple per layer = total / num_layers
                if (config_.hidden_size_per_layer_input == 0 &&
                    per_layer_token_embd_.dim() >= 2 && config_.num_layers > 0)
                    config_.hidden_size_per_layer_input =
                        static_cast<int>(per_layer_token_embd_.size(1)) / config_.num_layers;
                MLLM_INFO("GemmaRunner", "per_layer_token_embd ["
                    + std::to_string(per_layer_token_embd_.size(0)) + ","
                    + std::to_string(per_layer_token_embd_.size(1)) + "]"
                    + " D_ple_per_layer=" + std::to_string(config_.hidden_size_per_layer_input));
            }
        }

        // Load additional per-layer model-level weights if present
        {
            auto find = [&](const std::string& k) -> torch::Tensor {
                auto it = weights_.find(k);
                return (it != weights_.end()) ? it->second : torch::Tensor{};
            };
            per_layer_model_proj_ = find("per_layer_model_proj.weight");
            per_layer_proj_norm_  = find("per_layer_proj_norm.weight");
        }

        // Move to CUDA before building layer_weights_ so all views are CUDA
        if (torch::cuda::is_available())
        {
            MLLM_INFO("GemmaRunner", "Moving weights to CUDA (async)...");
            auto to_cuda = [](torch::Tensor& t) {
                if (t.defined()) t = t.to(torch::kCUDA, /*non_blocking=*/true);
            };
            for (auto& [name, w] : weights_) to_cuda(w);
            to_cuda(per_layer_token_embd_);
            to_cuda(per_layer_model_proj_);
            to_cuda(per_layer_proj_norm_);
            torch::cuda::synchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
        }

        LoadLayerWeights();  // build layer_weights_ from (now CUDA) weights_

        const bool has_altup = (config_.hidden_size_per_layer_input > 0);
        MLLM_INFO("GemmaRunner", "Loaded " + std::to_string(weights_.size()) + " tensors, "
            + std::to_string(layer_weights_.size()) + " layers"
            + (has_altup ? " (AltUP enabled)" : ""));

        if (torch::cuda::is_available())
        {
            const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            MLLM_INFO("GemmaRunner", "VRAM=" + std::to_string(stats.reserved_bytes[0].current / (1024*1024)) + "MB");
        }
    }

    void GemmaRunner::MoveWeightsToCuda()
    {
        // No-op — LoadAllWeights now handles CUDA movement before building layer_weights_
    }

    bool GemmaRunner::IsGlobalLayer(int i) const
    {
        if (config_.full_attention_interval <= 0) return true; // all global
        // Global layers have larger head_dim (derived from K weight shape).
        // After head_dim override, config_.head_dim = local head_dim (256).
        // Global layers have kd > local head_dim.
        if (i < static_cast<int>(layer_weights_.size()) && layer_weights_[i].w_k.defined())
        {
            const int kd = static_cast<int>(layer_weights_[i].w_k.size(0)) / config_.num_key_value_heads;
            return (kd > config_.head_dim);  // global head_dim (512) > local head_dim (256)
        }
        return ((i + 1) % config_.full_attention_interval == 0);
    }

    torch::Tensor GemmaRunner::Forward(
        const torch::Tensor& input_ids_cpu,
        const torch::Tensor& /*attention_mask*/)
    {
        if (!is_loaded_) throw InferenceError("GemmaRunner: model not loaded.");

        // Move input_ids to the same device as weights
        const auto wdev = weights_.at("model.embed_tokens.weight").device();
        const auto input_ids = input_ids_cpu.to(wdev);

        const auto S = input_ids.size(1);

        // Position IDs
        torch::Tensor position_ids;
        // Use len>0 (not IsInitialized): pre-allocated caches are always "initialized"
        // (tensor is defined) but len=0 means no tokens written yet (= prefill, not decode).
        const bool is_decode = (S == 1) && !kv_caches_.empty() && kv_caches_[0].len > 0;

        if (is_decode)
        {
            position_ids = torch::tensor(
                {kv_caches_[0].len},
                torch::TensorOptions().dtype(torch::kInt64).device(wdev));
        }
        else
        {
            position_ids = torch::arange(
                0, S,
                torch::TensorOptions().dtype(torch::kInt64).device(wdev));
        }

        // Embedding + scale by sqrt(H) (Gemma 1/2/4 all use this)
        auto hidden = EmbeddingLookup::Forward(
            input_ids, weights_.at("model.embed_tokens.weight"));
        hidden = hidden * std::sqrt(static_cast<double>(config_.hidden_size));

        const double eps = config_.rms_norm_eps;
        const int D_ple = config_.hidden_size_per_layer_input;  // AltUP per-layer embedding dim

        // Gemma 4 Per-Layer Inputs: computed once before layer loop
        // per_layer_inputs[B, S, num_layers * D_ple]
        torch::Tensor per_layer_inputs;
        if (D_ple > 0 && per_layer_token_embd_.defined() && per_layer_model_proj_.defined())
        {
            // 1. Per-layer token embedding: [B, S, num_layers*D_ple], scaled by sqrt(D_ple)
            auto tok_embs = EmbeddingLookup::Forward(input_ids, per_layer_token_embd_);
            tok_embs = tok_embs * std::sqrt(static_cast<double>(D_ple));

            // 2. Model projection: hidden → [B, S, num_layers*D_ple], scaled by 1/sqrt(H)
            auto model_proj = Linear::Forward(hidden, per_layer_model_proj_);
            model_proj = model_proj * (1.0 / std::sqrt(static_cast<double>(config_.hidden_size)));

            // 3. Apply per_layer_proj_norm to projection slices
            if (per_layer_proj_norm_.defined())
            {
                const auto dtype = model_proj.scalar_type();
                auto B2 = model_proj.size(0); auto S2 = model_proj.size(1);
                auto chunks = model_proj.view({B2, S2, config_.num_layers, D_ple});
                auto xf = chunks.to(torch::kFloat32);
                auto wf = per_layer_proj_norm_.to(torch::kFloat32);
                auto rms = torch::rsqrt(xf.pow(2).mean(-1, true) + eps);
                // Gemma 4: Gemma4RMSNorm uses plain w (not 1+w)
                const float wscale = norm_plain_weight_ ? 1.0f : 0.0f;  // xf*rms*(wscale*wf + (1-wscale)*wf*(1+1/wf))
                // Simpler: branch directly
                if (norm_plain_weight_)
                    model_proj = (xf * rms * wf).view({B2, S2, config_.num_layers * D_ple}).to(dtype);
                else
                    model_proj = (xf * rms * (1.0f + wf)).view({B2, S2, config_.num_layers * D_ple}).to(dtype);
            }

            // 4. Sum and scale by 1/sqrt(2)
            per_layer_inputs = (tok_embs + model_proj) * (1.0 / std::sqrt(2.0));
        }

        // Gemma 4 shared KV: layers >= kv_share_start reuse K/V from store layers.
        // Store layers: (kv_share_start - 2) = last local, (kv_share_start - 1) = last global.
        // Their KV caches are read by all subsequent shared layers.
        const int kv_share_start = (config_.num_shared_kv_layers > 0)
            ? (config_.num_layers - config_.num_shared_kv_layers)
            : config_.num_layers;  // no sharing

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const auto& lw  = layer_weights_[i];
            const bool global = IsGlobalLayer(i);
            const int window  = global ? 0 : config_.sliding_window_size;

            // ── Attention sub-block ──────────────────────────────────────────
            auto residual = hidden.clone();
            auto h = GemmaRMSNorm(hidden, lw.input_layernorm, eps, norm_plain_weight_);

            // Project Q / K / V
            const auto B   = h.size(0);
            const auto Sq  = h.size(1);
            const auto n_kv = config_.num_key_value_heads;

            auto q_raw = Linear::Forward(h, lw.w_q);

            // Derive head_dim from Q weight shape (n_h*hd)
            const int64_t n_h = static_cast<int64_t>(config_.num_attention_heads);
            const int64_t hd  = q_raw.size(2) / n_h;

            auto q = q_raw.view({B, Sq, n_h, hd}).transpose(1, 2);

            // QK-norm for Q
            q = GemmaRMSNorm(q, lw.w_q_norm, eps, norm_plain_weight_);

            // RoPE per layer type
            const double layer_rope_theta = global
                ? static_cast<double>(config_.rope_theta)
                : (config_.local_rope_theta > 0.0f
                   ? static_cast<double>(config_.local_rope_theta)
                   : static_cast<double>(config_.rope_theta));

            std::pair<torch::Tensor, torch::Tensor> cs;
            if (global)
            {
                const int active_pairs = static_cast<int>(hd) / 2 * config_.rope_global_partial_factor;
                if (active_pairs > 0 && active_pairs < static_cast<int>(hd) / 2)
                    cs = RoPE::BuildCosSinPartial(position_ids, static_cast<int>(hd),
                                                  layer_rope_theta, active_pairs);
                else
                    cs = RoPE::BuildCosSin(position_ids, static_cast<int>(hd), layer_rope_theta);
            }
            else
            {
                const int layer_rope_dim = config_.rope_dim_local > 0
                    ? config_.rope_dim_local : static_cast<int>(hd);
                cs = RoPE::BuildCosSin(position_ids, layer_rope_dim, layer_rope_theta);
            }
            q = RoPE::Apply(q, cs.first, cs.second);

            // ── K/V: shared or computed ──────────────────────────────────────
            torch::Tensor k, v;
            const bool is_kv_shared = (i >= kv_share_start);

            if (is_kv_shared)
            {
                // Reuse K/V from the store layer (last local or last global before share_start)
                const int store_idx = global
                    ? (kv_share_start - 1)   // last global before sharing = layer 14
                    : (kv_share_start - 2);  // last local before sharing  = layer 13
                auto* store_cache = &kv_caches_[store_idx];
                if (store_cache->len > 0)
                {
                    // For capacity caches: slice to current len
                    if (store_cache->capacity > 0)
                    {
                        k = store_cache->key.slice(2, 0, store_cache->len);
                        v = store_cache->value.slice(2, 0, store_cache->len);
                    }
                    else
                    {
                        k = store_cache->key;
                        v = store_cache->value;
                    }
                }
                else
                {
                    // Store layer not yet computed (shouldn't happen in normal order)
                    throw InferenceError("GemmaRunner: shared KV store not populated.");
                }
            }
            else
            {
                // Compute K/V normally
                const int64_t hd_kv = static_cast<int64_t>(lw.w_k.size(0)) / n_kv;
                auto k_raw = Linear::Forward(h, lw.w_k);
                auto v_raw = Linear::Forward(h, lw.w_v);

                k = k_raw.view({B, Sq, n_kv, hd_kv}).transpose(1, 2);
                v = v_raw.view({B, Sq, n_kv, hd_kv}).transpose(1, 2);

                k = GemmaRMSNorm(k, lw.w_k_norm, eps, norm_plain_weight_);
                k = RoPE::Apply(k, cs.first, cs.second);

                // Gemma 4: v_norm — pure RMSNorm (no learnable weight)
                if (norm_plain_weight_)
                {
                    auto vf = v.to(torch::kFloat32);
                    v = (vf * torch::rsqrt(vf.pow(2).mean(-1, true) + eps)).to(v.scalar_type());
                }
            }

            // KV cache update — only for layers that compute their own K/V
            if (!is_kv_shared)
            {
                auto* cache = &kv_caches_[i];
                if (cache->capacity > 0)
                {
                    int64_t old_len = cache->len;
                    int64_t new_len = old_len + Sq;
                    if (new_len > cache->capacity)
                        throw InferenceError("GemmaRunner: KV cache overflow.");
                    cache->key.slice(2, old_len, new_len).copy_(k);
                    cache->value.slice(2, old_len, new_len).copy_(v);
                    cache->len = new_len;
                    k = cache->key.slice(2, 0, new_len);
                    v = cache->value.slice(2, 0, new_len);
                }
                else
                {
                    if (cache->IsInitialized())
                    {
                        k = torch::cat({cache->key, k}, 2);
                        v = torch::cat({cache->value, v}, 2);
                    }
                    cache->key   = k;
                    cache->value = v;
                    cache->len  += static_cast<int>(Sq);
                }
            }

            // GQA expand: k.size(1) is actual n_kv heads (may differ if shared)
            {
                const int64_t actual_n_kv = k.size(1);
                const int64_t n_rep = n_h / actual_n_kv;
                if (n_rep > 1)
                {
                    k = k.repeat_interleave(n_rep, 1);
                    v = v.repeat_interleave(n_rep, 1);
                }
            }

            // Scaled dot-product attention with optional sliding window mask
            torch::Tensor attn_out;
            const int64_t kv_len = k.size(2);
            // Gemma 4: HF uses scaling=1.0 (QK-norm handles magnitude)
            const double kAttnScale = 1.0;
            if (window > 0 && Sq > 1)
            {
                auto mask = SlidingWindowMask(Sq, window, input_ids.device());
                attn_out = torch::scaled_dot_product_attention(
                    q, k, v, mask.to(q.scalar_type()), 0.0, false, kAttnScale);
            }
            else if (window > 0 && Sq == 1)
            {
                int64_t start = std::max(int64_t{0}, kv_len - window);
                attn_out = torch::scaled_dot_product_attention(
                    q, k.slice(2, start, kv_len), v.slice(2, start, kv_len),
                    {}, 0.0, false, kAttnScale);
            }
            else
            {
                const bool is_causal = (Sq == kv_len);
                attn_out = torch::scaled_dot_product_attention(
                    q, k, v, {}, 0.0, is_causal, kAttnScale);
            }

            // Merge heads + output projection
            h = attn_out.transpose(1, 2).contiguous().view({B, Sq, n_h * hd});
            h = Linear::Forward(h, lw.w_o);

            // Post-attention norm (Gemma 4)
            if (lw.post_attn_norm.defined())
                h = GemmaRMSNorm(h, lw.post_attn_norm, eps, norm_plain_weight_);

            hidden = residual + h;

            // Dump hidden states for cosine-sim comparison (prefill only, MLLM_DUMP_HIDDEN=1)
            static const bool s_dump = (std::getenv("MLLM_DUMP_HIDDEN") != nullptr);
            if (i == 0 && !is_decode) MLLM_DEBUG("GemmaRunner", "s_dump=" + std::to_string(s_dump) + " env=" + (std::getenv("MLLM_DUMP_HIDDEN") ? "SET" : "NULL"));
            if (s_dump && !is_decode) {
                auto h_f32 = hidden[0][-1].to(torch::kFloat32).contiguous();
                const float* ptr = h_f32.data_ptr<float>();
                static FILE* fp = nullptr;
                if (!fp) fp = fopen("C:/workspace/git/mLLM/hidden_dump_mllm.bin", "wb");
                fwrite(ptr, sizeof(float), h_f32.numel(), fp);
                fflush(fp);
            }

            // ── FFN sub-block ────────────────────────────────────────────────
            residual = hidden.clone();
            h = GemmaRMSNorm(hidden, lw.post_attention_layernorm, eps, norm_plain_weight_);
            h = GeGLU(h, lw.w_gate, lw.w_up, lw.w_down);

            // Post-FFN norm (Gemma 4)
            if (lw.post_ffn_norm.defined())
                h = GemmaRMSNorm(h, lw.post_ffn_norm, eps, norm_plain_weight_);

            hidden = residual + h;

            // ── Per-Layer Input (AltUP) ──────────────────────────────────────
            if (D_ple > 0 && per_layer_inputs.defined() && lw.w_per_layer_inp_gate.defined())
            {
                auto emb_i = per_layer_inputs.narrow(-1, i * D_ple, D_ple);
                hidden = PerLayerInputForward(
                    hidden, emb_i,
                    lw.w_per_layer_inp_gate, lw.w_per_layer_proj,
                    lw.per_layer_post_norm, lw.layer_scalar, eps, norm_plain_weight_);
            }

            // Apply layer_scalar to the ENTIRE hidden state (Gemma 4 design)
            // HF: hidden_states *= self.layer_scalar  (after AltUP, before next layer)
            if (lw.layer_scalar.defined())
                hidden = hidden * lw.layer_scalar.item<float>();

        }

        hidden = GemmaRMSNorm(hidden, weights_.at("model.norm.weight"), eps, norm_plain_weight_);

        const torch::Tensor& lm_w = config_.tie_word_embeddings
            ? weights_.at("model.embed_tokens.weight")
            : weights_.at("lm_head.weight");

        auto logits = Linear::Forward(hidden, lm_w);

        // Gemma 4: final logit softcapping
        if (config_.final_logit_softcapping > 0.0f)
        {
            const float cap = config_.final_logit_softcapping;
            logits = torch::tanh(logits.to(torch::kFloat32) / cap) * cap;
        }

        return logits;
    }

    GenerateResult GemmaRunner::Generate(
        const std::vector<int64_t>& input_ids,
        const GenerateOptions& options)
    {
        if (!is_loaded_) throw InferenceError("GemmaRunner: model not loaded.");

        GenerateResult result;
        std::vector<int64_t> current = input_ids;

        for (auto& c : kv_caches_) c.Clear();

        for (int step = 0; step < options.max_new_tokens; ++step)
        {
            if (config_.max_position_embeddings > 0 &&
                (int)current.size() >= config_.max_position_embeddings)
            {
                result.finish_reason = FinishReason::Length;
                break;
            }

            torch::Tensor input_t;
            if (step == 0)
            {
                input_t = torch::tensor(current, torch::kInt64).unsqueeze(0);
            }
            else
            {
                input_t = torch::tensor(
                    std::vector<int64_t>{current.back()},
                    torch::kInt64).unsqueeze(0);
            }

            auto mask   = torch::ones({1, input_t.size(1)}, torch::kInt64);
            auto logits = Forward(input_t, mask);

            int64_t next = Sampler::Sample(
                logits.index({0, logits.size(1) - 1}),
                options.temperature, options.top_k, options.top_p,
                options.use_greedy, current, options.repetition_penalty);

            if (AppendTokenOrStop(result, current, next, options))
                break;
        }

        return result;
    }

    void GemmaRunner::InitKVCache(int batch_size, int max_seq_len)
    {
        kv_caches_.clear();
        kv_caches_.resize(config_.num_layers);

        if (!is_loaded_ || layer_weights_.empty()) return;

        const int64_t alloc_seq = std::min((int64_t)max_seq_len, (int64_t)8192);
        const auto& ref = weights_.at("model.embed_tokens.weight");
        const auto device = ref.device();
        const auto dtype  = ref.scalar_type();

        // Shared KV layers (>= kv_share_start) read from store layers — no own cache needed.
        const int kv_share_start = (config_.num_shared_kv_layers > 0)
            ? (config_.num_layers - config_.num_shared_kv_layers)
            : config_.num_layers;

        // Gemma 4 has different head_dim per layer (local vs global use different K projection sizes).
        // Derive per-layer KV head dim from the actual w_k weight shape, same as Forward() line 420.
        const int64_t n_kv = config_.num_key_value_heads;

        for (int i = 0; i < kv_share_start; ++i)
        {
            const int64_t hd_kv_i = layer_weights_[i].w_k.size(0) / n_kv;
            kv_caches_[i].Allocate(batch_size, n_kv, alloc_seq, hd_kv_i, device, dtype);
        }
        // layers [kv_share_start, num_layers) stay as empty KVCache (capacity=0)
    }

    std::string GemmaRunner::GetModelType() const { return "gemma"; }

    bool GemmaRunner::LoadConfig(const std::string& config_path)
    {
        return LoadModelConfigFromJson(config_path, config_);
    }
}


