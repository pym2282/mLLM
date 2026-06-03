// src/models/gemma/GemmaRunner.cpp

#include "models/gemma/GemmaRunner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "models/base/GenerateResult.h"
#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"
#include "models/base/GgufLoader.h"
#include "models/base/GenerateOptions.h"

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
                std::cerr << "[GemmaRunner] Failed to load config\n";
                return false;
            }

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
            {
                std::cerr << "[GemmaRunner] Failed to parse safetensors header\n";
                return false;
            }

            LoadAllWeights();

            kv_caches_.clear();
            kv_caches_.resize(config_.num_layers);
            is_loaded_ = true;

            std::cout << "[GemmaRunner] Loaded (safetensors)"
                      << " layers=" << config_.num_layers
                      << " window=" << config_.sliding_window_size
                      << "\n";
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[GemmaRunner] Load failed: " << e.what() << "\n";
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
                    std::cerr << "[GemmaRunner] head_dim override: " << config_.head_dim
                              << " → " << derived
                              << " (from Q weight shape " << q_out << "/" << config_.num_attention_heads << ")\n";
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
        is_loaded_ = true;

        std::cout << "[GemmaRunner] Loaded (GGUF)"
                  << " layers=" << config_.num_layers
                  << " window=" << config_.sliding_window_size
                  << " global_every=" << config_.full_attention_interval
                  << "\n";
        return true;
    }

    torch::Tensor& GemmaRunner::LoadWeight(const std::string& name)
    {
        auto it = weights_.find(name);
        if (it != weights_.end()) return it->second;

        auto t = SafeTensorTensorLoader::LoadTensor(model_path_, name, tensor_map_);
        auto [ins, _] = weights_.emplace(name, std::move(t));
        return ins->second;
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
                std::cout << "[GemmaRunner] per_layer_token_embd ["
                          << per_layer_token_embd_.size(0) << ","
                          << per_layer_token_embd_.size(1) << "]"
                          << " D_ple_per_layer=" << config_.hidden_size_per_layer_input << "\n";
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
            std::cout << "[GemmaRunner] Moving weights to CUDA (async)...\n";
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
        std::cout << "[GemmaRunner] Loaded " << weights_.size() << " tensors, "
                  << layer_weights_.size() << " layers"
                  << (has_altup ? " (AltUP enabled)" : "") << "\n";

        if (torch::cuda::is_available())
        {
            const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            std::cout << "[GemmaRunner] VRAM=" << stats.reserved_bytes[0].current / (1024*1024) << "MB\n";
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
        if (!is_loaded_) throw std::runtime_error("GemmaRunner: model not loaded.");

        // Move input_ids to the same device as weights
        const auto wdev = weights_.at("model.embed_tokens.weight").device();
        const auto input_ids = input_ids_cpu.to(wdev);

        const auto S = input_ids.size(1);

        // Position IDs
        torch::Tensor position_ids;
        const bool is_decode = (S == 1) && !kv_caches_.empty() && kv_caches_[0].IsInitialized();

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

        // Embedding + scale
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

            // 3. Apply per_layer_proj_norm (Gemma4RMSNorm: x*(1+w)/rms) to projection slices
            if (per_layer_proj_norm_.defined())
            {
                const auto dtype = model_proj.scalar_type();
                auto B2 = model_proj.size(0); auto S2 = model_proj.size(1);
                auto chunks = model_proj.view({B2, S2, config_.num_layers, D_ple});
                // GemmaRMSNorm: y = x * (1 + w) / rms(x)
                auto xf = chunks.to(torch::kFloat32);
                auto wf = per_layer_proj_norm_.to(torch::kFloat32);
                auto rms = torch::rsqrt(xf.pow(2).mean(-1, true) + eps);
                model_proj = (xf * rms * (1.0f + wf)).view({B2, S2, config_.num_layers * D_ple}).to(dtype);
            }

            // 4. Sum and scale by 1/sqrt(2)
            per_layer_inputs = (tok_embs + model_proj) * (1.0 / std::sqrt(2.0));
        }

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const auto& lw  = layer_weights_[i];
            const bool global = IsGlobalLayer(i);
            const int window  = global ? 0 : config_.sliding_window_size;

            // ── Attention sub-block ──────────────────────────────────────────
            auto residual = hidden.clone();
            auto h = GemmaRMSNorm(hidden, lw.input_layernorm, eps);

            // Project Q/K/V
            const auto B   = h.size(0);
            const auto Sq  = h.size(1);
            const auto n_kv = config_.num_key_value_heads;

            auto q_raw = Linear::Forward(h, lw.w_q);
            auto k_raw = Linear::Forward(h, lw.w_k);
            auto v_raw = Linear::Forward(h, lw.w_v);

            // Derive head_dim from K weight shape (n_kv*hd) — handles per-layer variation
            const int64_t hd  = static_cast<int64_t>(lw.w_k.size(0)) / n_kv;
            const int64_t n_h = q_raw.size(2) / hd;

            auto q = q_raw.view({B, Sq, n_h, hd}).transpose(1, 2);
            auto k = k_raw.view({B, Sq, n_kv, hd}).transpose(1, 2);
            auto v = v_raw.view({B, Sq, n_kv, hd}).transpose(1, 2);

            // QK-norm: Gemma variant (x*(1+w)), weights are zero-centered in GGUF
            q = GemmaRMSNorm(q, lw.w_q_norm, eps);
            k = GemmaRMSNorm(k, lw.w_k_norm, eps);

            // RoPE — global layers use rope_theta (1e6 for Gemma 4), local layers use local_rope_theta
            const double layer_rope_theta = global
                ? static_cast<double>(config_.rope_theta)
                : (config_.local_rope_theta > 0.0f
                   ? static_cast<double>(config_.local_rope_theta)
                   : static_cast<double>(config_.rope_theta));
            // rope_dim: use full head_dim for global (hd=512), rope_dim_local for local (hd=256)
            const int layer_rope_dim = global
                ? static_cast<int>(hd)
                : (config_.rope_dim_local > 0 ? config_.rope_dim_local : static_cast<int>(hd));
            auto cs = RoPE::BuildCosSin(position_ids, layer_rope_dim, layer_rope_theta);
            q = RoPE::Apply(q, cs.first, cs.second);
            k = RoPE::Apply(k, cs.first, cs.second);

            // KV cache update
            auto* cache = &kv_caches_[i];
            if (cache->capacity > 0)
            {
                int64_t old_len = cache->len;
                int64_t new_len = old_len + Sq;
                if (new_len > cache->capacity)
                    throw std::runtime_error("GemmaRunner: KV cache overflow.");
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

            // GQA expand
            const int64_t n_rep = n_h / static_cast<int64_t>(n_kv);
            if (n_rep > 1)
            {
                k = k.repeat_interleave(n_rep, 1);
                v = v.repeat_interleave(n_rep, 1);
            }

            // Scaled dot-product attention with optional sliding window mask
            torch::Tensor attn_out;
            const int64_t kv_len = k.size(2);
            // Gemma 4: query_pre_attn_scalar=256 fixed across all layers
            const double kAttnScale = 1.0 / std::sqrt(static_cast<double>(config_.head_dim));
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
                h = GemmaRMSNorm(h, lw.post_attn_norm, eps);

            hidden = residual + h;

            // ── FFN sub-block ────────────────────────────────────────────────
            residual = hidden.clone();
            h = GemmaRMSNorm(hidden, lw.post_attention_layernorm, eps);
            h = GeGLU(h, lw.w_gate, lw.w_up, lw.w_down);

            // Post-FFN norm (Gemma 4)
            if (lw.post_ffn_norm.defined())
                h = GemmaRMSNorm(h, lw.post_ffn_norm, eps);

            hidden = residual + h;

            // ── Per-Layer Input (AltUP) ──────────────────────────────────────
            if (D_ple > 0 && per_layer_inputs.defined() && lw.w_per_layer_inp_gate.defined())
            {
                auto emb_i = per_layer_inputs.narrow(-1, i * D_ple, D_ple);
                hidden = PerLayerInputForward(
                    hidden, emb_i,
                    lw.w_per_layer_inp_gate, lw.w_per_layer_proj,
                    lw.per_layer_post_norm, lw.layer_scalar, eps);
            }

            // Debug: print norm of hidden after each layer (first token only)
            // (removed)
        }

        hidden = GemmaRMSNorm(hidden, weights_.at("model.norm.weight"), eps);

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
        if (!is_loaded_) throw std::runtime_error("GemmaRunner: model not loaded.");

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

            result.tokens.push_back(next);
            current.push_back(next);

            if (next == options.eos_token_id)
            {
                result.finish_reason = FinishReason::EOS;
                break;
            }

            for (const auto& stop : options.stop_sequence_ids)
            {
                if (stop.empty()) continue;
                const size_t n = stop.size();
                if (current.size() >= n &&
                    std::equal(stop.begin(), stop.end(),
                               current.end() - static_cast<ptrdiff_t>(n)))
                {
                    result.finish_reason = FinishReason::Stop;
                    goto done;
                }
            }
        }

    done:
        return result;
    }

    void GemmaRunner::InitKVCache(int batch_size, int max_seq_len)
    {
        (void)batch_size;
        (void)max_seq_len;
        kv_caches_.clear();
        kv_caches_.resize(config_.num_layers);
    }

    const ModelConfig& GemmaRunner::GetConfig() const { return config_; }

    std::string GemmaRunner::GetModelType() const { return "gemma"; }

    bool GemmaRunner::LoadConfig(const std::string& config_path)
    {
        return LoadModelConfigFromJson(config_path, config_);
    }
}
