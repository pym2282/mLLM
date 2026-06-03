// src/models/qwen/QwenRunner.cpp

#include "models/qwen/QwenRunner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "models/base/GenerateResult.h"

#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorTensorLoader.h"
#include "models/base/GgufLoader.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/Sampler.h"
#include "core/runtime/TransformerBlock.h"
#include "core/runtime/RMSNorm.h"
#include "debug/TensorCompare.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace mllm
{
    bool QwenRunner::LoadGguf(const std::string& gguf_path)
    {
        try
        {
            model_path_ = gguf_path;

            config_ = GgufLoader::ReadConfig(gguf_path);

            // Compute per-layer hybrid flags from config
            layer_is_hybrid_.assign(config_.num_layers, false);
            if (config_.full_attention_interval > 0)
            {
                for (int i = 0; i < config_.num_layers; ++i)
                    layer_is_hybrid_[i] = ((i + 1) % config_.full_attention_interval != 0);
            }

            std::cout << "[QwenRunner] Loading GGUF weights: " << gguf_path << std::endl;
            weights_ = GgufLoader::Load(gguf_path);

            kv_caches_.clear();
            kv_caches_.resize(config_.num_layers);
            ssm_caches_.clear();
            ssm_caches_.resize(config_.num_layers);
            prefilled_tokens_ = 0;

            LoadAllWeights();

            is_loaded_ = true;

            std::cout
                << "[QwenRunner] GGUF model loaded"
                << " layers=" << config_.num_layers
                << " weights=" << weights_.size()
                << std::endl;

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[QwenRunner] GGUF load failed:\n" << e.what() << std::endl;
            is_loaded_ = false;
            return false;
        }
    }

    bool QwenRunner::Load(const std::string& model_path)
    {
        if (GgufLoader::IsGguf(model_path))
            return LoadGguf(model_path);

        try
        {
            model_path_ = model_path;
            if (!LoadConfig(model_path + "/config.json"))
            {
                std::cerr << "[QwenRunner] Failed to load config" << std::endl;
                return false;
            }

            // 캐시 경로: config.json 기준으로 유효성 검사
            const std::string cache_path = model_path + "/.mllm_weights.mlm";
            const std::string src_key    = model_path + "/config.json";
            bool from_cache = false;

            if (GgufLoader::IsCacheValid(src_key, cache_path))
            {
                std::cout << "[QwenRunner] Loading from cache..." << std::endl;
                auto cached = GgufLoader::LoadCache(cache_path);
                if (!cached.empty())
                {
                    weights_ = std::move(cached);
                    from_cache = true;
                }
                else
                    std::cerr << "[QwenRunner] Cache corrupt, falling back to safetensors...\n";
            }

            if (!from_cache)
            {
                if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
                {
                    std::cerr << "[QwenRunner] Failed to parse safetensors header" << std::endl;
                    return false;
                }
            }

            // LoadAllWeights: weights_ 이미 채워져 있으면 LoadWeight()가 파일 I/O 건너뜀
            // 캐시 미스 시 GPU 전송 전에 저장
            if (!from_cache)
            {
                pending_cache_path_ = cache_path;
                pending_cache_src_  = src_key;
            }
            LoadAllWeights();


            kv_caches_.clear();
            kv_caches_.resize(config_.num_layers);

            is_loaded_ = true;

            std::cout
                << "[QwenRunner] Model loaded"
                << " path=" << model_path_
                << " layers=" << config_.num_layers
                << " weights=" << weights_.size()
                << std::endl;

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "[QwenRunner] Load failed:\n"
                << e.what()
                << std::endl;

            is_loaded_ = false;
            return false;
        }
    }

    torch::Tensor& QwenRunner::LoadWeight(const std::string& name)
    {
        auto it = weights_.find(name);

        if (it != weights_.end())
        {
            return it->second;
        }

        auto tensor = SafeTensorTensorLoader::LoadTensor(
            model_path_,
            name,
            tensor_map_
        );

        auto [ins, ok] = weights_.emplace(
            name,
            std::move(tensor)
        );

        (void)ok;
        return ins->second;
    }

    torch::Tensor QwenRunner::TryLoadWeight(const std::string& name)
    {
        // Check pre-loaded cache first (covers GGUF path)
        auto it = weights_.find(name);
        if (it != weights_.end()) return it->second;
        // Fall back to safetensors lazy-load
        if (tensor_map_.find(name) == tensor_map_.end()) return {};
        return LoadWeight(name);
    }

    // Dequantize one FP8 weight tensor using its block-wise scale.
    // Uses reshape+broadcast instead of repeat_interleave to avoid materializing
    // the full [out,in] scale tensor in float32.
    static torch::Tensor DequantizeFP8(
        const torch::Tensor& w,
        const torch::Tensor& scale,
        torch::ScalarType target_dtype)
    {
        if (!scale.defined())
            return w.to(target_dtype);

        if (scale.dim() != 2)
        {
            std::cerr << "[DequantizeFP8] FATAL: scale must be 2D, got ndim="
                      << scale.dim() << " sizes=" << scale.sizes() << std::endl;
            throw std::runtime_error("DequantizeFP8: scale must be 2D");
        }

        if (w.size(0) % scale.size(0) != 0 || w.size(1) % scale.size(1) != 0)
        {
            std::cerr << "[DequantizeFP8] FATAL: dimension mismatch "
                      << "w=" << w.sizes() << " scale=" << scale.sizes() << std::endl;
            throw std::runtime_error("DequantizeFP8: dimension mismatch");
        }

        const int64_t block_out = w.size(0) / scale.size(0);
        const int64_t block_in  = w.size(1) / scale.size(1);

        // w: [out, in] → [out/block, block_out, in/block, block_in]
        auto w_h = w.to(target_dtype)
                    .reshape({scale.size(0), block_out, scale.size(1), block_in});

        // scale: [out/block, in/block] → [out/block, 1, in/block, 1]  (broadcast, no copy)
        auto s_h = scale.to(target_dtype).unsqueeze(1).unsqueeze(3);

        return (w_h * s_h).reshape({w.size(0), w.size(1)});
    }

    // Returns a copy of lw with FP8 weights dequantized to target_dtype.
    // Non-projection weights (layernorm, qk_norm) are copied as-is.
    static LayerWeights DequantizeLayerWeights(
        const LayerWeights& lw,
        torch::ScalarType target_dtype = torch::kFloat16)
    {
        LayerWeights out    = lw;
        out.w_q             = DequantizeFP8(lw.w_q,   lw.w_q_scale,   target_dtype);
        out.w_k             = DequantizeFP8(lw.w_k,   lw.w_k_scale,   target_dtype);
        out.w_v             = DequantizeFP8(lw.w_v,   lw.w_v_scale,   target_dtype);
        out.w_o             = DequantizeFP8(lw.w_o,   lw.w_o_scale,   target_dtype);
        out.w_gate          = DequantizeFP8(lw.w_gate, lw.w_gate_scale, target_dtype);
        out.w_up            = DequantizeFP8(lw.w_up,  lw.w_up_scale,  target_dtype);
        out.w_down          = DequantizeFP8(lw.w_down, lw.w_down_scale, target_dtype);
        out.w_q_scale       = torch::Tensor();
        out.w_k_scale       = torch::Tensor();
        out.w_v_scale       = torch::Tensor();
        out.w_o_scale       = torch::Tensor();
        out.w_gate_scale    = torch::Tensor();
        out.w_up_scale      = torch::Tensor();
        out.w_down_scale    = torch::Tensor();
        return out;
    }

    void QwenRunner::LoadAllWeights()
    {
        LoadWeight("model.embed_tokens.weight");
        LoadWeight("model.norm.weight");

        if (!config_.tie_word_embeddings)
            LoadWeight("lm_head.weight");

        layer_weights_.clear();
        layer_weights_.reserve(config_.num_layers);
        linear_attn_weights_.clear();
        linear_attn_weights_.resize(config_.num_layers);

        // Ensure layer_is_hybrid_ is sized (may not be if loading HF format)
        if (static_cast<int>(layer_is_hybrid_.size()) != config_.num_layers)
            layer_is_hybrid_.assign(config_.num_layers, false);

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string p = "model.layers." + std::to_string(i);

            LayerWeights lw;
            lw.input_layernorm =
                LoadWeight(p + ".input_layernorm.weight");
            lw.post_attention_layernorm =
                LoadWeight(p + ".post_attention_layernorm.weight");

            if (layer_is_hybrid_[i])
            {
                // Gated DeltaNet weights
                const std::string la = p + ".linear_attn.";
                linear_attn_weights_[i].in_proj_qkv = LoadWeight(la + "in_proj_qkv.weight");
                linear_attn_weights_[i].in_proj_z   = LoadWeight(la + "in_proj_z.weight");
                linear_attn_weights_[i].in_proj_a   = LoadWeight(la + "in_proj_a.weight");
                linear_attn_weights_[i].in_proj_b   = LoadWeight(la + "in_proj_b.weight");
                linear_attn_weights_[i].conv1d       = LoadWeight(la + "conv1d.weight");
                linear_attn_weights_[i].dt_bias      = LoadWeight(la + "dt_bias");
                linear_attn_weights_[i].A_log        = LoadWeight(la + "A_log");
                linear_attn_weights_[i].norm         = LoadWeight(la + "norm.weight");
                linear_attn_weights_[i].out_proj     = LoadWeight(la + "out_proj.weight");
            }
            else
            {
                // Full-attention weights (Qwen3 style: separate q/k/v + qk-norm)
                lw.w_q      = LoadWeight(p + ".self_attn.q_proj.weight");
                lw.w_q_scale = TryLoadWeight(p + ".self_attn.q_proj.weight_scale_inv");

                lw.w_k      = LoadWeight(p + ".self_attn.k_proj.weight");
                lw.w_k_scale = TryLoadWeight(p + ".self_attn.k_proj.weight_scale_inv");

                lw.w_v      = LoadWeight(p + ".self_attn.v_proj.weight");
                lw.w_v_scale = TryLoadWeight(p + ".self_attn.v_proj.weight_scale_inv");

                lw.w_o      = LoadWeight(p + ".self_attn.o_proj.weight");
                lw.w_o_scale = TryLoadWeight(p + ".self_attn.o_proj.weight_scale_inv");

                lw.w_q_norm = TryLoadWeight(p + ".self_attn.q_norm.weight");
                lw.w_k_norm = TryLoadWeight(p + ".self_attn.k_norm.weight");
                lw.b_q      = TryLoadWeight(p + ".self_attn.q_proj.bias");
                lw.b_k      = TryLoadWeight(p + ".self_attn.k_proj.bias");
                lw.b_v      = TryLoadWeight(p + ".self_attn.v_proj.bias");
            }

            lw.w_gate      = LoadWeight(p + ".mlp.gate_proj.weight");
            lw.w_gate_scale = TryLoadWeight(p + ".mlp.gate_proj.weight_scale_inv");
            lw.w_up        = LoadWeight(p + ".mlp.up_proj.weight");
            lw.w_up_scale   = TryLoadWeight(p + ".mlp.up_proj.weight_scale_inv");
            lw.w_down      = LoadWeight(p + ".mlp.down_proj.weight");
            lw.w_down_scale = TryLoadWeight(p + ".mlp.down_proj.weight_scale_inv");

            layer_weights_.push_back(std::move(lw));
        }

        std::cout
            << "[QwenRunner] Loaded "
            << weights_.size()
            << " tensors"
            << std::endl;

        // 캐시 저장: GPU 전송 전 CPU 텐서 상태에서 저장
        if (!pending_cache_path_.empty())
        {
            std::cerr << "[QwenRunner] Saving cache to " << pending_cache_path_ << "...\n";
            GgufLoader::SaveCache(pending_cache_path_, pending_cache_src_, weights_);
            pending_cache_path_.clear();
            pending_cache_src_.clear();
        }

        gpu_ready_ = false;

        // layer_weights_ 빌드 (CPU 텐서 참조 — GPU 전송은 Generate() 첫 호출 시)
        {
            auto try_get = [&](const std::string& key) -> torch::Tensor
            {
                auto it = weights_.find(key);
                return (it != weights_.end()) ? it->second : torch::Tensor();
            };

            for (int i = 0; i < config_.num_layers; ++i)
            {
                const std::string p = "model.layers." + std::to_string(i);
                auto& lw = layer_weights_[i];
                lw.input_layernorm          = weights_.at(p + ".input_layernorm.weight");
                lw.post_attention_layernorm = weights_.at(p + ".post_attention_layernorm.weight");

                if (layer_is_hybrid_[i])
                {
                    const std::string la = p + ".linear_attn.";
                    linear_attn_weights_[i].in_proj_qkv = weights_.at(la + "in_proj_qkv.weight");
                    linear_attn_weights_[i].in_proj_z   = weights_.at(la + "in_proj_z.weight");
                    linear_attn_weights_[i].in_proj_a   = weights_.at(la + "in_proj_a.weight");
                    linear_attn_weights_[i].in_proj_b   = weights_.at(la + "in_proj_b.weight");
                    linear_attn_weights_[i].conv1d       = weights_.at(la + "conv1d.weight");
                    linear_attn_weights_[i].dt_bias      = weights_.at(la + "dt_bias");
                    linear_attn_weights_[i].A_log        = weights_.at(la + "A_log");
                    linear_attn_weights_[i].norm         = weights_.at(la + "norm.weight");
                    linear_attn_weights_[i].out_proj     = weights_.at(la + "out_proj.weight");
                }
                else
                {
                    lw.w_q      = weights_.at(p + ".self_attn.q_proj.weight");
                    lw.w_k      = weights_.at(p + ".self_attn.k_proj.weight");
                    lw.w_v      = weights_.at(p + ".self_attn.v_proj.weight");
                    lw.w_o      = weights_.at(p + ".self_attn.o_proj.weight");
                    lw.w_q_norm = try_get(p + ".self_attn.q_norm.weight");
                    lw.w_k_norm = try_get(p + ".self_attn.k_norm.weight");
                    lw.b_q      = try_get(p + ".self_attn.q_proj.bias");
                    lw.b_k      = try_get(p + ".self_attn.k_proj.bias");
                    lw.b_v      = try_get(p + ".self_attn.v_proj.bias");
                    lw.w_q_scale    = try_get(p + ".self_attn.q_proj.weight_scale_inv");
                    lw.w_k_scale    = try_get(p + ".self_attn.k_proj.weight_scale_inv");
                    lw.w_v_scale    = try_get(p + ".self_attn.v_proj.weight_scale_inv");
                    lw.w_o_scale    = try_get(p + ".self_attn.o_proj.weight_scale_inv");
                }

                lw.w_gate       = weights_.at(p + ".mlp.gate_proj.weight");
                lw.w_up         = weights_.at(p + ".mlp.up_proj.weight");
                lw.w_down       = weights_.at(p + ".mlp.down_proj.weight");
                lw.w_gate_scale = try_get(p + ".mlp.gate_proj.weight_scale_inv");
                lw.w_up_scale   = try_get(p + ".mlp.up_proj.weight_scale_inv");
                lw.w_down_scale = try_get(p + ".mlp.down_proj.weight_scale_inv");
            }
        }

        std::cout << "[QwenRunner] Weights on CPU. GPU transfer deferred to first Generate().\n";
    }

    void QwenRunner::EnsureOnGPU()
    {
        if (gpu_ready_ || !torch::cuda::is_available()) return;

        std::cout << "[QwenRunner] Moving weights to CUDA (lazy)..." << std::endl;
        for (auto& [name, w] : weights_)
            w = w.to(torch::kCUDA);

        // layer_weights_ 재빌드 (GPU 텐서로)
        auto try_get = [&](const std::string& key) -> torch::Tensor {
            auto it = weights_.find(key);
            return (it != weights_.end()) ? it->second : torch::Tensor();
        };
        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string p = "model.layers." + std::to_string(i);
            auto& lw = layer_weights_[i];
            lw.input_layernorm          = weights_.at(p + ".input_layernorm.weight");
            lw.post_attention_layernorm = weights_.at(p + ".post_attention_layernorm.weight");
            if (!layer_is_hybrid_[i])
            {
                lw.w_q = weights_.at(p + ".self_attn.q_proj.weight");
                lw.w_k = weights_.at(p + ".self_attn.k_proj.weight");
                lw.w_v = weights_.at(p + ".self_attn.v_proj.weight");
                lw.w_o = weights_.at(p + ".self_attn.o_proj.weight");
                lw.w_q_norm = try_get(p + ".self_attn.q_norm.weight");
                lw.w_k_norm = try_get(p + ".self_attn.k_norm.weight");
                lw.b_q = try_get(p + ".self_attn.q_proj.bias");
                lw.b_k = try_get(p + ".self_attn.k_proj.bias");
                lw.b_v = try_get(p + ".self_attn.v_proj.bias");
                lw.w_q_scale = try_get(p + ".self_attn.q_proj.weight_scale_inv");
                lw.w_k_scale = try_get(p + ".self_attn.k_proj.weight_scale_inv");
                lw.w_v_scale = try_get(p + ".self_attn.v_proj.weight_scale_inv");
                lw.w_o_scale = try_get(p + ".self_attn.o_proj.weight_scale_inv");
            }
            else
            {
                const std::string la = p + ".linear_attn.";
                linear_attn_weights_[i].in_proj_qkv = weights_.at(la + "in_proj_qkv.weight");
                linear_attn_weights_[i].in_proj_z   = weights_.at(la + "in_proj_z.weight");
                linear_attn_weights_[i].in_proj_a   = weights_.at(la + "in_proj_a.weight");
                linear_attn_weights_[i].in_proj_b   = weights_.at(la + "in_proj_b.weight");
                linear_attn_weights_[i].conv1d      = weights_.at(la + "conv1d.weight");
                linear_attn_weights_[i].dt_bias     = weights_.at(la + "dt_bias");
                linear_attn_weights_[i].A_log       = weights_.at(la + "A_log");
                linear_attn_weights_[i].norm        = weights_.at(la + "norm.weight");
                linear_attn_weights_[i].out_proj    = weights_.at(la + "out_proj.weight");
            }
            lw.w_gate      = weights_.at(p + ".mlp.gate_proj.weight");
            lw.w_up        = weights_.at(p + ".mlp.up_proj.weight");
            lw.w_down      = weights_.at(p + ".mlp.down_proj.weight");
            lw.w_gate_scale = try_get(p + ".mlp.gate_proj.weight_scale_inv");
            lw.w_up_scale   = try_get(p + ".mlp.up_proj.weight_scale_inv");
            lw.w_down_scale = try_get(p + ".mlp.down_proj.weight_scale_inv");
        }

        c10::cuda::CUDACachingAllocator::emptyCache();
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        std::cout << "[QwenRunner] VRAM reserved="
                  << stats.reserved_bytes[0].current / (1024*1024)
                  << "MB allocated="
                  << stats.allocated_bytes[0].current / (1024*1024) << "MB\n";
        gpu_ready_ = true;
    }

    torch::Tensor QwenRunner::Forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& /*attention_mask*/)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error(
                "QwenRunner: model not loaded."
            );
        }

        torch::NoGradGuard no_grad;

        const auto S = input_ids.size(1);
        const auto target_device = weights_.at("model.embed_tokens.weight").device();

        torch::Tensor position_ids;

        const bool is_decode = (S == 1) && prefilled_tokens_ > 0;

        // Prefix caching: consume prefix_kv_len_ on the first prefill call.
        // If set, KV caches already contain prefix_kv_len_ tokens from SetKVSnapshot.
        const int64_t prefix_offset = (!is_decode && prefix_kv_len_ > 0)
                                      ? prefix_kv_len_ : 0;
        if (prefix_offset > 0)
            prefix_kv_len_ = 0;  // consume

        if (!is_decode && prefix_offset == 0)
        {
            // Regular prefill: clear all caches
            for (auto& kvc : kv_caches_)
                kvc.Clear();
            for (auto& smc : ssm_caches_)
                smc.Clear();
            prefilled_tokens_ = 0;
        }

        if (is_decode)
        {
            position_ids = torch::tensor(
                { static_cast<int64_t>(prefilled_tokens_) },
                torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(target_device)
            );
        }
        else
        {
            // Prefill: positions start from prefix_offset (0 normally, >0 with prefix cache)
            position_ids = torch::arange(
                prefix_offset,
                prefix_offset + S,
                torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(target_device)
            );
        }

        const bool on_cuda = weights_.at("model.embed_tokens.weight").is_cuda();
        const torch::Tensor ids_dev = on_cuda
            ? input_ids.to(torch::kCUDA)
            : input_ids;

        auto hidden = EmbeddingLookup::Forward(
            ids_dev,
            weights_.at("model.embed_tokens.weight")
        );

        // CLI/parity paths do not preallocate caches. Linear-attention state
        // still needs storage before either prefill or single-step forward.
        if (config_.full_attention_interval > 0)
        {
            for (int i = 0; i < config_.num_layers; ++i)
            {
                if (layer_is_hybrid_[i] && !ssm_caches_[i].IsInitialized())
                {
                    ssm_caches_[i].Allocate(
                        static_cast<int>(hidden.size(0)),
                        config_.ssm_conv_dim,
                        config_.ssm_conv_kernel,
                        config_.ssm_num_v_heads,
                        config_.ssm_head_k_dim,
                        config_.ssm_head_v_dim,
                        hidden.device(),
                        hidden.scalar_type()
                    );
                }
            }
        }

        // -----------------------------
        // embedding parity compare
        // -----------------------------
        if (parity_mode_ && !is_decode)
        {
            TensorCompare::CompareTensor(
                hidden,
                parity_reference_dir_ + "/embedding.txt"
            );
        }

        for (int i = 0; i < config_.num_layers; ++i)
        {
            if (static_cast<int>(layer_is_hybrid_.size()) > i && layer_is_hybrid_[i])
            {
#if 0  // ISOLATION TEST: HYBRID pass-through — remove before ship
                (void)ssm_caches_[i];
#else
                hidden = ForwardHybridLayer(
                    hidden,
                    linear_attn_weights_[i],
                    layer_weights_[i],
                    &ssm_caches_[i]
                );
#endif
            }
            else
            {
                const bool is_fp8 =
                    layer_weights_[i].w_q.defined() &&
                    layer_weights_[i].w_q.scalar_type() == torch::kFloat8_e4m3fn;

                LayerWeights dequant_lw;
                if (is_fp8)
                    dequant_lw = DequantizeLayerWeights(layer_weights_[i], hidden.scalar_type());

                const LayerWeights& lw = is_fp8 ? dequant_lw : layer_weights_[i];

                hidden = TransformerBlock::Forward(
                    hidden,
                    lw,
                    config_.num_attention_heads,
                    config_.num_key_value_heads,
                    config_.head_dim,
                    static_cast<double>(config_.rope_theta),
                    static_cast<double>(config_.rms_norm_eps),
                    lw.w_q_norm.defined(),
                    position_ids,
                    &kv_caches_[i],
                    config_.rope_dim
                );
            }
        }

        prefilled_tokens_ += static_cast<int>(S);

        if (parity_mode_ && !is_decode) {
            TensorCompare::CompareTensor(
                hidden,
                parity_reference_dir_ + "/last_layer_output.txt"
            );
        }

        hidden = RMSNorm::Forward(
            hidden,
            weights_.at("model.norm.weight"),
            config_.rms_norm_eps
        );

        if (parity_mode_ && !is_decode)
        {
            TensorCompare::CompareTensor(
                hidden,
                parity_reference_dir_ + "/final_norm_output.txt"
            );
        }

        const bool has_lm_head = weights_.count("lm_head.weight") > 0;
        const torch::Tensor& lm_head_w =
            has_lm_head
                ? weights_.at("lm_head.weight")
                : weights_.at("model.embed_tokens.weight");

        auto logits = Linear::Forward(hidden, lm_head_w);

        if (parity_mode_ && !is_decode)
        {
            auto last_logits =
                logits.select(1, logits.size(1) - 1);

            TensorCompare::CompareTensor(
                last_logits,
                parity_reference_dir_ + "/final_logits.txt"
            );
        }

        return logits;
    }

    GenerateResult QwenRunner::Generate(
        const std::vector<int64_t>& input_ids,
        const GenerateOptions& options)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error(
                "QwenRunner: model not loaded."
            );
        }

        EnsureOnGPU();  // 첫 호출 시 GPU로 이동 (이후 no-op)

        GenerateResult result;

        // prefix_kv_len > 0: SetKVSnapshot으로 KV가 이미 로드됨 → prefill 단축
        const int64_t prefix_len = options.prefix_kv_len;

        if (prefix_len == 0)
        {
            // Normal: clear KV caches before prefill
            for (auto& cache : kv_caches_)
                cache.Clear();
        }
        // else: KV caches already set by SetKVSnapshot, don't clear

        std::vector<int64_t> current = input_ids;

        std::cout
            << "[QwenRunner] Generating: prompt_len="
            << current.size()
            << " max_new_tokens=" << options.max_new_tokens
            << " thinking=" << (options.enable_thinking ? "on" : "off")
            << std::endl;
        std::cout << "[QwenRunner] Prefill start..." << std::endl;
        std::cout.flush();

        for (int step = 0; step < options.max_new_tokens; ++step)
        {
            if (config_.max_position_embeddings > 0 &&
                (int)current.size() >= config_.max_position_embeddings)
            {
                result.finish_reason = FinishReason::Length;
                break;
            }

            torch::Tensor input_tensor;

            if (step == 0)
            {
                // Prefix caching: only process tokens after the cached prefix
                const auto begin = current.begin() + prefix_len;
                input_tensor = torch::tensor(
                    std::vector<int64_t>(begin, current.end()),
                    torch::TensorOptions().dtype(torch::kInt64)
                ).unsqueeze(0);
            }
            else
            {
                input_tensor = torch::tensor(
                    std::vector<int64_t>{ current.back() },
                    torch::TensorOptions().dtype(torch::kInt64)
                ).unsqueeze(0);
            }

            auto attention_mask = torch::ones(
                { 1, input_tensor.size(1) },
                torch::kInt64
            );

            auto logits = Forward(
                input_tensor,
                attention_mask
            );

            int64_t next_token =
                Sampler::Sample(
                    logits.index({
                        0,
                        logits.size(1) - 1
                    }),
                    options.temperature,
                    options.top_k,
                    options.top_p,
                    options.use_greedy,
                    current,
                    options.repetition_penalty
                );

            result.tokens.push_back(next_token);
            current.push_back(next_token);

            if (options.on_token && !options.on_token(next_token))
            {
                result.finish_reason = FinishReason::Stop;
                break;
            }

            if (step == 0)
            {
                std::cout << "[QwenRunner] Prefill done. First token: " << next_token << std::endl;
                std::cout.flush();
            }
            else if (step % 20 == 0)
            {
                std::cout << "[QwenRunner] Decode step " << step << std::endl;
                std::cout.flush();
            }

            if (next_token == options.eos_token_id)
            {
                result.finish_reason = FinishReason::EOS;
                std::cout << "[QwenRunner] EOS detected." << std::endl;
                break;
            }

            bool stop_hit = false;
            for (const auto& stop_seq : options.stop_sequence_ids)
            {
                if (stop_seq.empty()) continue;
                const size_t n = stop_seq.size();
                if (current.size() >= n &&
                    std::equal(stop_seq.begin(), stop_seq.end(),
                               current.end() - static_cast<ptrdiff_t>(n)))
                {
                    result.finish_reason = FinishReason::Stop;
                    stop_hit = true;
                    break;
                }
            }
            if (stop_hit) break;
        }

        return result;
    }

    KVSnapshot QwenRunner::GetKVSnapshot(int64_t len) const
    {
        KVSnapshot snap;
        snap.len = len;
        snap.keys.reserve(kv_caches_.size());
        snap.values.reserve(kv_caches_.size());
        for (const auto& c : kv_caches_)
        {
            if (!c.IsInitialized() || c.len < len)
                return {};
            snap.keys.push_back(c.key.slice(2, 0, len).cpu().contiguous());
            snap.values.push_back(c.value.slice(2, 0, len).cpu().contiguous());
        }
        return snap;
    }

    void QwenRunner::SetKVSnapshot(const KVSnapshot& snap)
    {
        if (snap.empty()) return;
        const int64_t len = snap.len;
        bool any_restored = false;
        for (int i = 0; i < (int)kv_caches_.size() && i < (int)snap.keys.size(); ++i)
        {
            auto& c = kv_caches_[i];
            if (!c.IsInitialized()) continue;
            const auto dev = c.key.device();
            c.key.slice(2, 0, len).copy_(snap.keys[i].to(dev));
            c.value.slice(2, 0, len).copy_(snap.values[i].to(dev));
            c.len = len;
            any_restored = true;
        }
        if (any_restored)
        {
            prefix_kv_len_    = len;
            prefilled_tokens_ = static_cast<int>(len);
        }
    }

    void QwenRunner::InitKVCache(
        int batch_size,
        int max_seq_len)
    {
        const int64_t alloc_seq = std::min(static_cast<int64_t>(max_seq_len), static_cast<int64_t>(8192));

        const auto& ref = weights_.at("model.embed_tokens.weight");
        const auto device = ref.device();
        const auto dtype  = ref.scalar_type();

        kv_caches_.clear();
        kv_caches_.resize(config_.num_layers);
        ssm_caches_.clear();
        ssm_caches_.resize(config_.num_layers);
        prefilled_tokens_ = 0;

        const bool is_hybrid_model = config_.full_attention_interval > 0;

        for (int i = 0; i < config_.num_layers; ++i)
        {
            if (is_hybrid_model && layer_is_hybrid_[i])
            {
                ssm_caches_[i].Allocate(
                    batch_size,
                    config_.ssm_conv_dim,
                    config_.ssm_conv_kernel,
                    config_.ssm_num_v_heads,
                    config_.ssm_head_k_dim,
                    config_.ssm_head_v_dim,
                    device,
                    dtype
                );
            }
            else
            {
                kv_caches_[i].Allocate(
                    batch_size,
                    config_.num_key_value_heads,
                    alloc_seq,
                    config_.head_dim,
                    device,
                    dtype
                );
            }
        }
    }

    const ModelConfig& QwenRunner::GetConfig() const
    {
        return config_;
    }

    std::string QwenRunner::GetModelType() const
    {
        return "qwen";
    }

    void QwenRunner::SetParityMode(bool enabled)
    {
        parity_mode_ = enabled;
    }

    void QwenRunner::SetParityReferenceDir(const std::string& path)
    {
        parity_reference_dir_ = path;
    }

    bool QwenRunner::LoadConfig(
        const std::string& config_path)
    {
        return LoadModelConfigFromJson(
            config_path,
            config_
        );
    }

    // ---------------------------------------------------------------------------
    // Gated DeltaNet (linear-attention) layer forward pass
    //
    // Reference: Qwen3_5GatedDeltaNet / torch_recurrent_gated_delta_rule
    // State update per time step t:
    //   state = state * exp(g_t) + k_t ⊗ delta_t
    //   delta_t = (v_t - (state * k_t).sum(-2)) * beta_t
    //   out_t   = (state * q_t).sum(-2)
    // ---------------------------------------------------------------------------

    torch::Tensor QwenRunner::ForwardHybridLayer(
        const torch::Tensor& hidden,
        const LinearAttnWeights& w,
        const LayerWeights& lw,
        SSMCache* cache)
    {
        const int B  = static_cast<int>(hidden.size(0));
        const int S  = static_cast<int>(hidden.size(1));
        const int H  = static_cast<int>(hidden.size(2));

        const int nv  = config_.ssm_num_v_heads;
        const int nk  = config_.ssm_num_k_heads;
        const int hvd = config_.ssm_head_v_dim;
        const int hkd = config_.ssm_head_k_dim;
        const int vdim = config_.ssm_inner_size;
        const int cdim = config_.ssm_conv_dim;
        const int K    = config_.ssm_conv_kernel;
        const int ratio = nv / nk;

        // ── Residual + input norm ────────────────────────────────────────────
        auto residual = hidden;
        auto h = RMSNorm::Forward(hidden, lw.input_layernorm, config_.rms_norm_eps);

        // ── Linear projections ───────────────────────────────────────────────
        auto h2d = h.reshape({B * S, H});
        auto qkv = torch::mm(h2d, w.in_proj_qkv.t()).reshape({B, S, cdim});
        auto z   = torch::mm(h2d, w.in_proj_z.t())  .reshape({B, S, vdim});
        auto a   = torch::mm(h2d, w.in_proj_a.t())  .reshape({B, S, nv});
        auto b   = torch::mm(h2d, w.in_proj_b.t())  .reshape({B, S, nv});

        // ── Causal depthwise conv1d ──────────────────────────────────────────
        // conv1d weight: [cdim, K]  (depthwise, no bias)
        auto qkv_t = qkv.transpose(1, 2).contiguous();  // [B, cdim, S]

        if (S == 1)
        {
            // Single-step: roll state and compute weighted sum
            cache->conv_state = torch::roll(cache->conv_state, -1, -1);
            cache->conv_state.select(2, K - 1).copy_(qkv_t.squeeze(2));
            qkv_t = (cache->conv_state * w.conv1d.unsqueeze(0)).sum(-1, true);  // [B,cdim,1]
        }
        else
        {
            // Prefill: causal conv with zero-padding on the left
            const auto raw_t = qkv_t.clone();  // save raw input for state update
            auto zero_pad = torch::zeros({B, cdim, K - 1}, qkv_t.options());
            auto padded   = torch::cat({zero_pad, qkv_t}, 2);  // [B, cdim, S+K-1]
            auto cw = w.conv1d.unsqueeze(1).to(padded.dtype());  // [cdim, 1, K]
            // Depthwise conv: stride=1, padding=0 (already padded), dilation=1, groups=cdim
            qkv_t = torch::conv1d(padded, cw, c10::nullopt,
                                  at::IntArrayRef{1}, at::IntArrayRef{0},
                                  at::IntArrayRef{1}, static_cast<int64_t>(cdim));

            // Store last K frames as new conv_state
            if (S >= K)
                cache->conv_state = raw_t.slice(2, S - K).contiguous();
            else
            {
                auto lpad = torch::zeros({B, cdim, K - S}, raw_t.options());
                cache->conv_state = torch::cat({lpad, raw_t}, 2);
            }
        }
        qkv_t = torch::silu(qkv_t);
        qkv = qkv_t.transpose(1, 2).contiguous();  // [B, S, cdim]

        // ── Split and reshape ────────────────────────────────────────────────
        const int kdim = nk * hkd;
        auto parts = qkv.split({kdim, kdim, vdim}, -1);
        auto q = parts[0].reshape({B, S, nk, hkd});
        auto k = parts[1].reshape({B, S, nk, hkd});
        auto v = parts[2].reshape({B, S, nv, hvd});
        z = z.reshape({B, S, nv, hvd});

        // Expand k/q from nk to nv heads
        if (ratio > 1)
        {
            q = q.repeat_interleave(ratio, 2);  // [B, S, nv, hkd]
            k = k.repeat_interleave(ratio, 2);
        }

        // Match the reference kernel: normalize in FP32, with epsilon under rsqrt.
        auto q_f = q.to(torch::kFloat32);
        auto k_f = k.to(torch::kFloat32);
        q_f = q_f * torch::rsqrt((q_f * q_f).sum(-1, true) + 1e-6f);
        k_f = k_f * torch::rsqrt((k_f * k_f).sum(-1, true) + 1e-6f);
        q_f = q_f / std::sqrt(static_cast<float>(hkd));

        // ── Gating signals ───────────────────────────────────────────────────
        auto v_f = v.to(torch::kFloat32);
        auto a_f = a.to(torch::kFloat32);
        auto b_f = b.to(torch::kFloat32);

        auto beta = torch::sigmoid(b_f);  // [B, S, nv]
        // GGUF stores ssm_a after conversion as -exp(HF A_log), matching
        // llama.cpp's direct multiply; applying exp again corrupts decay.
        auto A       = w.A_log.to(torch::kFloat32);                    // [nv], negative
        auto dt_b   = w.dt_bias.to(torch::kFloat32);                  // [nv]
        auto dt_sp  = torch::softplus(a_f + dt_b);                    // [B, S, nv]
        auto g_log  = A * dt_sp;                                      // [B, S, nv]
        auto g_fac  = g_log.exp();                                     // [B, S, nv] decay ∈ (0,1)

        // ── Recurrent gated delta rule ───────────────────────────────────────
        auto state  = cache->recurrent_state;  // [B, nv, hkd, hvd] float32
        auto out_buf = torch::zeros({B, S, nv, hvd}, torch::kFloat32).to(hidden.device());

        for (int t = 0; t < S; ++t)
        {
            auto q_t    = q_f.select(1, t);     // [B, nv, hkd]
            auto k_t    = k_f.select(1, t);
            auto v_t    = v_f.select(1, t);     // [B, nv, hvd]
            auto g_t    = g_fac.select(1, t).unsqueeze(-1).unsqueeze(-1); // [B,nv,1,1]
            auto beta_t = beta.select(1, t).unsqueeze(-1);  // [B, nv, 1]

            // Decay first, then read kv_mem from the decayed state (HF Qwen3NextRMSNormGated ref)
            state = state * g_t;
            auto kv_mem = (state * k_t.unsqueeze(-1)).sum(-2);  // [B, nv, hvd]
            auto delta  = (v_t - kv_mem) * beta_t;              // [B, nv, hvd]
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
            out_buf.select(1, t).copy_((state * q_t.unsqueeze(-1)).sum(-2));
        }
        cache->recurrent_state = state;
        cache->len += S;

        // ── Gated RMSNorm ────────────────────────────────────────────────────
        auto out = out_buf.to(hidden.scalar_type());  // [B, S, nv, hvd]
        out = out.reshape({B * S * nv, hvd});
        auto z2 = z.reshape({B * S * nv, hvd});

        // Gated RMSNorm: normalize FIRST, then gate (HF Qwen3NextRMSNormGated: # Norm before gate)
        auto out_f32 = out.to(torch::kFloat32);
        auto var = (out_f32 * out_f32).mean(-1, true);
        out_f32 = out_f32 * torch::rsqrt(var + static_cast<double>(config_.rms_norm_eps));
        out_f32 = out_f32 * w.norm.to(torch::kFloat32) * torch::silu(z2.to(torch::kFloat32));
        out = out_f32.to(hidden.scalar_type());

        out = out.reshape({B, S, vdim});

        // ── Output projection ────────────────────────────────────────────────
        auto ssm_out = torch::mm(out.reshape({B * S, vdim}), w.out_proj.t())
                           .reshape({B, S, H});

        // ── First residual connection ────────────────────────────────────────
        h = residual + ssm_out;

        // ── MLP block ────────────────────────────────────────────────────────
        residual = h;
        h = RMSNorm::Forward(h, lw.post_attention_layernorm, config_.rms_norm_eps);
        h = MLP::Forward(h, lw.w_gate, lw.w_up, lw.w_down);
        return residual + h;
    }
}
