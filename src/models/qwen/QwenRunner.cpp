// src/models/qwen/QwenRunner.cpp

#include "models/qwen/QwenRunner.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "models/base/GenerateResult.h"

#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorTensorLoader.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/Sampler.h"
#include "core/runtime/TransformerBlock.h"
#include "core/runtime/RMSNorm.h"
#include "debug/TensorCompare.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace mllm
{
    bool QwenRunner::Load(const std::string& model_path)
    {
        try
        {
            model_path_ = model_path;
            if (!LoadConfig(model_path + "/config.json"))
            {
                std::cerr << "[QwenRunner] Failed to load config" << std::endl;
                return false;
            }

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
            {
                std::cerr << "[QwenRunner] Failed to parse safetensors header" << std::endl;
                return false;
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
        if (tensor_map_.find(name) == tensor_map_.end())
            return {};
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
        {
            LoadWeight("lm_head.weight");
        }

        layer_weights_.clear();
        layer_weights_.reserve(config_.num_layers);

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string p =
                "model.layers." + std::to_string(i);

            LayerWeights lw;

            lw.input_layernorm =
                LoadWeight(p + ".input_layernorm.weight");

            lw.post_attention_layernorm =
                LoadWeight(p + ".post_attention_layernorm.weight");

            lw.w_q =
                LoadWeight(p + ".self_attn.q_proj.weight");
            lw.w_q_scale =
                TryLoadWeight(p + ".self_attn.q_proj.weight_scale_inv");

            lw.w_k =
                LoadWeight(p + ".self_attn.k_proj.weight");
            lw.w_k_scale =
                TryLoadWeight(p + ".self_attn.k_proj.weight_scale_inv");

            lw.w_v =
                LoadWeight(p + ".self_attn.v_proj.weight");
            lw.w_v_scale =
                TryLoadWeight(p + ".self_attn.v_proj.weight_scale_inv");

            lw.w_o =
                LoadWeight(p + ".self_attn.o_proj.weight");
            lw.w_o_scale =
                TryLoadWeight(p + ".self_attn.o_proj.weight_scale_inv");

            lw.w_q_norm =
                LoadWeight(p + ".self_attn.q_norm.weight");

            lw.w_k_norm =
                LoadWeight(p + ".self_attn.k_norm.weight");

            lw.w_gate =
                LoadWeight(p + ".mlp.gate_proj.weight");
            lw.w_gate_scale =
                TryLoadWeight(p + ".mlp.gate_proj.weight_scale_inv");

            lw.w_up =
                LoadWeight(p + ".mlp.up_proj.weight");
            lw.w_up_scale =
                TryLoadWeight(p + ".mlp.up_proj.weight_scale_inv");

            lw.w_down =
                LoadWeight(p + ".mlp.down_proj.weight");
            lw.w_down_scale =
                TryLoadWeight(p + ".mlp.down_proj.weight_scale_inv");

            layer_weights_.push_back(std::move(lw));
        }

        const size_t expected =
            2 +
            (config_.tie_word_embeddings ? 0 : 1) +
            static_cast<size_t>(config_.num_layers) * 11;

        std::cout
            << "[QwenRunner] Loaded "
            << weights_.size()
            << " tensors (expected="
            << expected
            << ")"
            << std::endl;

        if (torch::cuda::is_available())
        {
            std::cout << "[QwenRunner] Moving weights to CUDA..." << std::endl;
            // Move all weights to CUDA once.
            for (auto& [name, w] : weights_)
                w = w.to(torch::kCUDA);

            // Rebuild layer_weights_ as aliases into the already-CUDA weights_ map.
            // This avoids a second GPU copy of every projection weight.
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
                lw.w_q      = weights_.at(p + ".self_attn.q_proj.weight");
                lw.w_k      = weights_.at(p + ".self_attn.k_proj.weight");
                lw.w_v      = weights_.at(p + ".self_attn.v_proj.weight");
                lw.w_o      = weights_.at(p + ".self_attn.o_proj.weight");
                lw.w_q_norm = weights_.at(p + ".self_attn.q_norm.weight");
                lw.w_k_norm = weights_.at(p + ".self_attn.k_norm.weight");
                lw.w_gate   = weights_.at(p + ".mlp.gate_proj.weight");
                lw.w_up     = weights_.at(p + ".mlp.up_proj.weight");
                lw.w_down   = weights_.at(p + ".mlp.down_proj.weight");
                lw.w_q_scale    = try_get(p + ".self_attn.q_proj.weight_scale_inv");
                lw.w_k_scale    = try_get(p + ".self_attn.k_proj.weight_scale_inv");
                lw.w_v_scale    = try_get(p + ".self_attn.v_proj.weight_scale_inv");
                lw.w_o_scale    = try_get(p + ".self_attn.o_proj.weight_scale_inv");
                lw.w_gate_scale = try_get(p + ".mlp.gate_proj.weight_scale_inv");
                lw.w_up_scale   = try_get(p + ".mlp.up_proj.weight_scale_inv");
                lw.w_down_scale = try_get(p + ".mlp.down_proj.weight_scale_inv");
            }
            std::cout << "[QwenRunner] Weights on CUDA." << std::endl;

            // Return staging/fragmented memory to CUDA so temporaries
            // during inference have more room.
            c10::cuda::CUDACachingAllocator::emptyCache();

            const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            std::cout
                << "[QwenRunner] VRAM reserved="
                << stats.reserved_bytes[0].current / (1024*1024)
                << "MB allocated="
                << stats.allocated_bytes[0].current / (1024*1024)
                << "MB\n";
        }
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

        const auto S = input_ids.size(1);
        const auto target_device = weights_.at("model.embed_tokens.weight").device();

        torch::Tensor position_ids;

        const bool is_decode =
            (S == 1) &&
            !kv_caches_.empty() &&
            kv_caches_[0].IsInitialized();

        if (is_decode)
        {
            position_ids = torch::tensor(
                { kv_caches_[0].key.size(2) },
                torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(target_device)
            );
        }
        else
        {
            position_ids = torch::arange(
                0,
                S,
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
            const bool is_fp8 =
                layer_weights_[i].w_q.defined() &&
                layer_weights_[i].w_q.scalar_type() == torch::kFloat8_e4m3fn;

            LayerWeights dequant_lw;
            if (is_fp8)
            {
                dequant_lw = DequantizeLayerWeights(
                    layer_weights_[i], hidden.scalar_type());
            }

            const LayerWeights& lw = is_fp8
                ? dequant_lw
                : layer_weights_[i];

            hidden = TransformerBlock::Forward(
                hidden,
                lw,
                config_.num_attention_heads,
                config_.num_key_value_heads,
                config_.head_dim,
                static_cast<double>(config_.rope_theta),
                static_cast<double>(config_.rms_norm_eps),
                true,
                position_ids,
                &kv_caches_[i]
            );
        }

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

        const torch::Tensor& lm_head_w =
            (weights_.count("lm_head.weight") > 0)
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

        GenerateResult result;
        std::vector<int64_t> current = input_ids;

        for (auto& cache : kv_caches_)
        {
            cache.Clear();
        }

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
                input_tensor = torch::tensor(
                    current,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                ).unsqueeze(0);
            }
            else
            {
                input_tensor = torch::tensor(
                    std::vector<int64_t>{ current.back() },
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
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

    void QwenRunner::InitKVCache(
        int batch_size,
        int max_seq_len)
    {
        // Parameters are hints for future pre-allocation; currently caches
        // grow dynamically via torch::cat in TransformerBlock.
        (void)batch_size;
        (void)max_seq_len;

        kv_caches_.clear();
        kv_caches_.resize(config_.num_layers);
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
}
