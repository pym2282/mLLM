#include "models/llama/LlamaRunner.h"

#include <iostream>
#include <stdexcept>

#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorLoader.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/RMSNorm.h"
#include "core/runtime/Linear.h"
#include "core/runtime/TransformerBlock.h"

namespace mllm
{
    LlamaRunner::LlamaRunner() = default;

    bool LlamaRunner::Load(const std::string& model_path)
    {
        try
        {
            model_path_ = model_path;

            const std::string config_path = model_path + "/config.json";
            if (!LoadConfig(config_path))
            {
                std::cerr << "[LlamaRunner] Failed to load config: "
                          << config_path << std::endl;
                return false;
            }

            if (!SafeTensorLoader::Exists(model_path))
            {
                std::cerr << "[LlamaRunner] model.safetensors not found"
                          << std::endl;
                return false;
            }

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
            {
                std::cerr << "[LlamaRunner] Failed to parse safetensors header"
                          << std::endl;
                return false;
            }

            LoadAllWeights();

            is_loaded_ = true;

            std::cout << "[LlamaRunner] Model loaded: " << model_path
                      << "  weights=" << weights_.size()
                      << "  layers=" << layer_weights_.size() << std::endl;
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[LlamaRunner] Load failed:\n" << e.what() << std::endl;
            is_loaded_ = false;
            return false;
        }
    }

    torch::Tensor& LlamaRunner::LoadWeight(const std::string& name)
    {
        auto existing = weights_.find(name);
        if (existing != weights_.end())
        {
            return existing->second;
        }

        auto tensor = SafeTensorTensorLoader::LoadTensor(
            model_path_, name, tensor_map_);

        auto [it, inserted] = weights_.emplace(name, std::move(tensor));
        (void)inserted;
        return it->second;
    }

    void LlamaRunner::LoadAllWeights()
    {
        // Global
        LoadWeight("model.embed_tokens.weight");
        LoadWeight("model.norm.weight");

        if (!config_.tie_word_embeddings)
        {
            LoadWeight("lm_head.weight");
        }

        // Per-layer
        layer_weights_.clear();
        layer_weights_.reserve(config_.num_layers);

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string p = "model.layers." + std::to_string(i);

            LayerWeights lw;
            lw.input_layernorm =
                LoadWeight(p + ".input_layernorm.weight");
            lw.post_attention_layernorm =
                LoadWeight(p + ".post_attention_layernorm.weight");

            lw.w_q = LoadWeight(p + ".self_attn.q_proj.weight");
            lw.w_k = LoadWeight(p + ".self_attn.k_proj.weight");
            lw.w_v = LoadWeight(p + ".self_attn.v_proj.weight");
            lw.w_o = LoadWeight(p + ".self_attn.o_proj.weight");

            lw.w_gate = LoadWeight(p + ".mlp.gate_proj.weight");
            lw.w_up   = LoadWeight(p + ".mlp.up_proj.weight");
            lw.w_down = LoadWeight(p + ".mlp.down_proj.weight");

            layer_weights_.push_back(std::move(lw));
        }

        const size_t expected = 2
            + (config_.tie_word_embeddings ? 0 : 1)
            + static_cast<size_t>(config_.num_layers) * 9;

        std::cout << "[LlamaRunner] Loaded all weights: "
                  << weights_.size() << " tensors"
                  << " (per-layer views=" << layer_weights_.size()
                  << ", expected=" << expected << ")"
                  << std::endl;

        if (weights_.size() != expected)
        {
            std::cerr << "[LlamaRunner] WEIGHT COUNT MISMATCH — "
                      << "expected " << expected
                      << ", got " << weights_.size() << std::endl;
        }
    }

    torch::Tensor LlamaRunner::Forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& /*attention_mask*/)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error("Model is not loaded.");
        }

        const auto S = input_ids.size(1);

        std::cout << "[LlamaRunner] Forward started"
                  << "  input_ids shape=" << input_ids.sizes() << std::endl;

        // Positions for prefill: [0, 1, ..., S-1]. Shared across batch.
        // When KV cache lands, decode step will override this to [cache_len].
        auto position_ids = torch::arange(
            0, S,
            torch::TensorOptions()
                .dtype(torch::kInt64)
                .device(input_ids.device()));

        // Step 1: embedding lookup
        auto hidden = EmbeddingLookup::Forward(
            input_ids,
            weights_.at("model.embed_tokens.weight"));

        // Step 2: N transformer blocks
        for (int i = 0; i < config_.num_layers; ++i)
        {
            hidden = TransformerBlock::Forward(
                hidden,
                layer_weights_[i],
                config_.num_attention_heads,
                config_.num_key_value_heads,
                config_.head_dim,
                static_cast<double>(config_.rope_theta),
                static_cast<double>(config_.rms_norm_eps),
                position_ids);
        }

        // Step 3: final RMSNorm
        hidden = RMSNorm::Forward(
            hidden,
            weights_.at("model.norm.weight"),
            config_.rms_norm_eps);

        // Step 4: LM head → logits
        const torch::Tensor& lm_head_w = config_.tie_word_embeddings
            ? weights_.at("model.embed_tokens.weight")
            : weights_.at("lm_head.weight");

        auto logits = Linear::Forward(hidden, lm_head_w);

        std::cout << "[LlamaRunner] Forward step complete"
                  << "  logits shape=" << logits.sizes()
                  << "  dtype=" << logits.scalar_type() << std::endl;

        // Probe last-token logits — show first 5 values + argmax.
        {
            auto last = logits.index({0, S - 1}).to(torch::kFloat32);
            auto head5 = last.slice(0, 0, 5);
            auto top = torch::argmax(last).item<int64_t>();
            std::cout << "[LlamaRunner] logits[0,-1,:5]: " << head5
                      << std::endl;
            std::cout << "[LlamaRunner] last-token argmax token_id=" << top
                      << std::endl;
        }

        return logits;
    }

    void LlamaRunner::InitKVCache(int batch_size, int max_seq_len)
    {
        // TODO: real KV cache manager.
        std::cout << "[LlamaRunner] InitKVCache()"
                  << " batch=" << batch_size
                  << " max_seq=" << max_seq_len
                  << std::endl;
    }

    const ModelConfig& LlamaRunner::GetConfig() const
    {
        return config_;
    }

    std::string LlamaRunner::GetModelType() const
    {
        return "llama";
    }

    bool LlamaRunner::LoadConfig(const std::string& config_path)
    {
        return LoadModelConfigFromJson(config_path, config_);
    }
}
