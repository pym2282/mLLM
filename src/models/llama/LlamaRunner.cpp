#include "models/llama/LlamaRunner.h"

#include <iostream>
#include <stdexcept>

#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorLoader.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/RMSNorm.h"

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

            LoadEmbeddingAndNormWeights();

            is_loaded_ = true;

            std::cout << "[LlamaRunner] Model loaded: " << model_path
                      << "  weights=" << weights_.size() << std::endl;
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

    void LlamaRunner::LoadEmbeddingAndNormWeights()
    {
        LoadWeight("model.embed_tokens.weight");
        LoadWeight("model.norm.weight");

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string prefix = "model.layers." + std::to_string(i);
            LoadWeight(prefix + ".input_layernorm.weight");
            LoadWeight(prefix + ".post_attention_layernorm.weight");
        }

        std::cout << "[LlamaRunner] Loaded embedding + "
                  << (1 + 2 * config_.num_layers)
                  << " RMSNorm weights" << std::endl;
    }

    torch::Tensor LlamaRunner::Forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& /*attention_mask*/)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error("Model is not loaded.");
        }

        std::cout << "[LlamaRunner] Forward started" << std::endl;

        // Step 1: token ids -> embedding vectors
        auto hidden_states = EmbeddingLookup::Forward(
            input_ids,
            weights_.at("model.embed_tokens.weight")
        );

        // Step 2: Layer 0 input_layernorm ONLY.
        // Chaining 44 norms without Attention/MLP between them is numerically
        // meaningless — we expose exactly one norm so it can be verified
        // 1:1 against HF Python before building out the rest of the block.
        auto& layer0_pre_norm_w =
            weights_.at("model.layers.0.input_layernorm.weight");
        hidden_states = RMSNorm::Forward(
            hidden_states, layer0_pre_norm_w, config_.rms_norm_eps);

        std::cout << "[LlamaRunner] Layer0 input_layernorm output shape: "
                  << hidden_states.sizes()
                  << "  dtype=" << hidden_states.scalar_type()
                  << std::endl;

        // Probe values for HF parity check (first 5 elems of token 0).
        {
            auto probe = hidden_states.index({0, 0}).to(torch::kFloat32)
                                      .slice(0, 0, 5);
            std::cout << "[LlamaRunner] probe hidden[0,0,0:5]: "
                      << probe << std::endl;
        }

        // Step 4: LM Head — still placeholder (needs lm_head.weight matmul).
        auto logits = torch::zeros(
            {
                input_ids.size(0),
                input_ids.size(1),
                config_.vocab_size
            },
            torch::kFloat32
        );

        std::cout << "[LlamaRunner] Forward step complete" << std::endl;
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
