#include "models/llama/LlamaRunner.h"

#include <iostream>
#include <unordered_map>
#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorLoader.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"
#include "core/runtime/EmbeddingLookup.h"

namespace mllm
{
    LlamaRunner::LlamaRunner()
    {
    }

    // LlamaRunner.cpp 내부 Load() 수정 예시
    // include 추가:
    // #include "models/base/SafeTensorLoader.h"

    bool LlamaRunner::Load(const std::string& model_path)
    {
        try
        {
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

            // 새로 추가: safetensors header parsing
            std::unordered_map<std::string, mllm::TensorMeta> tensor_map;

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map))
            {
                std::cerr << "[LlamaRunner] Failed to parse safetensors header"
                          << std::endl;
                return false;
            }
            try
            {
                auto embedding_tensor = SafeTensorTensorLoader::LoadTensor(
                    model_path,
                    "model.embed_tokens.weight",
                    tensor_map
                );

                std::cout << "[LlamaRunner] Embedding tensor loaded" << std::endl;
                std::cout << "  shape: " << embedding_tensor.sizes() << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[LlamaRunner] Embedding tensor load failed:\n"
                          << e.what() << std::endl;

                return false;
            }

            // 예시 tensor 확인
            if (tensor_map.find("model.embed_tokens.weight") != tensor_map.end())
            {
                const auto& meta = tensor_map["model.embed_tokens.weight"];

                std::cout << "[LlamaRunner] Found embedding tensor" << std::endl;
                std::cout << "  dtype: " << meta.dtype << std::endl;
                std::cout << "  shape size: " << meta.shape.size() << std::endl;
            }

            embedding_weight_ = SafeTensorTensorLoader::LoadTensor(
                model_path,
                "model.embed_tokens.weight",
                tensor_map
            );

            std::cout << "[LlamaRunner] Embedding tensor cached" << std::endl;

            is_loaded_ = true;

            std::cout << "[LlamaRunner] Model directory loaded: "
                      << model_path << std::endl;

            std::cout << "[LlamaRunner] Config loaded successfully"
                      << std::endl;

            std::cout << "[LlamaRunner] SafeTensor check passed"
                      << std::endl;

            std::cout << "[LlamaRunner] Header parsing complete"
                      << std::endl;

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[LlamaRunner] Failed to load model:\n"
                      << e.what() << std::endl;

            is_loaded_ = false;
            return false;
        }
    }


    torch::Tensor LlamaRunner::Forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error("Model is not loaded.");
        }

        if (!embedding_weight_.defined())
        {
            throw std::runtime_error("Embedding weight is not loaded.");
        }

        std::cout << "[LlamaRunner] Real Forward started" << std::endl;

        // Step 1:
        // token ids -> embedding vectors
        auto hidden_states = mllm::EmbeddingLookup::Forward(
            input_ids,
            embedding_weight_
        );

        // TODO:
        // RMSNorm
        // Attention
        // MLP
        // LM Head

        // 현재는 embedding output을 기반으로 fake logits 반환
        auto logits = torch::rand(
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
        // TODO:
        // 실제 KV cache manager 구현 예정
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
