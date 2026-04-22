#pragma once

#include "models/base/IModelRunner.h"
#include "models/base/SafeTensorHeaderParser.h"

#include <torch/torch.h>

#include <string>
#include <unordered_map>

namespace mllm
{
    class LlamaRunner : public IModelRunner
    {
    public:
        LlamaRunner();
        ~LlamaRunner() override = default;

        bool Load(const std::string& model_path) override;

        torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) override;

        void InitKVCache(int batch_size, int max_seq_len) override;

        const ModelConfig& GetConfig() const override;

        std::string GetModelType() const override;

    private:
        bool LoadConfig(const std::string& config_path);

        // Read tensor from model.safetensors using cached metadata map and
        // insert it into weights_. Throws on failure.
        torch::Tensor& LoadWeight(const std::string& name);

        // Bulk load all RMSNorm weights for this architecture:
        //   model.embed_tokens.weight
        //   model.norm.weight
        //   model.layers.{i}.input_layernorm.weight
        //   model.layers.{i}.post_attention_layernorm.weight
        void LoadEmbeddingAndNormWeights();

    private:
        ModelConfig config_;

        std::string model_path_;

        // safetensors header metadata (offsets, shapes, dtypes) kept for
        // on-demand weight loading in future steps.
        std::unordered_map<std::string, TensorMeta> tensor_map_;

        // Loaded weight storage, keyed by safetensors tensor name.
        std::unordered_map<std::string, torch::Tensor> weights_;

        bool is_loaded_ = false;
    };
}
