#pragma once

#include "models/base/IModelRunner.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "core/runtime/TransformerBlock.h"

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "models/base/GenerateOptions.h"
namespace mllm
{
    class LlamaRunner : public IModelRunner
    {
    public:
        LlamaRunner();
        ~LlamaRunner() override = default;

        bool Load(
            const std::string& model_path
        ) override;

        torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask
        ) override;

        GenerateResult Generate(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options
        ) override;

        void InitKVCache(
            int batch_size,
            int max_seq_len
        ) override;

        const ModelConfig& GetConfig() const override;

        std::string GetModelType() const override;

    private:
        bool LoadConfig(
            const std::string& config_path
        );

        torch::Tensor& LoadWeight(
            const std::string& name
        );

        void LoadAllWeights();

    private:
        ModelConfig config_;

        std::string model_path_;

        // safetensors header metadata
        std::unordered_map<
            std::string,
            TensorMeta
        > tensor_map_;

        // loaded tensors
        std::unordered_map<
            std::string,
            torch::Tensor
        > weights_;

        // per-layer views
        std::vector<
            LayerWeights
        > layer_weights_;

        std::vector<
            KVCache
        > kv_caches_;

        bool is_loaded_ = false;
    };
}