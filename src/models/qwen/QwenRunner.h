// src/models/qwen/QwenRunner.h

#pragma once

#include "models/base/IModelRunner.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "core/runtime/TransformerBlock.h"

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/base/GenerateOptions.h"
#include "tokenizer/ITokenizer.h"

namespace mllm
{
    class QwenRunner : public IModelRunner
    {
    public:
        QwenRunner() = default;
        ~QwenRunner() override = default;

        bool Load(const std::string& model_path) override;

        torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) override;

        std::vector<int64_t> Generate(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options) override;

        void InitKVCache(
            int batch_size,
            int max_seq_len) override;

        const ModelConfig& GetConfig() const override;

        std::string GetModelType() const override;

        void SetParityMode(bool enabled) override;

        void SetParityReferenceDir(
            const std::string& path) override;

    private:
        bool LoadConfig(
            const std::string& config_path);

        torch::Tensor& LoadWeight(
            const std::string& name);

        void LoadAllWeights();

        std::vector<int64_t> GenerateInternal(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options,
            bool streaming,
            ITokenizer* tokenizer);

    private:
        ModelConfig config_;

        std::string model_path_;

        std::unordered_map<
            std::string,
            TensorMeta
        > tensor_map_;

        std::unordered_map<
            std::string,
            torch::Tensor
        > weights_;

        std::vector<LayerWeights>
            layer_weights_;

        std::vector<KVCache>
            kv_caches_;

        bool is_loaded_ = false;
        bool parity_mode_ = false;
        std::string parity_reference_dir_ =
            "../scripts/parity";
    };
}
