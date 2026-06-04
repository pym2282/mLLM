#pragma once

#include "models/base/BaseModelRunner.h"
#include "core/runtime/TransformerBlock.h"

#include <torch/torch.h>
#include <string>

#include "models/base/GenerateOptions.h"

namespace mllm
{
    class LlamaRunner : public BaseModelRunner
    {
    public:
        LlamaRunner() = default;
        ~LlamaRunner() override = default;

        bool Load(const std::string& model_path) override;

        torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) override;

        GenerateResult Generate(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options) override;

        void InitKVCache(int batch_size, int max_seq_len) override;

        std::string GetModelType() const override;

    private:
        bool LoadConfig(const std::string& config_path);
        void LoadAllWeights();
    };
}
