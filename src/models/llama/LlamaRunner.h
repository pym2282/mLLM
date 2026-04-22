#pragma once

#include "models/base/IModelRunner.h"
#include <torch/script.h>

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

    private:
        ModelConfig config_;

        // 초기 단계에서는 TorchScript module 사용
        // 이후 safetensors 직접 로딩으로 확장 예정
        torch::jit::script::Module module_;

        bool is_loaded_ = false;

        torch::Tensor embedding_weight_;
    };
}
