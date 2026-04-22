#pragma once

#include <string>
#include <memory>
#include <torch/torch.h>

namespace mllm
{
    struct ModelConfig
    {
        std::string model_name;
        int hidden_size = 0;
        int num_layers = 0;
        int num_attention_heads = 0;
        int num_key_value_heads = 0;
        int vocab_size = 0;
        int max_position_embeddings = 0;
        int intermediate_size = 0;
        int head_dim = 0;
        float rms_norm_eps = 1e-5f;
        float rope_theta = 10000.0f;
    };

    class IModelRunner
    {
    public:
        virtual ~IModelRunner() = default;

        // config.json + weights + tokenizer metadata 로드
        virtual bool Load(const std::string& model_path) = 0;

        // 1 step forward
        virtual torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) = 0;

        // KV cache 초기화
        virtual void InitKVCache(int batch_size, int max_seq_len) = 0;

        // config 접근
        virtual const ModelConfig& GetConfig() const = 0;

        // 모델 타입 식별
        virtual std::string GetModelType() const = 0;
    };

    using ModelRunnerPtr = std::shared_ptr<IModelRunner>;
}
