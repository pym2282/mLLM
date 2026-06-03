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

        GenerateResult Generate(
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

        // .gguf path: load config + weights via GgufLoader
        bool LoadGguf(const std::string& gguf_path);

        torch::Tensor& LoadWeight(
            const std::string& name);

        // Returns undefined tensor if name not in tensor_map_ or weights_
        torch::Tensor TryLoadWeight(
            const std::string& name);

        void LoadAllWeights();
        void EnsureOnGPU();  // Lazy: 첫 Generate() 호출 시 CPU→GPU 이동

        // Forward pass for one Qwen3.5 hybrid (Gated DeltaNet) layer
        torch::Tensor ForwardHybridLayer(
            const torch::Tensor& hidden,
            const LinearAttnWeights& law,
            const LayerWeights& lw,
            SSMCache* cache);

    private:
        ModelConfig config_;

        std::string model_path_;
        std::string pending_cache_path_;
        std::string pending_cache_src_;
        bool gpu_ready_ = false;  // GPU 전송 완료 여부

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

        std::vector<LinearAttnWeights>
            linear_attn_weights_;

        std::vector<KVCache>
            kv_caches_;

        std::vector<SSMCache>
            ssm_caches_;

        std::vector<bool>
            layer_is_hybrid_;

        // Total tokens processed since last prefill (0 = no prefill done)
        int prefilled_tokens_ = 0;

        bool is_loaded_ = false;
        bool parity_mode_ = false;
        std::string parity_reference_dir_ =
            "../scripts/parity";
    };
}
