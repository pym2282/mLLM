// src/models/qwen/QwenRunner.h

#pragma once

#include "models/base/BaseModelRunner.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "core/runtime/TransformerBlock.h"

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/base/GenerateOptions.h"

namespace mllm
{
    class QwenRunner : public BaseModelRunner
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

        std::string GetModelType() const override;

        void SetParityMode(bool enabled) override;

        void SetParityReferenceDir(
            const std::string& path) override;

        KVSnapshot GetKVSnapshot(int64_t len) const override;
        void SetKVSnapshot(const KVSnapshot& snap) override;

    private:
        bool LoadConfig(
            const std::string& config_path);

        // .gguf path: load config + weights via GgufLoader
        bool LoadGguf(const std::string& gguf_path);

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
        // Qwen-specific state (not in BaseModelRunner)
        std::string pending_cache_path_;
        std::string pending_cache_src_;
        bool gpu_ready_ = false;  // lazy GPU transfer flag

        std::vector<LinearAttnWeights> linear_attn_weights_;
        std::vector<SSMCache>          ssm_caches_;
        std::vector<bool>              layer_is_hybrid_;

        int     prefilled_tokens_ = 0;
        int64_t prefix_kv_len_    = 0;

        bool parity_mode_ = false;
        std::string parity_reference_dir_ = "../scripts/parity";
    };
}
