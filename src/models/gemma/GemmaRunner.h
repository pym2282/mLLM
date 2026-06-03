// src/models/gemma/GemmaRunner.h

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
    class GemmaRunner : public IModelRunner
    {
    public:
        GemmaRunner() = default;
        ~GemmaRunner() override = default;

        bool Load(const std::string& model_path) override;

        torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) override;

        GenerateResult Generate(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options) override;

        void InitKVCache(int batch_size, int max_seq_len) override;

        const ModelConfig& GetConfig() const override;

        std::string GetModelType() const override;

    private:
        bool LoadConfig(const std::string& config_path);
        bool LoadGguf(const std::string& gguf_path);

        torch::Tensor& LoadWeight(const std::string& name);
        void LoadAllWeights();
        void LoadLayerWeights();   // assemble layer_weights_ from weights_ (call after CUDA move)
        void MoveWeightsToCuda();  // no-op placeholder kept for compatibility

        // Returns true if layer i uses full (global) attention
        bool IsGlobalLayer(int i) const;

    private:
        ModelConfig config_;
        std::string model_path_;

        std::unordered_map<std::string, TensorMeta>     tensor_map_;
        std::unordered_map<std::string, torch::Tensor>  weights_;
        std::vector<LayerWeights>                        layer_weights_;
        std::vector<KVCache>                             kv_caches_;
        torch::Tensor  per_layer_token_embd_;    // [vocab, num_layers*D_ple]
        torch::Tensor  per_layer_model_proj_;    // [num_layers*D_ple, H]
        torch::Tensor  per_layer_proj_norm_;     // [D_ple_per_layer]

        bool is_loaded_ = false;
        bool norm_plain_weight_ = false;  // true for Gemma 4: RMSNorm uses w (not 1+w)
    };
}
