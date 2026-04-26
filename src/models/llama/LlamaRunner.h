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

        bool Load(const std::string& model_path) override;

        torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) override;

        std::vector<int64_t> Generate(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options);

        void InitKVCache(int batch_size, int max_seq_len) override;

        const ModelConfig& GetConfig() const override;

        std::string GetModelType() const override;

    private:
        bool LoadConfig(const std::string& config_path);

        // Read tensor from model.safetensors using cached metadata and cache
        // the handle in weights_. Throws on failure. Returns a reference to
        // the cached tensor so it can be stored in a LayerWeights view.
        torch::Tensor& LoadWeight(const std::string& name);

        // Load every weight needed by Forward:
        //   model.embed_tokens.weight
        //   model.norm.weight
        //   lm_head.weight              (skipped if tie_word_embeddings)
        //   per layer:
        //     input_layernorm, post_attention_layernorm
        //     self_attn.{q,k,v,o}_proj
        //     mlp.{gate,up,down}_proj
        // Also populates layer_weights_ with tensor views per layer.
        void LoadAllWeights();

    private:
        ModelConfig config_;

        std::string model_path_;

        // safetensors header metadata (offsets, shapes, dtypes).
        std::unordered_map<std::string, TensorMeta> tensor_map_;

        // Owning storage for every loaded tensor, keyed by HF tensor name.
        std::unordered_map<std::string, torch::Tensor> weights_;

        // Per-layer tensor views into weights_. Size = config_.num_layers.
        std::vector<LayerWeights> layer_weights_;

        std::vector<KVCache> kv_caches_;

        bool is_loaded_ = false;
    };
}
