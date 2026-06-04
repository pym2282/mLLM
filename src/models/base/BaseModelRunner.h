// src/models/base/BaseModelRunner.h
//
// Shared base for LlamaRunner, GemmaRunner, QwenRunner.
// Holds the members and helpers that are identical in all three runners.
// Model-specific state (AltUP, SSM/hybrid, gpu_ready) stays in each runner.

#pragma once

#include "models/base/IModelRunner.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"

#include "core/runtime/TransformerBlock.h"
#include "core/runtime/KVCache.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mllm
{
    class BaseModelRunner : public IModelRunner
    {
    public:
        // Implemented here — all runners return config_ identically.
        const ModelConfig& GetConfig() const override { return config_; }

    protected:
        ModelConfig   config_;
        std::string   model_path_;

        // SafeTensor metadata (populated by SafeTensorHeaderParser::Parse)
        std::unordered_map<std::string, TensorMeta>    tensor_map_;

        // Weight storage (populated by GgufLoader::Load or lazy SafeTensor loads)
        std::unordered_map<std::string, torch::Tensor> weights_;

        // Per-layer weights assembled after loading
        std::vector<LayerWeights> layer_weights_;

        // KV caches — one per layer, pre-allocated or dynamic
        std::vector<KVCache> kv_caches_;

        bool is_loaded_ = false;

        // SafeTensor lazy-load: returns cached or newly loaded tensor.
        // For GGUF models weights_ is pre-populated; find() returns immediately.
        torch::Tensor& LoadWeight(const std::string& name)
        {
            auto it = weights_.find(name);
            if (it != weights_.end()) return it->second;
            auto t = SafeTensorTensorLoader::LoadTensor(model_path_, name, tensor_map_);
            auto [ins, _] = weights_.emplace(name, std::move(t));
            return ins->second;
        }
    };
}
