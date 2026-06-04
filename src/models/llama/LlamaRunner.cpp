#include "models/llama/LlamaRunner.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "core/Logger.h"
#include "core/MllmException.h"

#include "models/base/GenerateResult.h"

#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorHeaderParser.h"
#include "models/base/SafeTensorTensorLoader.h"
#include "models/base/GenerateOptions.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/Sampler.h"
#include "core/runtime/TransformerBlock.h"
#include "core/runtime/RMSNorm.h"
#include "core/runtime/Linear.h"

namespace mllm
{
    LlamaRunner::LlamaRunner()
        : is_loaded_(false)
    {
    }

    bool LlamaRunner::Load(const std::string& model_path)
    {
        try
        {
            model_path_ = model_path;

            const std::string config_path =
                model_path + "/config.json";

            if (!LoadConfig(config_path))
            {
                MLLM_ERROR("LlamaRunner", "Failed to load config: " + config_path);
                return false;
            }

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
            {
                MLLM_ERROR("LlamaRunner", "Failed to parse safetensors header");
                return false;
            }

            LoadAllWeights();

            kv_caches_.clear();
            kv_caches_.resize(config_.num_layers);

            is_loaded_ = true;

            MLLM_INFO("LlamaRunner", "Loaded layers=" + std::to_string(config_.num_layers)
                + " weights=" + std::to_string(weights_.size())
                + " kv=" + std::to_string(kv_caches_.size()));

            return true;
        }
        catch (const std::exception& e)
        {
            MLLM_ERROR("LlamaRunner", "Load failed: " + std::string(e.what()));
            is_loaded_ = false;
            return false;
        }
    }

    torch::Tensor& LlamaRunner::LoadWeight(
        const std::string& name)
    {
        auto existing = weights_.find(name);

        if (existing != weights_.end())
        {
            return existing->second;
        }

        auto tensor =
            SafeTensorTensorLoader::LoadTensor(
                model_path_,
                name,
                tensor_map_);

        auto [it, inserted] =
            weights_.emplace(
                name,
                std::move(tensor));

        (void)inserted;

        return it->second;
    }

    void LlamaRunner::LoadAllWeights()
    {
        LoadWeight("model.embed_tokens.weight");
        LoadWeight("model.norm.weight");

        if (!config_.tie_word_embeddings)
        {
            LoadWeight("lm_head.weight");
        }

        layer_weights_.clear();
        layer_weights_.reserve(config_.num_layers);

        for (int i = 0; i < config_.num_layers; ++i)
        {
            const std::string p =
                "model.layers." +
                std::to_string(i);

            LayerWeights lw;

            lw.input_layernorm =
                LoadWeight(
                    p + ".input_layernorm.weight");

            lw.post_attention_layernorm =
                LoadWeight(
                    p + ".post_attention_layernorm.weight");

            lw.w_q =
                LoadWeight(
                    p + ".self_attn.q_proj.weight");

            lw.w_k =
                LoadWeight(
                    p + ".self_attn.k_proj.weight");

            lw.w_v =
                LoadWeight(
                    p + ".self_attn.v_proj.weight");

            lw.w_o =
                LoadWeight(
                    p + ".self_attn.o_proj.weight");

            lw.w_gate =
                LoadWeight(
                    p + ".mlp.gate_proj.weight");

            lw.w_up =
                LoadWeight(
                    p + ".mlp.up_proj.weight");

            lw.w_down =
                LoadWeight(
                    p + ".mlp.down_proj.weight");

            layer_weights_.push_back(
                std::move(lw));
        }

        const size_t expected =
            2 +
            (config_.tie_word_embeddings ? 0 : 1) +
            static_cast<size_t>(config_.num_layers) * 9;

        MLLM_INFO("LlamaRunner", "Loaded " + std::to_string(weights_.size()) + " tensors"
            + " (layers=" + std::to_string(layer_weights_.size())
            + ", expected=" + std::to_string(expected) + ")");
    }

    torch::Tensor LlamaRunner::Forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& /*attention_mask*/)
    {
        if (!is_loaded_)
        {
            throw InferenceError("LlamaRunner: model not loaded.");
        }

        const auto S =
            input_ids.size(1);

        torch::Tensor position_ids;

        const bool is_decode_step =
            (S == 1) &&
            (!kv_caches_.empty()) &&
            (kv_caches_[0].IsInitialized());

        if (is_decode_step)
        {
            int64_t cache_len =
                kv_caches_[0]
                    .key
                    .size(2);

            position_ids =
                torch::tensor(
                    { cache_len },
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(input_ids.device()));
        }
        else
        {
            position_ids =
                torch::arange(
                    0,
                    S,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(input_ids.device()));
        }

        auto hidden =
            EmbeddingLookup::Forward(
                input_ids,
                weights_.at(
                    "model.embed_tokens.weight"));

        for (int i = 0; i < config_.num_layers; ++i)
        {
            hidden =
                TransformerBlock::Forward(
                    hidden,
                    layer_weights_[i],
                    config_.num_attention_heads,
                    config_.num_key_value_heads,
                    config_.head_dim,
                    static_cast<double>(
                        config_.rope_theta),
                    static_cast<double>(
                        config_.rms_norm_eps),

                    // pure llama only
                    false,

                    position_ids,
                    &kv_caches_[i]);
        }

        hidden =
            RMSNorm::Forward(
                hidden,
                weights_.at(
                    "model.norm.weight"),
                config_.rms_norm_eps);

        const torch::Tensor& lm_head_w =
            config_.tie_word_embeddings
            ? weights_.at(
                "model.embed_tokens.weight")
            : weights_.at(
                "lm_head.weight");

        auto logits =
            Linear::Forward(
                hidden,
                lm_head_w);

        return logits;
    }

    GenerateResult LlamaRunner::Generate(
        const std::vector<int64_t>& input_ids,
        const GenerateOptions& options)
    {
        if (!is_loaded_)
        {
            throw InferenceError("LlamaRunner: model not loaded.");
        }

        GenerateResult result;
        std::vector<int64_t> current = input_ids;

        for (auto& cache : kv_caches_)
        {
            cache.Clear();
        }

        for (int step = 0;
             step < options.max_new_tokens;
             ++step)
        {
            if (config_.max_position_embeddings > 0 &&
                (int)current.size() >= config_.max_position_embeddings)
            {
                result.finish_reason = FinishReason::Length;
                break;
            }

            torch::Tensor input_tensor;

            if (step == 0)
            {
                input_tensor =
                    torch::tensor(
                        current,
                        torch::TensorOptions()
                            .dtype(torch::kInt64))
                        .unsqueeze(0);
            }
            else
            {
                input_tensor =
                    torch::tensor(
                        std::vector<int64_t>{
                            current.back()
                        },
                        torch::TensorOptions()
                            .dtype(torch::kInt64))
                        .unsqueeze(0);
            }

            auto attention_mask =
                torch::ones(
                    {1, input_tensor.size(1)},
                    torch::kInt64);

            auto logits =
                Forward(
                    input_tensor,
                    attention_mask);

            int64_t next_token =
                Sampler::Sample(
                    logits.index({
                        0,
                        logits.size(1) - 1
                    }),
                    options.temperature,
                    options.top_k,
                    options.top_p,
                    options.use_greedy,
                    current,
                    options.repetition_penalty);

            result.tokens.push_back(next_token);
            current.push_back(next_token);

            if (next_token == options.eos_token_id)
            {
                result.finish_reason = FinishReason::EOS;
                MLLM_DEBUG("LlamaRunner", "EOS detected");
                break;
            }

            bool stop_hit = false;
            for (const auto& stop_seq : options.stop_sequence_ids)
            {
                if (stop_seq.empty()) continue;
                const size_t n = stop_seq.size();
                if (current.size() >= n &&
                    std::equal(stop_seq.begin(), stop_seq.end(),
                               current.end() - static_cast<ptrdiff_t>(n)))
                {
                    result.finish_reason = FinishReason::Stop;
                    stop_hit = true;
                    break;
                }
            }
            if (stop_hit) break;
        }

        return result;
    }

    void LlamaRunner::InitKVCache(
        int batch_size,
        int max_seq_len)
    {
        // Parameters are hints for future pre-allocation; currently caches
        // grow dynamically via torch::cat in TransformerBlock.
        (void)batch_size;
        (void)max_seq_len;

        kv_caches_.clear();
        kv_caches_.resize(
            config_.num_layers);
    }

    const ModelConfig&
    LlamaRunner::GetConfig() const
    {
        return config_;
    }

    std::string
    LlamaRunner::GetModelType() const
    {
        return "llama";
    }

    bool LlamaRunner::LoadConfig(
        const std::string& config_path)
    {
        return LoadModelConfigFromJson(
            config_path,
            config_);
    }
}
