// src/models/qwen/QwenRunner.cpp

#include "models/qwen/QwenRunner.h"

#include <iostream>
#include <stdexcept>

#include "models/base/ModelConfigLoader.h"
#include "models/base/SafeTensorLoader.h"
#include "models/base/SafeTensorTensorLoader.h"

#include "core/runtime/EmbeddingLookup.h"
#include "core/runtime/Sampler.h"
#include "core/runtime/TransformerBlock.h"
#include "core/runtime/RMSNorm.h"
#include "debug/TensorCompare.h"

namespace mllm
{
    bool QwenRunner::Load(const std::string& model_path)
    {
        try
        {
            model_path_ = model_path;
            if (!LoadConfig(model_path + "/config.json"))
            {
                std::cerr << "[QwenRunner] Failed to load config" << std::endl;
                return false;
            }

            if (!SafeTensorLoader::Exists(model_path))
            {
                std::cerr << "[QwenRunner] model.safetensors not found" << std::endl;
                return false;
            }

            if (!SafeTensorHeaderParser::Parse(model_path, tensor_map_))
            {
                std::cerr << "[QwenRunner] Failed to parse safetensors header" << std::endl;
                return false;
            }

            LoadAllWeights();

            kv_caches_.clear();
            kv_caches_.resize(config_.num_layers);

            is_loaded_ = true;

            std::cout
                << "[QwenRunner] Model loaded"
                << " path=" << model_path_
                << " layers=" << config_.num_layers
                << " weights=" << weights_.size()
                << std::endl;

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "[QwenRunner] Load failed:\n"
                << e.what()
                << std::endl;

            is_loaded_ = false;
            return false;
        }
    }

    torch::Tensor& QwenRunner::LoadWeight(const std::string& name)
    {
        auto it = weights_.find(name);

        if (it != weights_.end())
        {
            return it->second;
        }

        auto tensor = SafeTensorTensorLoader::LoadTensor(
            model_path_,
            name,
            tensor_map_
        );

        auto [ins, ok] = weights_.emplace(
            name,
            std::move(tensor)
        );

        (void)ok;
        return ins->second;
    }

    void QwenRunner::LoadAllWeights()
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
                "model.layers." + std::to_string(i);

            LayerWeights lw;

            lw.input_layernorm =
                LoadWeight(p + ".input_layernorm.weight");

            lw.post_attention_layernorm =
                LoadWeight(p + ".post_attention_layernorm.weight");

            lw.w_q =
                LoadWeight(p + ".self_attn.q_proj.weight");

            lw.w_k =
                LoadWeight(p + ".self_attn.k_proj.weight");

            lw.w_v =
                LoadWeight(p + ".self_attn.v_proj.weight");

            lw.w_o =
                LoadWeight(p + ".self_attn.o_proj.weight");

            lw.w_q_norm =
                LoadWeight(p + ".self_attn.q_norm.weight");

            lw.w_k_norm =
                LoadWeight(p + ".self_attn.k_norm.weight");

            lw.w_gate =
                LoadWeight(p + ".mlp.gate_proj.weight");

            lw.w_up =
                LoadWeight(p + ".mlp.up_proj.weight");

            lw.w_down =
                LoadWeight(p + ".mlp.down_proj.weight");

            layer_weights_.push_back(std::move(lw));
        }

        const size_t expected =
            2 +
            (config_.tie_word_embeddings ? 0 : 1) +
            static_cast<size_t>(config_.num_layers) * 11;

        std::cout
            << "[QwenRunner] Loaded "
            << weights_.size()
            << " tensors (expected="
            << expected
            << ")"
            << std::endl;
    }

    torch::Tensor QwenRunner::Forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& /*attention_mask*/)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error(
                "QwenRunner: model not loaded."
            );
        }

        const auto S = input_ids.size(1);

        torch::Tensor position_ids;


        const bool is_decode =
            (S == 1) &&
            !kv_caches_.empty() &&
            kv_caches_[0].IsInitialized();

        if (is_decode)
        {
            position_ids = torch::tensor(
                { kv_caches_[0].key.size(2) },
                torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(input_ids.device())
            );
        }
        else
        {
            position_ids = torch::arange(
                0,
                S,
                torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(input_ids.device())
            );
        }

        auto hidden = EmbeddingLookup::Forward(
            input_ids,
            weights_.at("model.embed_tokens.weight")
        );

        // -----------------------------
        // embedding parity compare
        // -----------------------------
        if (parity_mode_ && !is_decode)
        {
            TensorCompare::CompareTensor(
                hidden,
                parity_reference_dir_ + "/embedding.txt"
            );
        }

        for (int i = 0; i < config_.num_layers; ++i)
        {
            hidden = TransformerBlock::Forward(
                hidden,
                layer_weights_[i],
                config_.num_attention_heads,
                config_.num_key_value_heads,
                config_.head_dim,
                static_cast<double>(config_.rope_theta),
                static_cast<double>(config_.rms_norm_eps),
                true,
                position_ids,
                &kv_caches_[i]
            );
        }

        if (parity_mode_ && !is_decode) {
            TensorCompare::CompareTensor(
                hidden,
                parity_reference_dir_ + "/last_layer_output.txt"
            );
        }

        hidden = RMSNorm::Forward(
            hidden,
            weights_.at("model.norm.weight"),
            config_.rms_norm_eps
        );

        if (parity_mode_ && !is_decode)
        {
            TensorCompare::CompareTensor(
                hidden,
                parity_reference_dir_ + "/final_norm_output.txt"
            );
        }

        const torch::Tensor& lm_head_w =
            (weights_.count("lm_head.weight") > 0)
                ? weights_.at("lm_head.weight")
                : weights_.at("model.embed_tokens.weight");

        auto logits = Linear::Forward(hidden, lm_head_w);

        if (parity_mode_ && !is_decode)
        {
            auto last_logits =
                logits.select(1, logits.size(1) - 1);

            TensorCompare::CompareTensor(
                last_logits,
                parity_reference_dir_ + "/final_logits.txt"
            );
        }

        return logits;
    }

    std::vector<int64_t> QwenRunner::GenerateInternal(
        const std::vector<int64_t>& input_ids,
        const GenerateOptions& options,
        bool streaming,
        ITokenizer* tokenizer)
    {
        if (!is_loaded_)
        {
            throw std::runtime_error(
                "QwenRunner: model not loaded."
            );
        }

        if (streaming && tokenizer == nullptr)
        {
            throw std::runtime_error(
                "QwenRunner: tokenizer is null."
            );
        }

        std::vector<int64_t> generated;
        std::vector<int64_t> current = input_ids;

        for (auto& cache : kv_caches_)
        {
            cache.Clear();
        }

        for (int step = 0; step < options.max_new_tokens; ++step)
        {
            torch::Tensor input_tensor;

            if (step == 0)
            {
                input_tensor = torch::tensor(
                    current,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                ).unsqueeze(0);
            }
            else
            {
                input_tensor = torch::tensor(
                    std::vector<int64_t>{ current.back() },
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                ).unsqueeze(0);
            }

            auto attention_mask = torch::ones(
                { 1, input_tensor.size(1) },
                torch::kInt64
            );

            auto logits = Forward(
                input_tensor,
                attention_mask
            );

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
                    options.repetition_penalty
                );

            generated.push_back(next_token);
            current.push_back(next_token);

            if (streaming)
            {
                std::cout
                    << tokenizer->Decode(
                        std::vector<int64_t>{ next_token }
                    )
                    << std::flush;
            }

            if (next_token == options.eos_token_id)
            {
                if (streaming)
                {
                    std::cout << std::endl;
                }

                std::cout
                    << "[QwenRunner] EOS detected."
                    << std::endl;

                break;
            }
        }

        return generated;
    }

    std::vector<int64_t> QwenRunner::Generate(
        const std::vector<int64_t>& input_ids,
        const GenerateOptions& options)
    {
        return GenerateInternal(
            input_ids,
            options,
            false,
            nullptr
        );
    }

    void QwenRunner::InitKVCache(
        int batch_size,
        int max_seq_len)
    {
        (void)batch_size;
        (void)max_seq_len;

        kv_caches_.clear();
        kv_caches_.resize(config_.num_layers);
    }

    const ModelConfig& QwenRunner::GetConfig() const
    {
        return config_;
    }

    std::string QwenRunner::GetModelType() const
    {
        return "qwen";
    }

    void QwenRunner::SetParityMode(bool enabled)
    {
        parity_mode_ = enabled;
    }

    void QwenRunner::SetParityReferenceDir(const std::string& path)
    {
        parity_reference_dir_ = path;
    }

    bool QwenRunner::LoadConfig(
        const std::string& config_path)
    {
        return LoadModelConfigFromJson(
            config_path,
            config_
        );
    }
}
