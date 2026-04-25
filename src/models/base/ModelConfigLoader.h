#pragma once

#include "models/base/IModelRunner.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace mllm
{
    inline bool LoadModelConfigFromJson(
        const std::string& config_path,
        ModelConfig& out_config)
    {
        try
        {
            std::ifstream file(config_path);
            if (!file.is_open())
            {
                std::cerr << "[ConfigLoader] Failed to open: "
                          << config_path << std::endl;
                return false;
            }

            nlohmann::json j;
            file >> j;

            out_config.model_name = j.value("_name_or_path", "unknown");
            out_config.hidden_size = j.value("hidden_size", 0);
            out_config.num_layers = j.value("num_hidden_layers", 0);
            out_config.num_attention_heads = j.value("num_attention_heads", 0);
            out_config.num_key_value_heads =
                j.value("num_key_value_heads", out_config.num_attention_heads);
            out_config.vocab_size = j.value("vocab_size", 0);
            out_config.max_position_embeddings = j.value("max_position_embeddings", 0);
            out_config.intermediate_size = j.value("intermediate_size", 0);
            out_config.rms_norm_eps = j.value("rms_norm_eps", 1e-5f);
            out_config.tie_word_embeddings =
                j.value("tie_word_embeddings", false);

            // head_dim: explicit if present, else hidden_size / num_attention_heads
            if (j.contains("head_dim"))
            {
                out_config.head_dim = j.value("head_dim", 0);
            }
            else if (out_config.num_attention_heads > 0)
            {
                out_config.head_dim =
                    out_config.hidden_size / out_config.num_attention_heads;
            }

            // rope_theta: HF stores it either as top-level key or inside
            // rope_parameters / rope_scaling depending on transformers version.
            if (j.contains("rope_theta"))
            {
                out_config.rope_theta = j.value("rope_theta", 10000.0f);
            }
            else if (j.contains("rope_parameters") &&
                     j["rope_parameters"].is_object())
            {
                out_config.rope_theta =
                    j["rope_parameters"].value("rope_theta", 10000.0f);
            }

            std::cout << "[ConfigLoader] Config parsed successfully" << std::endl;
            std::cout << "  model_name: " << out_config.model_name << std::endl;
            std::cout << "  hidden_size: " << out_config.hidden_size << std::endl;
            std::cout << "  num_layers: " << out_config.num_layers << std::endl;
            std::cout << "  num_attention_heads: " << out_config.num_attention_heads << std::endl;
            std::cout << "  num_key_value_heads: " << out_config.num_key_value_heads << std::endl;
            std::cout << "  head_dim: " << out_config.head_dim << std::endl;
            std::cout << "  intermediate_size: " << out_config.intermediate_size << std::endl;
            std::cout << "  vocab_size: " << out_config.vocab_size << std::endl;
            std::cout << "  rms_norm_eps: " << out_config.rms_norm_eps << std::endl;
            std::cout << "  rope_theta: " << out_config.rope_theta << std::endl;
            std::cout << "  tie_word_embeddings: "
                      << (out_config.tie_word_embeddings ? "true" : "false")
                      << std::endl;

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[ConfigLoader] JSON parse failed:\n"
                      << e.what() << std::endl;
            return false;
        }
    }
}
