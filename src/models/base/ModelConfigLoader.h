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
            out_config.num_key_value_heads = j.value("num_key_value_heads", 0);
            out_config.vocab_size = j.value("vocab_size", 0);
            out_config.max_position_embeddings = j.value("max_position_embeddings", 0);

            std::cout << "[ConfigLoader] Config parsed successfully" << std::endl;
            std::cout << "  model_name: " << out_config.model_name << std::endl;
            std::cout << "  hidden_size: " << out_config.hidden_size << std::endl;
            std::cout << "  num_layers: " << out_config.num_layers << std::endl;
            std::cout << "  vocab_size: " << out_config.vocab_size << std::endl;

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
