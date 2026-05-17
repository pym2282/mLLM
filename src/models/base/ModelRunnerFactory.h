#pragma once

#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "models/base/IModelRunner.h"
#include "tokenizer/ITokenizer.h"

#include "models/llama/LlamaRunner.h"
#include "models/qwen/QwenRunner.h"
#include "tokenizer/LlamaTokenizer.h"
#include "tokenizer/QwenTokenizer.h"

namespace mllm
{
    struct ModelBundle
    {
        std::unique_ptr<IModelRunner> runner;
        std::unique_ptr<ITokenizer>   tokenizer;
    };

    class ModelRunnerFactory
    {
    public:
        // Read model_type from config.json and create the matching runner + tokenizer.
        // Unknown or unreadable model types fail explicitly to avoid loading the
        // wrong architecture and producing misleading tensor errors later.
        static ModelBundle Create(const std::string& model_path)
        {
            const std::string model_type =
                ReadModelType(model_path + "/config.json");

            std::cerr
                << "[ModelRunnerFactory] model_type=" << model_type
                << " path=" << model_path
                << std::endl;

            ModelBundle bundle;

            if (model_type == "qwen" || model_type == "qwen3" || model_type == "qwen2")
            {
                bundle.runner    = std::make_unique<QwenRunner>();
                bundle.tokenizer = std::make_unique<QwenTokenizer>();
            }
            else if (model_type == "llama")
            {
                bundle.runner    = std::make_unique<LlamaRunner>();
                bundle.tokenizer = std::make_unique<LlamaTokenizer>();
            }
            else
            {
                throw std::runtime_error(
                    "Unsupported model_type '" + model_type +
                    "' in " + model_path + "/config.json");
            }

            return bundle;
        }

    private:
        static std::string ReadModelType(const std::string& config_path)
        {
            std::ifstream f(config_path);
            if (!f.is_open())
            {
                throw std::runtime_error(
                    "Failed to open model config: " + config_path);
            }

            nlohmann::json j;
            f >> j;

            if (!j.contains("model_type"))
            {
                throw std::runtime_error(
                    "Missing model_type in config: " + config_path);
            }

            return j.value("model_type", "");
        }
    };
}
