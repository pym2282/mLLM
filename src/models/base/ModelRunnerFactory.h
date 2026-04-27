#pragma once

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

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
        // Falls back to Llama on any parse error.
        static ModelBundle Create(const std::string& model_path)
        {
            const std::string model_type =
                ReadModelType(model_path + "/config.json");

            std::cout
                << "[ModelRunnerFactory] model_type=" << model_type
                << " path=" << model_path
                << std::endl;

            ModelBundle bundle;

            if (model_type == "qwen" || model_type == "qwen3" || model_type == "qwen2")
            {
                bundle.runner    = std::make_unique<QwenRunner>();
                bundle.tokenizer = std::make_unique<QwenTokenizer>();
            }
            else
            {
                bundle.runner    = std::make_unique<LlamaRunner>();
                bundle.tokenizer = std::make_unique<LlamaTokenizer>();
            }

            return bundle;
        }

    private:
        static std::string ReadModelType(const std::string& config_path)
        {
            try
            {
                std::ifstream f(config_path);
                if (!f.is_open())
                    return "llama";

                nlohmann::json j;
                f >> j;

                return j.value("model_type", "llama");
            }
            catch (...)
            {
                return "llama";
            }
        }
    };
}
