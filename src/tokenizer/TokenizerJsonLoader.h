#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace mllm
{
    class TokenizerJsonLoader
    {
    public:
        static nlohmann::json Load(
            const std::string& model_path
        );
    };
}