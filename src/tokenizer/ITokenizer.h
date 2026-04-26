#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace mllm
{
    class ITokenizer
    {
    public:
        virtual ~ITokenizer() = default;

        virtual bool Load(
            const std::string& model_path
        ) = 0;

        virtual std::vector<int64_t> Encode(
            const std::string& text
        ) = 0;

        virtual std::string Decode(
            const std::vector<int64_t>& tokens
        ) = 0;

        virtual int64_t GetEOSTokenId() const = 0;
    };
}