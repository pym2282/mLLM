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
        ) const = 0;

        virtual std::string Decode(
            const std::vector<int64_t>& tokens
        ) const = 0;

        virtual int64_t GetEOSTokenId() const = 0;

        virtual bool SupportsThinking() const { return false; }

        virtual std::string BuildChatPrompt(
                const std::string& system_prompt,
                const std::string& user_prompt,
                bool enable_thinking = false
        ) const
        {
            return system_prompt + "\n\nUser: " +
                   user_prompt +
                   "\nAssistant: ";
        }

        // Returns the text to append for a follow-up user turn after an
        // assistant response has already been appended to the token history.
        virtual std::string BuildNextUserTurn(
                const std::string& user_prompt,
                bool enable_thinking = false
        ) const
        {
            return "\n\nUser: " + user_prompt + "\nAssistant: ";
        }
    };
}