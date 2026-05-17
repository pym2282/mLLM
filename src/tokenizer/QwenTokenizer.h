#pragma once

#include "tokenizer/BpeTokenizer.h"

#include <unordered_map>
#include <string>
#include <vector>

namespace mllm
{
    class QwenTokenizer : public BpeTokenizer
    {
    public:
        QwenTokenizer();

        bool Load(
            const std::string& model_path
        ) override;

        std::string BuildChatPrompt(
            const std::string& system_prompt,
            const std::string& user_prompt,
            bool enable_thinking = false
        ) const override;

        std::string BuildNextUserTurn(
            const std::string& user_prompt,
            bool enable_thinking = false
        ) const override;

        // special token handling 추가
        std::vector<int64_t> Encode(
            const std::string& text
        ) const override;

        // Qwen must override Decode for GPT2 byte-level BPE
        std::string Decode(
            const std::vector<int64_t>& token_ids
        ) const override;

        // Qwen must have correct EOS/BOS handling
        int64_t GetEOSTokenId() const override;

    protected:
        std::string PreTokenize(
            const std::string& text
        ) const override;

    private:
        std::unordered_map<
            std::string,
            int64_t
        > special_tokens_;
    };
}
