// src/tokenizer/QwenTokenizer.h

#pragma once

#include "tokenizer/BpeTokenizer.h"

namespace mllm
{
    class QwenTokenizer : public BpeTokenizer
    {
    public:
        std::string BuildChatPrompt(
            const std::string& system_prompt,
            const std::string& user_prompt
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
    };
}