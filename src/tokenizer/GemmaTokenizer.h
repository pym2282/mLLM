// src/tokenizer/GemmaTokenizer.h
// Gemma 4 chat tokenizer (unsloth GGUF format).
// In unsloth Gemma 4 GGUF: token 106 = "<turn|>" serves as start/end of turn marker.
// Chat template: <bos><turn|>user\n{text}<turn|>\n<turn|>model\n

#pragma once

#include "tokenizer/LlamaTokenizer.h"

namespace mllm
{
    class GemmaTokenizer : public LlamaTokenizer
    {
    public:
        bool Load(const std::string& model_path) override
        {
            if (!LlamaTokenizer::Load(model_path)) return false;
            // GGUF eos_token_id already set to 106 ("<turn|>") by LoadFromGguf
            std::cerr << "[GemmaTokenizer] EOS=" << eos_token_id_ << "\n";
            return true;
        }

        std::string BuildChatPrompt(
            const std::string& system_prompt,
            const std::string& user_prompt,
            bool /*enable_thinking*/ = false) const override
        {
            std::string result = "<bos>";
            if (!system_prompt.empty())
                result += "<turn|>system\n" + system_prompt + "<turn|>\n";
            result += "<turn|>user\n" + user_prompt + "<turn|>\n<turn|>model\n";
            return result;
        }

        std::string BuildPromptFromMessages(
            const std::vector<Message>& messages,
            bool /*enable_thinking*/ = false) const override
        {
            std::string result = "<bos>";
            for (const auto& m : messages)
                result += "<turn|>" + m.role + "\n" + m.content + "<turn|>\n";
            result += "<turn|>model\n";
            return result;
        }

        std::string BuildNextUserTurn(
            const std::string& user_prompt,
            bool /*enable_thinking*/ = false) const override
        {
            return "<turn|>\n<turn|>user\n" + user_prompt + "<turn|>\n<turn|>model\n";
        }
    };
}
