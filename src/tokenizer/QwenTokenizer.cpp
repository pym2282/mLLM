// src/tokenizer/QwenTokenizer.cpp

#include "tokenizer/QwenTokenizer.h"

namespace mllm
{
    std::string QwenTokenizer::PreTokenize(
        const std::string& text
    ) const
    {
        if (text.empty())
        {
            return text;
        }

        std::string result;
        result.reserve(
            text.size() * 2
        );

        for (char c : text)
        {
            if (c == ' ')
            {
                result += "Ġ";
            }
            else if (c == '\n')
            {
                result += "Ċ";
            }
            else
            {
                result += c;
            }
        }

        return result;
    }

    std::string QwenTokenizer::BuildChatPrompt(
        const std::string& system_prompt,
        const std::string& user_prompt
    ) const
    {
        // Qwen chat template
        //
        // 실제 HF tokenizer.apply_chat_template()의
        // 최소 동작 형태로 맞춤
        //
        // 이후 special token exact parity는
        // tokenizer_config 기반으로 확장 가능

        std::string prompt;

        prompt += "<|im_start|>system\n";
        prompt += system_prompt;
        prompt += "\n<|im_end|>\n";

        prompt += "<|im_start|>user\n";
        prompt += user_prompt;
        prompt += "\n<|im_end|>\n";

        prompt += "<|im_start|>assistant\n\n";

        return prompt;
    }

    // src/tokenizer/QwenTokenizer.cpp
    // 아래 함수들 추가

    std::string QwenTokenizer::Decode(
        const std::vector<int64_t>& token_ids
    ) const
    {
        // 먼저 base decode 사용
        std::string text =
            BpeTokenizer::Decode(token_ids);

        // -------------------------------------------------
        // GPT2-style byte-level BPE cleanup
        //
        // Ġ -> space
        // Ċ -> newline
        //
        // 최소 parity용
        // 이후 bytes_to_unicode reverse map까지
        // 확장 가능
        // -------------------------------------------------

        auto ReplaceAll =
            [](
                std::string& str,
                const std::string& from,
                const std::string& to
            )
            {
                if (from.empty())
                {
                    return;
                }

                size_t start_pos = 0;

                while (
                    (start_pos = str.find(from, start_pos))
                    != std::string::npos
                )
                {
                    str.replace(
                        start_pos,
                        from.length(),
                        to
                    );

                    start_pos += to.length();
                }
            };

        ReplaceAll(text, "Ġ", " ");
        ReplaceAll(text, "Ċ", "\n");

        ReplaceAll(text, "<|im_end|>", "");
        ReplaceAll(text, "<|im_start|>", "");

        ReplaceAll(text, "system", "");
        ReplaceAll(text, "user", "");
        ReplaceAll(text, "assistant", "");

        return text;
    }

    int64_t QwenTokenizer::GetEOSTokenId() const
    {
        // Qwen commonly uses <|im_end|>
        // exact id depends on tokenizer.json,
        // but for now we try known fallback ids

        const std::vector<std::string> candidates =
        {
            "<|im_end|>",
            "<|endoftext|>",
            "</s>"
        };

        for (const auto& token : candidates)
        {
            auto it = token_to_id_.find(token);
            if (it != token_to_id_.end())
            {
                return it->second;
            }
        }

        std::cout
            << "[QwenTokenizer] EOS token not found. fallback = -1"
            << std::endl;

        return -1;
    }
}