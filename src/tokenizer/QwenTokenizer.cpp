// src/tokenizer/QwenTokenizer.cpp

#include "tokenizer/QwenTokenizer.h"

#include <cstdint>
#include <iostream>

namespace
{
    static bool IsGpt2VisibleByte(uint8_t b)
    {
        return (b >= static_cast<uint8_t>('!') && b <= static_cast<uint8_t>('~')) ||
               (b >= 0xA1 && b <= 0xAC) ||
               (b >= 0xAE && b <= 0xFF);
    }

    static int ByteToUnicode(uint8_t b)
    {
        if (IsGpt2VisibleByte(b))
        {
            return b;
        }

        int n = 0;
        for (int i = 0; i < static_cast<int>(b); ++i)
        {
            if (!IsGpt2VisibleByte(static_cast<uint8_t>(i)))
            {
                ++n;
            }
        }

        return 256 + n;
    }

    static bool UnicodeToByte(int codepoint, uint8_t& out)
    {
        if (codepoint >= 0 && codepoint <= 0xFF &&
            IsGpt2VisibleByte(static_cast<uint8_t>(codepoint)))
        {
            out = static_cast<uint8_t>(codepoint);
            return true;
        }

        int n = 0;
        for (int i = 0; i <= 0xFF; ++i)
        {
            if (IsGpt2VisibleByte(static_cast<uint8_t>(i)))
            {
                continue;
            }

            if (codepoint == 256 + n)
            {
                out = static_cast<uint8_t>(i);
                return true;
            }

            ++n;
        }

        return false;
    }

    static bool ReadUtf8Codepoint(
        const std::string& text,
        size_t& pos,
        int& codepoint)
    {
        const auto b0 =
            static_cast<uint8_t>(text[pos]);

        if (b0 < 0x80)
        {
            codepoint = b0;
            ++pos;
            return true;
        }

        int len = 0;
        int value = 0;

        if ((b0 & 0xE0) == 0xC0)
        {
            len = 2;
            value = b0 & 0x1F;
        }
        else if ((b0 & 0xF0) == 0xE0)
        {
            len = 3;
            value = b0 & 0x0F;
        }
        else if ((b0 & 0xF8) == 0xF0)
        {
            len = 4;
            value = b0 & 0x07;
        }
        else
        {
            return false;
        }

        if (pos + static_cast<size_t>(len) > text.size())
        {
            return false;
        }

        for (int i = 1; i < len; ++i)
        {
            const auto bx =
                static_cast<uint8_t>(text[pos + i]);

            if ((bx & 0xC0) != 0x80)
            {
                return false;
            }

            value = (value << 6) | (bx & 0x3F);
        }

        codepoint = value;
        pos += static_cast<size_t>(len);
        return true;
    }

    static void AppendUtf8(std::string& out, int codepoint)
    {
        if (codepoint <= 0x7F)
        {
            out.push_back(static_cast<char>(codepoint));
        }
        else if (codepoint <= 0x7FF)
        {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }
}

namespace mllm
{
    QwenTokenizer::QwenTokenizer()
    {
    }

    bool QwenTokenizer::Load(const std::string& model_path)
    {
        if (!BpeTokenizer::Load(model_path))
        {
            return false;
        }

        special_tokens_.clear();

        for (const auto& [id, token] : id_to_token_)
        {
            if (token.size() >= 4 &&
                token.rfind("<|", 0) == 0 &&
                token.find("|>") == token.size() - 2)
            {
                special_tokens_[token] = id;
            }
        }

        if (!special_tokens_.count("<|im_start|>") ||
            !special_tokens_.count("<|im_end|>"))
        {
            std::cerr
                << "[QwenTokenizer] Missing required chat special tokens."
                << std::endl;
            return false;
        }

        return true;
    }

    std::vector<int64_t> QwenTokenizer::Encode(
        const std::string& text
    ) const
    {
        std::vector<int64_t> result;

        size_t cursor = 0;

        while (cursor < text.size())
        {
            size_t next_special_pos = std::string::npos;
            std::string matched_special;

            // -------------------------------------------------
            // find nearest next special token
            // -------------------------------------------------

            for (const auto& kv : special_tokens_)
            {
                const std::string& token_text =
                    kv.first;

                size_t pos =
                    text.find(
                        token_text,
                        cursor
                    );

                if (pos == std::string::npos)
                {
                    continue;
                }

                if (
                    next_special_pos == std::string::npos
                    ||
                    pos < next_special_pos
                )
                {
                    next_special_pos = pos;
                    matched_special = token_text;
                }
            }

            // -------------------------------------------------
            // no more special token:
            // encode remaining full chunk
            // -------------------------------------------------

            if (next_special_pos == std::string::npos)
            {
                std::string chunk =
                    text.substr(cursor);

                auto partial =
                    BpeTokenizer::Encode(
                        chunk
                    );

                result.insert(
                    result.end(),
                    partial.begin(),
                    partial.end()
                );

                break;
            }

            // -------------------------------------------------
            // encode normal text before special token
            // -------------------------------------------------

            if (next_special_pos > cursor)
            {
                std::string chunk =
                    text.substr(
                        cursor,
                        next_special_pos - cursor
                    );

                auto partial =
                    BpeTokenizer::Encode(
                        chunk
                    );

                result.insert(
                    result.end(),
                    partial.begin(),
                    partial.end()
                );
            }

            // -------------------------------------------------
            // push special token as single token
            // -------------------------------------------------

            result.push_back(
                special_tokens_.at(
                    matched_special
                )
            );

            cursor =
                next_special_pos +
                matched_special.size();
        }

        return result;
    }

    std::string QwenTokenizer::PreTokenize(
        const std::string& text
    ) const
    {
        std::string result;
        result.reserve(text.size() * 2);

        for (unsigned char c : text)
        {
            AppendUtf8(
                result,
                ByteToUnicode(static_cast<uint8_t>(c))
            );
        }

        return result;
    }

    std::string QwenTokenizer::BuildChatPrompt(
        const std::string& system_prompt,
        const std::string& user_prompt
    ) const
    {
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

    std::string QwenTokenizer::Decode(
        const std::vector<int64_t>& token_ids
    ) const
    {
        std::string text =
            BpeTokenizer::Decode(
                token_ids
            );

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
                    (start_pos =
                        str.find(
                            from,
                            start_pos
                        ))
                    != std::string::npos
                )
                {
                    str.replace(
                        start_pos,
                        from.length(),
                        to
                    );

                    start_pos +=
                        to.length();
                }
            };

        // special tokens cleanup
        ReplaceAll(text, "<|im_end|>", "");
        ReplaceAll(text, "<|im_start|>", "");

        std::string decoded;
        decoded.reserve(text.size());

        size_t pos = 0;
        while (pos < text.size())
        {
            const size_t original_pos = pos;
            int codepoint = 0;

            if (!ReadUtf8Codepoint(text, pos, codepoint))
            {
                decoded.push_back(text[original_pos]);
                pos = original_pos + 1;
                continue;
            }

            uint8_t byte = 0;
            if (UnicodeToByte(codepoint, byte))
            {
                decoded.push_back(static_cast<char>(byte));
            }
            else
            {
                AppendUtf8(decoded, codepoint);
            }
        }

        return decoded;
    }

    int64_t QwenTokenizer::GetEOSTokenId() const
    {
        const std::vector<std::string> candidates =
        {
            "<|im_end|>",
            "<|endoftext|>",
            "</s>"
        };

        for (const auto& token : candidates)
        {
            auto it =
                token_to_id_.find(
                    token
                );

            if (it != token_to_id_.end())
            {
                return it->second;
            }
        }

        std::cerr
            << "[QwenTokenizer] EOS token not found. fallback = -1"
            << std::endl;

        return -1;
    }
}
