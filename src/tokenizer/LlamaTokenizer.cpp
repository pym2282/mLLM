// src/tokenizer/LlamaTokenizer.cpp

#include "tokenizer/LlamaTokenizer.h"

namespace mllm
{
    // Metaspace pre-tokenizer:
    //   - replace every ' ' with ▁
    //   - prepend ▁ if the result doesn't already start with ▁
    //     (prepend_scheme: "first" — adds ▁ at the text boundary)
    std::string LlamaTokenizer::PreTokenize(
        const std::string& text
    ) const
    {
        if (text.empty()) return text;

        static const std::string kMarker = "▁";  // U+2581

        std::string result;
        result.reserve(text.size() * 2);

        for (char c : text)
        {
            if (c == ' ')
                result += kMarker;
            else
                result += c;
        }

        if (result.compare(0, kMarker.size(), kMarker) != 0)
            result = kMarker + result;

        return result;
    }
}
