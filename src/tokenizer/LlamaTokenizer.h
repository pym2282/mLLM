// src/tokenizer/LlamaTokenizer.h

#pragma once

#include "tokenizer/BpeTokenizer.h"

namespace mllm
{
    // Llama / Mistral tokenizer.
    // Pre-tokenizer: Metaspace (prepend_scheme=first, split=false)
    //   — replaces spaces with ▁ and prepends ▁ at the start.
    // Everything else (BPE merge, decode, vocab) is in BpeTokenizer.
    class LlamaTokenizer : public BpeTokenizer
    {
    protected:
        std::string PreTokenize(
            const std::string& text
        ) const override;
    };
}
