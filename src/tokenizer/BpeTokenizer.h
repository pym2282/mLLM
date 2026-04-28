// src/tokenizer/BpeTokenizer.h

#pragma once

#include "tokenizer/ITokenizer.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>

namespace mllm
{
    // Generic BPE tokenizer base.
    // Subclasses implement PreTokenize() for model-specific text normalization
    // (e.g. Metaspace for Llama/Mistral, ByteLevel for Qwen).
    // Everything else — vocab/merge loading, BPE algorithm, decode — is shared.
    class BpeTokenizer : public ITokenizer
    {
    public:
        bool Load(const std::string& model_path) override;

        std::vector<int64_t> Encode(
            const std::string& text
        ) const override;

        std::string Decode(
            const std::vector<int64_t>& tokens
        ) const override;

        int64_t GetEOSTokenId() const override;

    protected:
        // Convert raw text to the pre-tokenized form expected by BPE.
        // Called by Encode() before splitting into initial symbols.
        virtual std::string PreTokenize(
            const std::string& text
        ) const = 0;

        std::unordered_map<std::string, int64_t> token_to_id_;
        std::unordered_map<int64_t, std::string> id_to_token_;

        int64_t bos_token_id_ = -1;
        int64_t eos_token_id_ = -1;
        bool byte_fallback_ = false;

    private:
        std::unordered_map<std::string, int> merge_rank_;

        std::vector<std::string> InitialSymbols(
            const std::string& preprocessed
        ) const;

        std::vector<std::string> ApplyMerges(
            std::vector<std::string> symbols
        ) const;
    };
}
