// src/tokenizer/BpeTokenizer.cpp

#include "tokenizer/BpeTokenizer.h"
#include "tokenizer/TokenizerJsonLoader.h"

#include <algorithm>
#include <iostream>
#include <climits>
#include <cstdio>

namespace
{
    // Returns byte length of a UTF-8 character from its first byte.
    static int Utf8CharLen(uint8_t b)
    {
        if (b < 0x80) return 1;
        if (b < 0xE0) return 2;
        if (b < 0xF0) return 3;
        return 4;
    }

    // Checks if tok is a byte-fallback token like <0x41>.
    // Format is always exactly 6 chars: <0xXX>
    static bool IsByteToken(const std::string& tok)
    {
        return tok.size() == 6
            && tok[0] == '<'
            && tok[1] == '0'
            && tok[2] == 'x'
            && tok[5] == '>';
    }

    static uint8_t ParseByteToken(const std::string& tok)
    {
        unsigned int v = 0;
        std::sscanf(tok.c_str() + 3, "%02X", &v);
        return static_cast<uint8_t>(v);
    }
}

namespace mllm
{
    // -------------------------------------------------------
    // Load
    // -------------------------------------------------------

    bool BpeTokenizer::Load(const std::string& model_path)
    {
        try
        {
            token_to_id_.clear();
            id_to_token_.clear();
            merge_rank_.clear();
            bos_token_id_ = -1;
            eos_token_id_ = -1;
            byte_fallback_ = false;

            auto j = TokenizerJsonLoader::Load(model_path);

            // vocab
            const auto& vocab = j["model"]["vocab"];
            token_to_id_.reserve(vocab.size());
            id_to_token_.reserve(vocab.size());

            for (auto it = vocab.begin(); it != vocab.end(); ++it)
            {
                const std::string token = it.key();
                const int64_t id = it.value().get<int64_t>();
                token_to_id_[token] = id;
                id_to_token_[id] = token;
            }

            std::cerr
                << "[BpeTokenizer] vocab: "
                << token_to_id_.size() << " tokens"
                << std::endl;

            // merges: [[first, second], ...]  rank = index
            const auto& merges = j["model"]["merges"];
            merge_rank_.reserve(merges.size());

            for (int rank = 0;
                 rank < static_cast<int>(merges.size());
                 ++rank)
            {
                std::string first  = merges[rank][0].get<std::string>();
                std::string second = merges[rank][1].get<std::string>();

                // null byte as separator — safe because token strings
                // never contain raw null bytes
                merge_rank_[first + '\0' + second] = rank;
            }

            std::cerr
                << "[BpeTokenizer] merges: "
                << merge_rank_.size()
                << std::endl;

            // flags
            byte_fallback_ = j["model"].value("byte_fallback", false);

            // special tokens
            for (const auto& item : j["added_tokens"])
            {
                const std::string content =
                    item["content"].get<std::string>();
                const int64_t id = item["id"].get<int64_t>();

                token_to_id_[content] = id;
                id_to_token_[id] = content;

                if      (content == "<s>")  bos_token_id_ = id;
                else if (content == "</s>") eos_token_id_ = id;
            }

            std::cerr
                << "[BpeTokenizer] BOS=" << bos_token_id_
                << " EOS=" << eos_token_id_
                << " byte_fallback=" << byte_fallback_
                << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "[BpeTokenizer] Load failed: "
                << e.what()
                << std::endl;
            return false;
        }

        return true;
    }

    // -------------------------------------------------------
    // InitialSymbols
    //
    // Split pre-tokenized text into BPE initial symbols.
    // Each Unicode code point:
    //   - in vocab → one symbol
    //   - not in vocab + byte_fallback → one <0xXX> symbol per byte
    //   - not in vocab + no fallback  → kept as-is (will become UNK)
    // -------------------------------------------------------

    std::vector<std::string> BpeTokenizer::InitialSymbols(
        const std::string& text
    ) const
    {
        std::vector<std::string> syms;
        syms.reserve(text.size());

        size_t i = 0;
        while (i < text.size())
        {
            const int len = Utf8CharLen(
                static_cast<uint8_t>(text[i])
            );
            const int safe_len = static_cast<int>(
                std::min(
                    static_cast<size_t>(len),
                    text.size() - i
                )
            );

            std::string ch = text.substr(i, safe_len);

            if (token_to_id_.count(ch))
            {
                syms.push_back(std::move(ch));
            }
            else if (byte_fallback_)
            {
                for (int j = 0; j < safe_len; ++j)
                {
                    char buf[8];
                    std::snprintf(
                        buf, sizeof(buf),
                        "<0x%02X>",
                        static_cast<uint8_t>(text[i + j])
                    );
                    syms.emplace_back(buf);
                }
            }
            else
            {
                syms.push_back(std::move(ch));
            }

            i += safe_len;
        }

        return syms;
    }

    // -------------------------------------------------------
    // ApplyMerges
    //
    // Standard BPE: each iteration finds the pair with the
    // lowest merge rank and merges ALL occurrences of it,
    // then repeats until no more merges apply.
    // -------------------------------------------------------

    std::vector<std::string> BpeTokenizer::ApplyMerges(
        std::vector<std::string> syms
    ) const
    {
        while (syms.size() >= 2)
        {
            // Find the best (lowest-rank) adjacent pair
            int best_rank = INT_MAX;
            std::string best_first, best_second;

            for (size_t i = 0; i + 1 < syms.size(); ++i)
            {
                const std::string key =
                    syms[i] + '\0' + syms[i + 1];

                auto it = merge_rank_.find(key);
                if (it != merge_rank_.end() &&
                    it->second < best_rank)
                {
                    best_rank = it->second;
                    best_first  = syms[i];
                    best_second = syms[i + 1];
                }
            }

            if (best_rank == INT_MAX) break;

            // Merge ALL occurrences of the best pair
            std::vector<std::string> next;
            next.reserve(syms.size());

            size_t i = 0;
            while (i < syms.size())
            {
                if (i + 1 < syms.size()
                    && syms[i]     == best_first
                    && syms[i + 1] == best_second)
                {
                    next.push_back(best_first + best_second);
                    i += 2;
                }
                else
                {
                    next.push_back(syms[i]);
                    ++i;
                }
            }

            syms = std::move(next);
        }

        return syms;
    }

    // -------------------------------------------------------
    // Encode
    // -------------------------------------------------------

    std::vector<int64_t> BpeTokenizer::Encode(
        const std::string& text
    ) const
    {
        std::vector<int64_t> result;

        if (bos_token_id_ >= 0)
            result.push_back(bos_token_id_);

        if (text.empty()) return result;

        // 1. Model-specific pre-tokenization (Metaspace, ByteLevel, …)
        const std::string preprocessed = PreTokenize(text);

        // 2. Character-level split with byte fallback
        auto syms = InitialSymbols(preprocessed);

        // 3. BPE merge
        syms = ApplyMerges(std::move(syms));

        // 4. Symbols → IDs
        for (const auto& sym : syms)
        {
            auto it = token_to_id_.find(sym);
            if (it != token_to_id_.end())
            {
                result.push_back(it->second);
            }
            // byte_fallback ensures no unknown symbols for supported models;
            // unrecognised symbols are silently dropped rather than inserting
            // a wrong ID
        }

        return result;
    }

    // -------------------------------------------------------
    // Decode
    //
    // Pipeline mirroring tokenizer.json decoder:
    //   1. Concatenate token strings, replacing ▁ with space
    //   2. Byte-fallback tokens (<0xXX>) become raw bytes
    //   3. Strip one leading space (artifact of prepend_scheme=first)
    // -------------------------------------------------------

    std::string BpeTokenizer::Decode(
        const std::vector<int64_t>& tokens
    ) const
    {
        static const std::string kBos = "<s>";
        static const std::string kEos = "</s>";
        static const std::string kMarker = "▁";  // U+2581, 3 bytes

        std::vector<uint8_t> bytes;
        bytes.reserve(tokens.size() * 4);

        for (auto id : tokens)
        {
            auto it = id_to_token_.find(id);
            if (it == id_to_token_.end()) continue;

            const std::string& tok = it->second;
            if (tok == kBos || tok == kEos) continue;

            if (IsByteToken(tok))
            {
                bytes.push_back(ParseByteToken(tok));
            }
            else
            {
                // Replace ▁ → space; copy remaining bytes as-is
                size_t pos = 0;
                while (pos < tok.size())
                {
                    if (tok.compare(pos, kMarker.size(), kMarker) == 0)
                    {
                        bytes.push_back(' ');
                        pos += kMarker.size();
                    }
                    else
                    {
                        bytes.push_back(
                            static_cast<uint8_t>(tok[pos++])
                        );
                    }
                }
            }
        }

        std::string result(bytes.begin(), bytes.end());

        // Strip one leading space produced by the ▁ prepend
        if (!result.empty() && result[0] == ' ')
            result.erase(0, 1);

        return result;
    }

    // -------------------------------------------------------
    // GetEOSTokenId
    // -------------------------------------------------------

    int64_t BpeTokenizer::GetEOSTokenId() const
    {
        return eos_token_id_;
    }
}
