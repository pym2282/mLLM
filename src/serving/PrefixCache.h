// src/serving/PrefixCache.h
//
// Hash-based KV prefix cache (nano-vllm 방식).
// Block-aligned prefix matching: 16-token blocks만 캐시.
// FNV-1a hash로 토큰 시퀀스를 식별하고 LRU eviction 사용.

#pragma once

#include "models/base/IModelRunner.h"
#include "core/Logger.h"
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace mllm
{
    class PrefixCacheManager
    {
    public:
        static constexpr int BLOCK_SIZE = 16;

        explicit PrefixCacheManager(size_t max_entries = 128)
            : max_entries_(max_entries) {}

        // 가장 긴 prefix match를 반환.
        // 반환값: 매칭된 블록 수 (BLOCK_SIZE 단위). 0 = miss.
        // out_snap: 히트 시 KV 스냅샷 채움.
        int64_t Lookup(const std::vector<int64_t>& tokens,
                       KVSnapshot& out_snap) const
        {
            // 가장 긴 prefix부터 찾기
            const int64_t max_blocks = static_cast<int64_t>(tokens.size()) / BLOCK_SIZE;
            for (int64_t nblocks = max_blocks; nblocks > 0; --nblocks)
            {
                const int64_t prefix_len = nblocks * BLOCK_SIZE;
                const uint64_t h = HashTokens(tokens.data(), prefix_len);
                auto it = cache_.find(h);
                if (it != cache_.end() && TokensMatch(it->second.snap.len, it->second.tokens, tokens))
                {
                    it->second.last_used = ++tick_;
                    out_snap = it->second.snap;
                    MLLM_INFO("PrefixCache", "HIT len=" + std::to_string(prefix_len));
                    return prefix_len;
                }
            }
            return 0;
        }

        // Generate 완료 후 prompt tokens[0..prefix_len]의 KV를 저장.
        void Store(const std::vector<int64_t>& tokens, KVSnapshot snap)
        {
            if (snap.empty()) return;

            const int64_t prefix_len = snap.len;
            if (prefix_len < BLOCK_SIZE) return;  // 너무 짧으면 저장 안 함

            // 블록 정렬
            const int64_t aligned = (prefix_len / BLOCK_SIZE) * BLOCK_SIZE;
            if (aligned == 0) return;

            // 이미 있으면 업데이트만
            const uint64_t h = HashTokens(tokens.data(), aligned);
            if (cache_.count(h)) { cache_[h].last_used = ++tick_; return; }

            // 캐시 꽉 찼으면 LRU 제거
            if (cache_.size() >= max_entries_) Evict();

            Entry e;
            e.tokens.assign(tokens.begin(), tokens.begin() + aligned);
            e.snap = std::move(snap);
            e.snap.len = aligned;
            // KV를 aligned length로 잘라내기
            for (auto& k : e.snap.keys)
                if (k.defined() && k.size(2) > aligned) k = k.slice(2, 0, aligned).contiguous();
            for (auto& v : e.snap.values)
                if (v.defined() && v.size(2) > aligned) v = v.slice(2, 0, aligned).contiguous();
            e.last_used = ++tick_;

            cache_.emplace(h, std::move(e));
            MLLM_INFO("PrefixCache", "STORE len=" + std::to_string(aligned) +
                      " total=" + std::to_string(cache_.size() + 1));
        }

        size_t Size() const { return cache_.size(); }
        void Clear() { cache_.clear(); }

    private:
        struct Entry {
            std::vector<int64_t> tokens;
            KVSnapshot           snap;
            mutable uint64_t     last_used = 0;
        };

        static uint64_t HashTokens(const int64_t* tokens, int64_t len)
        {
            // FNV-1a 64-bit
            uint64_t h = 14695981039346656037ULL;
            const uint8_t* p = reinterpret_cast<const uint8_t*>(tokens);
            for (int64_t i = 0; i < len * 8; ++i)
            {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
            return h;
        }

        static bool TokensMatch(int64_t snap_len,
                                const std::vector<int64_t>& cached,
                                const std::vector<int64_t>& query)
        {
            if ((int64_t)cached.size() != snap_len) return false;
            if ((int64_t)query.size() < snap_len)   return false;
            return std::equal(cached.begin(), cached.end(), query.begin());
        }

        void Evict()
        {
            // LRU: last_used가 가장 작은 항목 제거
            auto oldest = cache_.begin();
            for (auto it = cache_.begin(); it != cache_.end(); ++it)
                if (it->second.last_used < oldest->second.last_used)
                    oldest = it;
            cache_.erase(oldest);
        }

        std::unordered_map<uint64_t, Entry> cache_;
        size_t   max_entries_;
        mutable uint64_t tick_ = 0;
    };
}
