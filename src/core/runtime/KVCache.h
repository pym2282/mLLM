// src/core/runtime/KVCache.h

#pragma once

#include <torch/torch.h>

namespace mllm
{
    struct KVCache
    {
        // Shape:
        // key   = [batch, num_kv_heads, seq_len, head_dim]
        // value = [batch, num_kv_heads, seq_len, head_dim]

        torch::Tensor key;
        torch::Tensor value;

        bool IsInitialized() const
        {
            return key.defined() && value.defined();
        }

        void Clear()
        {
            key = torch::Tensor();
            value = torch::Tensor();
        }
    };
}