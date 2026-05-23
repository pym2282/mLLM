// src/core/runtime/KVCache.h

#pragma once

#include <torch/torch.h>

namespace mllm
{
    struct KVCache
    {
        // Shape: [batch, num_kv_heads, seq_len, head_dim]
        torch::Tensor key;
        torch::Tensor value;

        int64_t len      = 0;  // tokens written so far
        int64_t capacity = 0;  // 0 = dynamic (legacy), >0 = pre-allocated

        bool IsInitialized() const
        {
            return key.defined() && value.defined();
        }

        void Allocate(
            int64_t batch,
            int64_t num_kv_heads,
            int64_t max_seq,
            int64_t head_dim,
            torch::Device device,
            torch::Dtype dtype)
        {
            auto opts = torch::TensorOptions()
                .dtype(dtype)
                .device(device);

            key   = torch::empty({batch, num_kv_heads, max_seq, head_dim}, opts);
            value = torch::empty({batch, num_kv_heads, max_seq, head_dim}, opts);
            len      = 0;
            capacity = max_seq;
        }

        void Clear()
        {
            if (capacity > 0)
            {
                len = 0;
            }
            else
            {
                key   = torch::Tensor();
                value = torch::Tensor();
            }
        }
    };
}
