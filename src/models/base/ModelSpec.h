#pragma once

#include <string>
#include "ModelType.h"

namespace mllm
{
    struct ModelSpec
    {
        bool tie_word_embeddings = false;
        bool use_gqa = true;
        bool use_swiglu = true;
        bool use_rmsnorm = true;
        bool use_rope = true;
        bool use_qk_norm = false;

        int num_attention_heads = 32;
        int num_key_value_heads = 32;
    };

    inline ModelSpec GetModelSpec(ModelType type)
    {
        switch (type)
        {
            case ModelType::Qwen3:
            {
                ModelSpec spec;
                spec.tie_word_embeddings = false;
                spec.use_gqa = true;
                spec.use_qk_norm = true;
                spec.num_attention_heads = 32;
                spec.num_key_value_heads = 8;
                return spec;
            }

            case ModelType::Llama:
            default:
            {
                ModelSpec spec;
                spec.tie_word_embeddings = true;
                spec.use_gqa = true;
                spec.use_qk_norm = false;
                return spec;
            }
        }
    }
}