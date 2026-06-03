#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "models/base/GenerateOptions.h"
#include "models/base/GenerateResult.h"

namespace mllm
{
    // Snapshot of KV cache state for prefix caching.
    // Keys/values are stored on CPU to allow long-term storage.
    struct KVSnapshot
    {
        std::vector<torch::Tensor> keys;    // [num_layers], each [1, kv_heads, len, head_dim]
        std::vector<torch::Tensor> values;
        int64_t len = 0;

        bool empty() const { return len == 0 || keys.empty(); }
    };
}

namespace mllm
{
    struct ModelConfig
    {
        std::string model_name;
        int hidden_size = 0;
        int num_layers = 0;
        int num_attention_heads = 0;
        int num_key_value_heads = 0;
        int vocab_size = 0;
        int max_position_embeddings = 0;
        int intermediate_size = 0;
        int head_dim = 0;
        int rope_dim = 0;  // partial RoPE: only first rope_dim dims rotated (0 = use head_dim)
        float rms_norm_eps = 1e-5f;
        float rope_theta = 10000.0f;
        float local_rope_theta = 0.0f;  // Gemma 4 local-layer RoPE base (0 = use rope_theta)
        bool tie_word_embeddings = false;

        // Sliding window attention (Gemma 4) — 0 means full attention for all layers
        int sliding_window_size = 0;
        // Gemma 4 Per-Layer Input dim (D_ple) — 0 means no AltUP mechanism
        int hidden_size_per_layer_input = 0;

        // Qwen3.5 hybrid (Gated DeltaNet) — zero means pure transformer
        // Reused for Gemma 4 global attention interval (every N-th layer uses full attention)
        int full_attention_interval = 0;
        int ssm_num_v_heads  = 0;  // DeltaNet value/state heads
        int ssm_num_k_heads  = 0;  // DeltaNet key heads
        int ssm_head_v_dim   = 0;  // state dim per value head (= state_size)
        int ssm_head_k_dim   = 0;  // key dim per key head (derived = head_v_dim)
        int ssm_inner_size   = 0;  // value_dim = num_v_heads * head_v_dim
        int ssm_conv_dim     = 0;  // conv channels = 2*key_dim + value_dim
        int ssm_conv_kernel  = 4;  // causal conv kernel size
    };

    class IModelRunner
    {
    public:
        virtual ~IModelRunner() = default;

        // config.json + weights + tokenizer metadata 로드
        virtual bool Load(const std::string& model_path) = 0;

        // 1 step forward
        virtual torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& attention_mask) = 0;

        // KV cache 초기화
        virtual void InitKVCache(int batch_size, int max_seq_len) = 0;

        // config 접근
        virtual const ModelConfig& GetConfig() const = 0;

        // 모델 타입 식별
        virtual std::string GetModelType() const = 0;

        // Multi-token generation loop (prefill + decode)
        virtual GenerateResult Generate(
            const std::vector<int64_t>& input_ids,
            const GenerateOptions& options) = 0;

        // Prefix caching: export KV state for the first `len` tokens (CPU tensors).
        // Returns empty snapshot if unsupported or caches not initialized.
        // Currently implemented: QwenRunner only.
        virtual KVSnapshot GetKVSnapshot(int64_t len) const { (void)len; return {}; }

        // Restore KV state from a snapshot (GPU transfer happens inside).
        // Must be called before Generate() when prefix_kv_len > 0.
        // No-op if unsupported (safe to call unconditionally).
        virtual void SetKVSnapshot(const KVSnapshot& snap) { (void)snap; }

        // Optional diagnostics path. Production runners should keep this off.
        virtual void SetParityMode(bool enabled)
        {
            (void)enabled;
        }

        virtual void SetParityReferenceDir(const std::string& path)
        {
            (void)path;
        }
    };

    using ModelRunnerPtr = std::shared_ptr<IModelRunner>;
}
