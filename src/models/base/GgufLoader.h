// src/models/base/GgufLoader.h
//
// GGUF v1-v3 parser + dequantization -> BF16 torch::Tensor
//
// Supported tensor types:
//   F32, F16, BF16            — direct copy / cast
//   Q4_0                      — 18B / 32 elements
//   Q8_0                      — 34B / 32 elements
//   Q4_K (Q4_K_S, Q4_K_M)    — 144B / 256 elements
//   Q6_K                      — 210B / 256 elements
//
// Name mapping: GGUF blk.{i}.* → HuggingFace model.layers.{i}.*
// The returned tensor map is a drop-in replacement for SafeTensorTensorLoader output.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <future>
#include <thread>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

#include <torch/torch.h>
#include "models/base/IModelRunner.h"

namespace mllm
{

class GgufLoader
{
    // ------------------------------------------------------------------
    // GGUF constants
    // ------------------------------------------------------------------
    static constexpr uint32_t GGUF_MAGIC = 0x46554747u; // "GGUF" LE
    static constexpr uint64_t GGUF_ALIGN = 32;

    // KV value type ids
    enum GgufVT : uint32_t
    {
        VT_UINT8   = 0,
        VT_INT8    = 1,
        VT_UINT16  = 2,
        VT_INT16   = 3,
        VT_UINT32  = 4,
        VT_INT32   = 5,
        VT_FLOAT32 = 6,
        VT_BOOL    = 7,
        VT_STRING  = 8,
        VT_ARRAY   = 9,
        VT_UINT64  = 10,
        VT_INT64   = 11,
        VT_FLOAT64 = 12,
    };

    // GGML tensor types
    enum GgmlType : uint32_t
    {
        GT_F32  = 0,
        GT_F16  = 1,
        GT_Q4_0 = 2,
        GT_Q4_1 = 3,
        GT_Q5_0 = 6,
        GT_Q5_1 = 7,
        GT_Q8_0 = 8,
        GT_Q8_1 = 9,
        GT_Q2_K = 10,
        GT_Q3_K = 11,
        GT_Q4_K = 12,
        GT_Q5_K = 13,
        GT_Q6_K = 14,
        GT_BF16 = 30,
    };

    // ------------------------------------------------------------------
    // Block structures (must match ggml-quants.h exactly)
    // ------------------------------------------------------------------
#pragma pack(push, 1)
    struct block_q4_0 { uint16_t d; uint8_t qs[16]; };           // 18 B / 32 el
    struct block_q8_0 { uint16_t d; int8_t  qs[32]; };           // 34 B / 32 el
    struct block_q4_K { uint16_t d, dmin; uint8_t sc[12]; uint8_t qs[128]; }; // 144 B / 256 el
    struct block_q5_K { uint16_t d, dmin; uint8_t sc[12]; uint8_t qh[32]; uint8_t qs[128]; }; // 176 B / 256 el
    struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t sc[16]; uint16_t d; }; // 210 B / 256 el
#pragma pack(pop)

    static_assert(sizeof(block_q4_0) == 18,  "q4_0 block size mismatch");
    static_assert(sizeof(block_q8_0) == 34,  "q8_0 block size mismatch");
    static_assert(sizeof(block_q4_K) == 144, "q4_K block size mismatch");
    static_assert(sizeof(block_q5_K) == 176, "q5_K block size mismatch");
    static_assert(sizeof(block_q6_K) == 210, "q6_K block size mismatch");

    // ------------------------------------------------------------------
    // Tensor metadata entry
    // ------------------------------------------------------------------
    struct TensorInfo
    {
        std::string              name;
        GgmlType                 type;
        std::vector<uint64_t>    shape;   // GGUF order: dim[0]=innermost
        uint64_t                 offset;  // relative to data section start
    };

    // ------------------------------------------------------------------
    // Parse output
    // ------------------------------------------------------------------
    struct ParseResult
    {
        uint32_t                                      version;
        std::unordered_map<std::string, std::string>  kv_str;
        std::unordered_map<std::string, double>       kv_num;
        std::vector<TensorInfo>                       tensors;
        uint64_t                                      data_offset; // absolute file pos
        // Tokenizer arrays (populated when present)
        std::vector<std::string> tok_tokens;   // tokenizer.ggml.tokens
        std::vector<int32_t>     tok_types;    // tokenizer.ggml.token_type
        std::vector<std::string> tok_merges;   // tokenizer.ggml.merges
    };

public:
    // Tokenizer data extracted from GGUF metadata
    struct TokenizerData
    {
        std::string              model_type;    // "llama", "gpt2", etc.
        std::vector<std::string> tokens;        // vocab (index = token id)
        std::vector<int32_t>     token_types;   // 0=normal,1=unknown,2=control,3=user,4=unused,5=byte
        std::vector<std::string> merges;        // "a b" format BPE merge rules
        int64_t bos_id       = -1;
        int64_t eos_id       = -1;
        std::string chat_template;
    };

    static TokenizerData LoadTokenizerData(const std::string& path)
    {
        auto pr = Parse(path);
        TokenizerData td;
        td.model_type    = pr.kv_str.count("tokenizer.ggml.model")
                           ? pr.kv_str.at("tokenizer.ggml.model") : "";
        td.tokens        = std::move(pr.tok_tokens);
        td.token_types   = std::move(pr.tok_types);
        td.merges        = std::move(pr.tok_merges);
        td.chat_template = pr.kv_str.count("tokenizer.chat_template")
                           ? pr.kv_str.at("tokenizer.chat_template") : "";
        if (pr.kv_num.count("tokenizer.ggml.bos_token_id"))
            td.bos_id = static_cast<int64_t>(pr.kv_num.at("tokenizer.ggml.bos_token_id"));
        if (pr.kv_num.count("tokenizer.ggml.eos_token_id"))
            td.eos_id = static_cast<int64_t>(pr.kv_num.at("tokenizer.ggml.eos_token_id"));
        return td;
    }

private:

    // ------------------------------------------------------------------
    // Low-level binary readers
    // ------------------------------------------------------------------
    static uint8_t  ru8 (std::ifstream& f) { uint8_t  v; f.read(reinterpret_cast<char*>(&v), 1); return v; }
    static uint16_t ru16(std::ifstream& f) { uint16_t v; f.read(reinterpret_cast<char*>(&v), 2); return v; }
    static uint32_t ru32(std::ifstream& f) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v; }
    static uint64_t ru64(std::ifstream& f) { uint64_t v; f.read(reinterpret_cast<char*>(&v), 8); return v; }
    static int8_t   ri8 (std::ifstream& f) { int8_t   v; f.read(reinterpret_cast<char*>(&v), 1); return v; }
    static int32_t  ri32(std::ifstream& f) { int32_t  v; f.read(reinterpret_cast<char*>(&v), 4); return v; }
    static int64_t  ri64(std::ifstream& f) { int64_t  v; f.read(reinterpret_cast<char*>(&v), 8); return v; }
    static float    rf32(std::ifstream& f) { float    v; f.read(reinterpret_cast<char*>(&v), 4); return v; }
    static double   rf64(std::ifstream& f) { double   v; f.read(reinterpret_cast<char*>(&v), 8); return v; }

    static std::string rstr(std::ifstream& f)
    {
        uint64_t n = ru64(f);
        std::string s(n, '\0');
        f.read(s.data(), static_cast<std::streamsize>(n));
        return s;
    }

    // ------------------------------------------------------------------
    // FP16 bit-pattern → float32
    // ------------------------------------------------------------------
    static float fp16f(uint16_t h)
    {
        const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
        const uint32_t e = static_cast<uint32_t>(h & 0x7C00u) >> 10;
        const uint32_t m = static_cast<uint32_t>(h & 0x03FFu);
        if (e == 0)
        {
            // fp16 subnormal: value = ±m × 2^(-24)
            // (zero when m==0, subnormal fp16 otherwise)
            float r = std::ldexp(static_cast<float>(m), -24);
            if (s) r = -r;
            return r;
        }
        if (e == 31)
        {
            uint32_t u = s | 0x7F800000u | (m << 13);
            float r; std::memcpy(&r, &u, 4); return r;
        }
        uint32_t u = s | ((e + 112u) << 23) | (m << 13);
        float r; std::memcpy(&r, &u, 4); return r;
    }

    // ------------------------------------------------------------------
    // Skip one KV value (for types we don't need)
    // ------------------------------------------------------------------
    static void skipVal(std::ifstream& f, uint32_t vt)
    {
        switch (static_cast<GgufVT>(vt))
        {
            case VT_UINT8: case VT_INT8: case VT_BOOL:
                f.seekg(1, std::ios::cur); break;
            case VT_UINT16: case VT_INT16:
                f.seekg(2, std::ios::cur); break;
            case VT_UINT32: case VT_INT32: case VT_FLOAT32:
                f.seekg(4, std::ios::cur); break;
            case VT_UINT64: case VT_INT64: case VT_FLOAT64:
                f.seekg(8, std::ios::cur); break;
            case VT_STRING:
                rstr(f); break;
            case VT_ARRAY: {
                uint32_t avt = ru32(f);
                uint64_t n   = ru64(f);
                for (uint64_t i = 0; i < n; ++i) skipVal(f, avt);
                break;
            }
            default:
                throw std::runtime_error(
                    "GgufLoader: unknown KV value type " + std::to_string(vt));
        }
    }

    // Read a scalar numeric KV value as double (avoids switch duplication)
    static double readNum(std::ifstream& f, uint32_t vt)
    {
        switch (static_cast<GgufVT>(vt))
        {
            case VT_UINT8:   return static_cast<double>(ru8(f));
            case VT_INT8:    return static_cast<double>(ri8(f));
            case VT_UINT16:  return static_cast<double>(ru16(f));
            case VT_INT16:   { int16_t v; f.read(reinterpret_cast<char*>(&v), 2); return static_cast<double>(v); }
            case VT_UINT32:  return static_cast<double>(ru32(f));
            case VT_INT32:   return static_cast<double>(ri32(f));
            case VT_FLOAT32: return static_cast<double>(rf32(f));
            case VT_BOOL:    return static_cast<double>(ru8(f));
            case VT_UINT64:  return static_cast<double>(ru64(f));
            case VT_INT64:   return static_cast<double>(ri64(f));
            case VT_FLOAT64: return rf64(f);
            default: return 0.0;
        }
    }

    // ------------------------------------------------------------------
    // Parse full GGUF header (KV + tensor index)
    // ------------------------------------------------------------------
    static ParseResult Parse(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("GgufLoader: cannot open " + path);

        uint32_t magic = ru32(f);
        if (magic != GGUF_MAGIC)
            throw std::runtime_error("GgufLoader: not a GGUF file: " + path);

        ParseResult r;
        r.version         = ru32(f);
        uint64_t n_tensors = ru64(f);
        uint64_t n_kv      = ru64(f);

        // -- KV metadata --
        for (uint64_t i = 0; i < n_kv; ++i)
        {
            std::string key = rstr(f);
            uint32_t    vt  = ru32(f);

            if (vt == static_cast<uint32_t>(VT_STRING))
            {
                r.kv_str[key] = rstr(f);
            }
            else if (vt == static_cast<uint32_t>(VT_ARRAY))
            {
                uint32_t elem_vt = ru32(f);
                uint64_t n       = ru64(f);

                if (key == "tokenizer.ggml.tokens" && elem_vt == VT_STRING)
                {
                    r.tok_tokens.reserve(static_cast<size_t>(n));
                    for (uint64_t j = 0; j < n; ++j)
                        r.tok_tokens.push_back(rstr(f));
                }
                else if (key == "tokenizer.ggml.token_type")
                {
                    r.tok_types.reserve(static_cast<size_t>(n));
                    for (uint64_t j = 0; j < n; ++j)
                        r.tok_types.push_back(static_cast<int32_t>(readNum(f, elem_vt)));
                }
                else if (key == "tokenizer.ggml.merges" && elem_vt == VT_STRING)
                {
                    r.tok_merges.reserve(static_cast<size_t>(n));
                    for (uint64_t j = 0; j < n; ++j)
                        r.tok_merges.push_back(rstr(f));
                }
                else
                {
                    // Skip remaining array elements
                    for (uint64_t j = 0; j < n; ++j)
                        skipVal(f, elem_vt);
                }
            }
            else
            {
                r.kv_num[key] = readNum(f, vt);
            }
        }

        // -- Tensor info --
        r.tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i)
        {
            auto& ti    = r.tensors[i];
            ti.name     = rstr(f);
            uint32_t nd = ru32(f);
            ti.shape.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) ti.shape[d] = ru64(f);
            ti.type     = static_cast<GgmlType>(ru32(f));
            ti.offset   = ru64(f);
        }

        // -- Data section start (aligned to GGUF_ALIGN bytes) --
        uint64_t pos = static_cast<uint64_t>(f.tellg());
        r.data_offset = (pos + GGUF_ALIGN - 1) / GGUF_ALIGN * GGUF_ALIGN;

        return r;
    }

    // ------------------------------------------------------------------
    // Compute raw byte count for a tensor
    // ------------------------------------------------------------------
    static uint64_t byteCount(const TensorInfo& ti)
    {
        uint64_t n = 1;
        for (auto d : ti.shape) n *= d;

        switch (ti.type)
        {
            case GT_F32:  return n * 4;
            case GT_F16:  return n * 2;
            case GT_BF16: return n * 2;
            case GT_Q4_0: return (n / 32) * sizeof(block_q4_0);
            case GT_Q8_0: return (n / 32) * sizeof(block_q8_0);
            case GT_Q4_K: return (n / 256) * sizeof(block_q4_K);
            case GT_Q5_K: return (n / 256) * sizeof(block_q5_K);
            case GT_Q6_K: return (n / 256) * sizeof(block_q6_K);
            default:
                throw std::runtime_error(
                    "GgufLoader: unsupported type " + std::to_string(ti.type) +
                    " for tensor " + ti.name);
        }
    }

    // ------------------------------------------------------------------
    // Dequantization routines (output to float32 buffer)
    // ------------------------------------------------------------------

    static void dq_q4_0(const uint8_t* src, float* dst, int64_t n)
    {
        int64_t nb = n / 32;
        for (int64_t b = 0; b < nb; ++b)
        {
            const auto* blk = reinterpret_cast<const block_q4_0*>(src) + b;
            float d = fp16f(blk->d);
            for (int i = 0; i < 16; ++i)
            {
                dst[b * 32 + i]      = d * static_cast<float>((blk->qs[i] & 0xF) - 8);
                dst[b * 32 + i + 16] = d * static_cast<float>((blk->qs[i] >> 4)  - 8);
            }
        }
    }

    static void dq_q8_0(const uint8_t* src, float* dst, int64_t n)
    {
        int64_t nb = n / 32;
        for (int64_t b = 0; b < nb; ++b)
        {
            const auto* blk = reinterpret_cast<const block_q8_0*>(src) + b;
            float d = fp16f(blk->d);
            for (int i = 0; i < 32; ++i)
                dst[b * 32 + i] = d * static_cast<float>(blk->qs[i]);
        }
    }

    // Extract 6-bit scale/min from Q4_K superblock (matches ggml-quants.c get_scale_min_k4)
    static void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m)
    {
        if (j < 4)
        {
            d = q[j]     & 63;
            m = q[j + 4] & 63;
        }
        else
        {
            d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
        }
    }

    static void dq_q4_K(const uint8_t* src, float* dst, int64_t n)
    {
        int64_t nb = n / 256;
        for (int64_t b = 0; b < nb; ++b)
        {
            const auto* blk = reinterpret_cast<const block_q4_K*>(src) + b;
            float       d   = fp16f(blk->d);
            float       dm  = fp16f(blk->dmin);
            const uint8_t* q = blk->qs;
            float* y = dst + b * 256;
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < 256; j += 64)
            {
                get_scale_min_k4(is,     blk->sc, sc, m);
                float d1 = d * sc, m1 = dm * m;
                get_scale_min_k4(is + 1, blk->sc, sc, m);
                float d2 = d * sc, m2 = dm * m;
                for (int l = 0; l < 32; ++l) y[l]      = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) y[l + 32] = d2 * (q[l] >> 4)  - m2;
                y += 64; q += 32; is += 2;
            }
        }
    }

    // Q5_K: 256 elements / 176 bytes. High bit packed 2-per-byte across all groups.
    // Reference: ggml dequantize_row_q5_K (u1/u2 shift pattern).
    static void dq_q5_K(const uint8_t* src, float* dst, int64_t n)
    {
        int64_t nb = n / 256;
        for (int64_t b = 0; b < nb; ++b)
        {
            const auto* blk = reinterpret_cast<const block_q5_K*>(src) + b;
            float d    = fp16f(blk->d);
            float dmin = fp16f(blk->dmin);
            const uint8_t* ql = blk->qs;
            const uint8_t* qh = blk->qh;

            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            float* y = dst + b * 256;

            for (int j = 0; j < 256; j += 64)
            {
                uint8_t sc, m;
                get_scale_min_k4(is,     blk->sc, sc, m);
                float d1 = d * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, blk->sc, sc, m);
                float d2 = d * sc, m2 = dmin * m;

                for (int l = 0; l < 32; ++l)
                    *y++ = d1 * static_cast<float>((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l)
                    *y++ = d2 * static_cast<float>((ql[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2;

                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
    }

    static void dq_q6_K(const uint8_t* src, float* dst, int64_t n)
    {
        int64_t nb = n / 256;
        for (int64_t b = 0; b < nb; ++b)
        {
            const auto* blk = reinterpret_cast<const block_q6_K*>(src) + b;
            float d = fp16f(blk->d);
            const uint8_t* ql = blk->ql;
            const uint8_t* qh = blk->qh;
            const int8_t*  sc = blk->sc;
            float* y = dst + b * 256;
            for (int chunk = 0; chunk < 256; chunk += 128)
            {
                for (int l = 0; l < 32; ++l)
                {
                    const int is = l / 16;
                    // Reconstruct 6-bit signed values (range -32..31)
                    int8_t q1 = static_cast<int8_t>(((ql[l]      & 0xF) | ((qh[l] >> 0 & 3) << 4)) - 32);
                    int8_t q2 = static_cast<int8_t>(((ql[l + 32]  & 0xF) | ((qh[l] >> 2 & 3) << 4)) - 32);
                    int8_t q3 = static_cast<int8_t>(((ql[l]      >> 4)  | ((qh[l] >> 4 & 3) << 4)) - 32);
                    int8_t q4 = static_cast<int8_t>(((ql[l + 32]  >> 4)  | ((qh[l] >> 6 & 3) << 4)) - 32);
                    y[l]       = d * sc[is + 0] * q1;
                    y[l + 32]  = d * sc[is + 2] * q2;
                    y[l + 64]  = d * sc[is + 4] * q3;
                    y[l + 96]  = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
    }

    // ------------------------------------------------------------------
    // Convert raw bytes for one tensor -> BF16 torch::Tensor (on CPU)
    //
    // GGUF shape is innermost-first; we reverse to get PyTorch outermost-first.
    // Data layout is identical after reversing (row-major in both cases).
    // ------------------------------------------------------------------
    static torch::Tensor dequant(const uint8_t* raw, const TensorInfo& ti)
    {
        // Reverse GGUF shape → PyTorch shape
        std::vector<int64_t> shape(ti.shape.rbegin(), ti.shape.rend());
        int64_t n = 1;
        for (auto d : shape) n *= d;

        if (ti.type == GT_BF16)
            return torch::from_blob(const_cast<uint8_t*>(raw), shape, torch::kBFloat16).clone();
        if (ti.type == GT_F16)
            return torch::from_blob(const_cast<uint8_t*>(raw), shape, torch::kFloat16).clone()
                         .to(torch::kBFloat16);
        if (ti.type == GT_F32)
            return torch::from_blob(const_cast<uint8_t*>(raw), shape, torch::kFloat32).clone()
                         .to(torch::kBFloat16);

        // Quantized: unpack -> float32 -> BF16
        auto f32 = torch::empty(shape, torch::kFloat32);
        float* dst = f32.data_ptr<float>();

        switch (ti.type)
        {
            case GT_Q4_0: dq_q4_0(raw, dst, n); break;
            case GT_Q8_0: dq_q8_0(raw, dst, n); break;
            case GT_Q4_K: dq_q4_K(raw, dst, n); break;
            case GT_Q5_K: dq_q5_K(raw, dst, n); break;
            case GT_Q6_K: dq_q6_K(raw, dst, n); break;
            default:
                throw std::runtime_error(
                    "GgufLoader: unsupported quant type " +
                    std::to_string(ti.type) + " for tensor " + ti.name);
        }

        return f32.to(torch::kBFloat16);
    }

    // ------------------------------------------------------------------
    // GGUF name → HuggingFace name (Qwen3 / Qwen2 family)
    //
    // GGUF pattern: blk.{i}.{key}
    // HF  pattern:  model.layers.{i}.{hf_key}
    // ------------------------------------------------------------------
    static std::string mapName(const std::string& gguf)
    {
        // blk.{i}.* → model.layers.{i}.*
        if (gguf.size() > 4 && gguf.compare(0, 4, "blk.") == 0)
        {
            size_t dot = gguf.find('.', 4);
            if (dot != std::string::npos)
            {
                const std::string idx  = gguf.substr(4, dot - 4);
                const std::string rest = gguf.substr(dot + 1);
                const std::string p    = "model.layers." + idx;

                // Full-attention layer weights (Qwen3/Qwen2/LLaMA style)
                if (rest == "attn_q.weight")       return p + ".self_attn.q_proj.weight";
                if (rest == "attn_k.weight")       return p + ".self_attn.k_proj.weight";
                if (rest == "attn_v.weight")       return p + ".self_attn.v_proj.weight";
                if (rest == "attn_output.weight")  return p + ".self_attn.o_proj.weight";
                if (rest == "attn_q_norm.weight")  return p + ".self_attn.q_norm.weight";
                if (rest == "attn_k_norm.weight")  return p + ".self_attn.k_norm.weight";
                if (rest == "attn_q.bias")         return p + ".self_attn.q_proj.bias";
                if (rest == "attn_k.bias")         return p + ".self_attn.k_proj.bias";
                if (rest == "attn_v.bias")         return p + ".self_attn.v_proj.bias";
                if (rest == "ffn_gate.weight")     return p + ".mlp.gate_proj.weight";
                if (rest == "ffn_up.weight")       return p + ".mlp.up_proj.weight";
                if (rest == "ffn_down.weight")     return p + ".mlp.down_proj.weight";
                if (rest == "attn_norm.weight")    return p + ".input_layernorm.weight";
                if (rest == "ffn_norm.weight")          return p + ".post_attention_layernorm.weight";
                if (rest == "post_attention_norm.weight") return p + ".post_attn_norm.weight";
                if (rest == "post_norm.weight")           return p + ".post_attn_norm.weight"; // Gemma 4
                if (rest == "post_ffw_norm.weight")       return p + ".post_ffn_norm.weight";
                // Gemma 4 Per-Layer Input (AltUP)
                if (rest == "inp_gate.weight")            return p + ".per_layer_inp_gate.weight";
                if (rest == "proj.weight")                return p + ".per_layer_proj.weight";
                if (rest == "layer_output_scale.weight")  return p + ".layer_scalar.weight";

                // Qwen3.5 hybrid (Gated DeltaNet) linear-attention layer weights
                if (rest == "attn_qkv.weight")   return p + ".linear_attn.in_proj_qkv.weight";
                if (rest == "attn_gate.weight")  return p + ".linear_attn.in_proj_z.weight";
                if (rest == "ssm_a")             return p + ".linear_attn.A_log";
                if (rest == "ssm_alpha.weight")  return p + ".linear_attn.in_proj_a.weight";
                if (rest == "ssm_beta.weight")   return p + ".linear_attn.in_proj_b.weight";
                if (rest == "ssm_conv1d.weight") return p + ".linear_attn.conv1d.weight";
                if (rest == "ssm_dt.bias")       return p + ".linear_attn.dt_bias";
                if (rest == "ssm_norm.weight")   return p + ".linear_attn.norm.weight";
                if (rest == "ssm_out.weight")    return p + ".linear_attn.out_proj.weight";

                std::cerr << "[GgufLoader] unmapped blk tensor: " << gguf << "\n";
                return gguf;
            }
        }

        if (gguf == "token_embd.weight")  return "model.embed_tokens.weight";
        if (gguf == "output_norm.weight") return "model.norm.weight";
        if (gguf == "output.weight")      return "lm_head.weight";

        // rope_freqs.weight and similar — pass through, runner ignores unknown names
        return gguf;
    }

public:
    // ------------------------------------------------------------------
    // Disk cache helpers: skip dequantization/SafeTensors loading on repeated runs
    // src_path: source file to validate against (GGUF path or config.json)
    // cache_path: where to store the cache (*.mlm)
    // ------------------------------------------------------------------
    static constexpr uint32_t CACHE_MAGIC   = 0x4D4C4D43u;
    static constexpr uint32_t CACHE_VERSION = 2u;

    static bool IsCacheValid(const std::string& gguf_path, const std::string& cache_path)
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (!fs::exists(cache_path, ec)) return false;

        // Cache must be non-empty
        if (fs::file_size(cache_path, ec) < 32) return false;

        std::ifstream cf(cache_path, std::ios::binary);
        if (!cf) return false;

        uint32_t magic, ver;
        cf.read(reinterpret_cast<char*>(&magic), 4);
        cf.read(reinterpret_cast<char*>(&ver),   4);
        if (magic != CACHE_MAGIC || ver != CACHE_VERSION) return false;

        uint64_t saved_sz, saved_mt;
        cf.read(reinterpret_cast<char*>(&saved_sz), 8);
        cf.read(reinterpret_cast<char*>(&saved_mt), 8);

        // Validate GGUF size
        uint64_t cur_sz = static_cast<uint64_t>(fs::file_size(gguf_path, ec));
        if (ec || cur_sz != saved_sz) return false;

        // Validate GGUF mtime (skip if unsupported)
        auto mt = fs::last_write_time(gguf_path, ec);
        if (!ec)
        {
            uint64_t cur_mt = static_cast<uint64_t>(mt.time_since_epoch().count());
            if (cur_mt != saved_mt) return false;
        }

        return true;
    }

    // mmap 핸들러: RAII 래퍼 (Windows)
    struct MmapView
    {
        void*  base = nullptr;
        size_t size = 0;
#ifdef _WIN32
        HANDLE hFile = INVALID_HANDLE_VALUE;
        HANDLE hMap  = INVALID_HANDLE_VALUE;
#else
        int fd = -1;
#endif
        ~MmapView() { release(); }

        bool open(const std::string& path)
        {
            release();
#ifdef _WIN32
            hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (hFile == INVALID_HANDLE_VALUE) return false;
            LARGE_INTEGER sz; GetFileSizeEx(hFile, &sz);
            size = static_cast<size_t>(sz.QuadPart);
            hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (!hMap) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; return false; }
            base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
            if (!base) { CloseHandle(hMap); CloseHandle(hFile); hMap=INVALID_HANDLE_VALUE; hFile=INVALID_HANDLE_VALUE; return false; }
#else
            fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) return false;
            struct stat st; fstat(fd, &st); size = st.st_size;
            base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (base == MAP_FAILED) { ::close(fd); fd=-1; base=nullptr; return false; }
#endif
            return true;
        }

        void release()
        {
            if (!base) return;
#ifdef _WIN32
            UnmapViewOfFile(base); CloseHandle(hMap); CloseHandle(hFile);
            base=nullptr; hMap=INVALID_HANDLE_VALUE; hFile=INVALID_HANDLE_VALUE;
#else
            ::munmap(base, size); ::close(fd); base=nullptr; fd=-1;
#endif
        }

        MmapView() = default;
        MmapView(const MmapView&) = delete;
        MmapView& operator=(const MmapView&) = delete;
    };

    static std::unordered_map<std::string, torch::Tensor>
    LoadCache(const std::string& cache_path)
    {
        // mmap 파일 전체를 메모리에 매핑 (zero-copy 읽기)
        auto mmap = std::make_shared<MmapView>();
        if (!mmap->open(cache_path)) return {};

        const uint8_t* p = static_cast<const uint8_t*>(mmap->base);
        const uint8_t* end = p + mmap->size;

        auto read_u32 = [&](uint32_t& v) { memcpy(&v, p, 4); p += 4; };
        auto read_u64 = [&](uint64_t& v) { memcpy(&v, p, 8); p += 8; };

        // 헤더 건너뜀 (IsCacheValid에서 이미 검증)
        p += 4 + 4 + 8 + 8;  // magic + ver + src_sz + src_mt

        uint32_t n_tensors; read_u32(n_tensors);

        std::unordered_map<std::string, torch::Tensor> out;
        out.reserve(n_tensors);

        for (uint32_t i = 0; i < n_tensors; ++i)
        {
            uint32_t name_len; read_u32(name_len);
            std::string name(reinterpret_cast<const char*>(p), name_len); p += name_len;

            uint32_t ndims; read_u32(ndims);
            std::vector<int64_t> shape(ndims);
            memcpy(shape.data(), p, ndims * 8); p += ndims * 8;

            int64_t numel = 1;
            for (auto d : shape) numel *= d;

            if (p + numel * 2 > end) return {};

            // mmap 영역을 from_blob으로 직접 참조한 후 clone (zero-copy read)
            auto t = torch::from_blob(
                const_cast<uint8_t*>(p), shape, torch::kBFloat16).clone();
            p += numel * 2;

            out[name] = std::move(t);

            if ((i + 1) % 100 == 0 || i + 1 == n_tensors)
                std::cerr << "[GgufLoader] cache " << (i + 1) << "/" << n_tensors << "\r";
        }

        std::cerr << "\n[GgufLoader] Cache loaded (" << n_tensors << " tensors, mmap).\n";
        return out;
    }

    static void SaveCache(
        const std::string& cache_path,
        const std::string& gguf_path,
        const std::unordered_map<std::string, torch::Tensor>& tensors)
    {
        namespace fs = std::filesystem;
        std::error_code ec;

        uint64_t src_sz = static_cast<uint64_t>(fs::file_size(gguf_path, ec));
        if (ec) return;
        auto mt = fs::last_write_time(gguf_path, ec);
        if (ec) return;
        uint64_t src_mt = static_cast<uint64_t>(mt.time_since_epoch().count());

        std::ofstream cf(cache_path, std::ios::binary);
        if (!cf) return;

        uint32_t magic = CACHE_MAGIC, ver = CACHE_VERSION;
        cf.write(reinterpret_cast<char*>(&magic),  4);
        cf.write(reinterpret_cast<char*>(&ver),    4);
        cf.write(reinterpret_cast<char*>(&src_sz), 8);
        cf.write(reinterpret_cast<char*>(&src_mt), 8);

        uint32_t n = static_cast<uint32_t>(tensors.size());
        cf.write(reinterpret_cast<char*>(&n), 4);

        for (const auto& [name, tensor] : tensors)
        {
            uint32_t nl = static_cast<uint32_t>(name.size());
            cf.write(reinterpret_cast<char*>(&nl), 4);
            cf.write(name.data(), nl);

            auto shape = tensor.sizes();
            uint32_t nd = static_cast<uint32_t>(shape.size());
            cf.write(reinterpret_cast<char*>(&nd), 4);
            cf.write(reinterpret_cast<const char*>(shape.data()), nd * 8);

            auto t_bf16 = tensor.to(torch::kBFloat16).contiguous();
            cf.write(reinterpret_cast<char*>(t_bf16.data_ptr()), t_bf16.numel() * 2);
        }

        if (cf.good())
            std::cerr << "[GgufLoader] Cache saved: " << cache_path << "\n";
        else
        {
            cf.close();
            fs::remove(cache_path, ec);  // partial file 삭제
        }
    }

public:
    // ------------------------------------------------------------------
    // Load all tensors from a .gguf file.
    // 캐시(*.gguf.mlm)가 있으면 역양자화 없이 직접 로드.
    // 없으면 역양자화 + 병렬 처리 + 캐시 저장.
    // ------------------------------------------------------------------
    static std::unordered_map<std::string, torch::Tensor> Load(const std::string& path)
    {
        const std::string cache_path = path + ".mlm";

        // 1. 캐시 히트
        if (IsCacheValid(path, cache_path))
        {
            auto cached = LoadCache(cache_path);
            if (!cached.empty()) return cached;
            std::cerr << "[GgufLoader] Cache corrupt, regenerating...\n";
        }

        // 2. GGUF 파싱 + 역양자화 (병렬)
        auto pr = Parse(path);
        std::cerr << "[GgufLoader] GGUF v" << pr.version
                  << " | " << pr.tensors.size() << " tensors (dequantizing...)\n";

        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("GgufLoader: cannot reopen " + path);

        const size_t n = pr.tensors.size();
        const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());

        // 각 텐서의 raw bytes를 읽고 이름 매핑
        struct RawEntry {
            std::string   hf_name;
            TensorInfo    ti;
            std::vector<uint8_t> raw;
        };
        std::vector<RawEntry> entries(n);

        for (size_t i = 0; i < n; ++i)
        {
            const auto& ti = pr.tensors[i];
            uint64_t nb  = byteCount(ti);
            uint64_t pos = pr.data_offset + ti.offset;

            f.seekg(static_cast<std::streamoff>(pos));
            entries[i].hf_name = mapName(ti.name);
            entries[i].ti      = ti;
            entries[i].raw.resize(static_cast<size_t>(nb));
            f.read(reinterpret_cast<char*>(entries[i].raw.data()),
                   static_cast<std::streamsize>(nb));
        }
        f.close();

        // 역양자화: 청크로 나눠서 병렬 처리
        std::unordered_map<std::string, torch::Tensor> out;
        out.reserve(n);

        const size_t chunk = (n + n_threads - 1) / n_threads;
        std::vector<std::future<std::vector<std::pair<std::string, torch::Tensor>>>> futs;
        futs.reserve(n_threads);

        for (size_t t = 0; t < n_threads; ++t)
        {
            size_t lo = t * chunk;
            size_t hi = std::min(lo + chunk, n);
            if (lo >= n) break;

            futs.push_back(std::async(std::launch::async,
                [&entries, lo, hi]()
                {
                    std::vector<std::pair<std::string, torch::Tensor>> res;
                    res.reserve(hi - lo);
                    for (size_t k = lo; k < hi; ++k)
                        res.emplace_back(entries[k].hf_name,
                                         dequant(entries[k].raw.data(), entries[k].ti));
                    return res;
                }));
        }

        size_t done = 0;
        for (auto& fut : futs)
        {
            for (auto& [name, t] : fut.get())
                out[name] = std::move(t);
            done += chunk;
            std::cerr << "[GgufLoader] " << std::min(done, n) << "/" << n << "\r";
        }
        std::cerr << "\n[GgufLoader] Load complete.\n";

        // 3. 캐시 저장 (동기 — 다음 실행부터 즉시 히트)
        std::cerr << "[GgufLoader] Saving cache...\n";
        SaveCache(cache_path, path, out);

        return out;
    }

    // ------------------------------------------------------------------
    // Read ModelConfig from GGUF metadata (KV pairs).
    // vocab_size is derived from the token_embd tensor shape.
    // ------------------------------------------------------------------
    static ModelConfig ReadConfig(const std::string& path)
    {
        auto pr = Parse(path);

        const auto& ks = pr.kv_str;
        const auto& kn = pr.kv_num;

        const std::string arch =
            ks.count("general.architecture") ? ks.at("general.architecture") : "qwen3";

        // Helper: look up arch.key in kv_num
        auto geti = [&](const std::string& k, int def = 0) -> int
        {
            const std::string key = arch + "." + k;
            if (kn.count(key)) return static_cast<int>(kn.at(key));
            return def;
        };
        auto getf = [&](const std::string& k, double def = 0.0) -> double
        {
            const std::string key = arch + "." + k;
            if (kn.count(key)) return kn.at(key);
            return def;
        };

        ModelConfig c;
        c.model_name          = ks.count("general.name") ? ks.at("general.name") : arch;
        c.num_layers          = geti("block_count");
        c.hidden_size         = geti("embedding_length");
        c.num_attention_heads = geti("attention.head_count");
        c.num_key_value_heads = geti("attention.head_count_kv", c.num_attention_heads);
        c.intermediate_size   = geti("feed_forward_length");
        c.rms_norm_eps        = static_cast<float>(getf("attention.layer_norm_rms_epsilon", 1e-6));
        c.rope_theta          = static_cast<float>(getf("rope.freq_base", 10000.0));
        // Gemma 4: separate local-layer RoPE base (global uses rope_theta above)
        // Try multiple key names used by different converters
        for (const char* suffix : {"rope.local_freq_base", "rope.freq_base.swa",
                                    "rope.local_base", "rope.swa_freq_base"})
        {
            const std::string k = arch + "." + suffix;
            if (kn.count(k)) { c.local_rope_theta = static_cast<float>(kn.at(k)); break; }
        }
        std::cerr << "[GgufLoader] local_rope_theta=" << c.local_rope_theta << "\n";
        c.max_position_embeddings = geti("context_length", 8192);

        // head_dim: prefer explicit key, fall back to hidden/heads
        if (kn.count(arch + ".attention.key_length"))
            c.head_dim = static_cast<int>(kn.at(arch + ".attention.key_length"));
        else if (c.num_attention_heads > 0)
            c.head_dim = c.hidden_size / c.num_attention_heads;

        // partial RoPE: only first rope_dim dimensions are rotated
        c.rope_dim = geti("rope.dimension_count", 0);
        if (c.rope_dim <= 0) c.rope_dim = c.head_dim;

        // vocab_size from token_embd.weight shape
        // GGUF shape is innermost-first: [hidden_size, vocab_size]
        // so vocab_size = shape[1]
        bool has_output = false;
        for (const auto& ti : pr.tensors)
        {
            if (ti.name == "token_embd.weight" && ti.shape.size() >= 2)
                c.vocab_size = static_cast<int>(ti.shape[1]);
            if (ti.name == "output.weight")
                has_output = true;
        }
        c.tie_word_embeddings = !has_output;

        // Gemma 4: per-layer input hidden size (D_ple), 0 = no AltUP
        c.hidden_size_per_layer_input = geti("per_layer_input_hidden_size", 0);

        // Gemma 4: sliding window size (0 = full attention)
        c.sliding_window_size = geti("attention.sliding_window", 0);

        // Gemma 4 global attention interval — every N-th layer is full attention
        // Pattern [1,1,1,1,1,0] → interval=6. Same field reused for Qwen3.5 hybrid.
        if (c.sliding_window_size > 0)
            c.full_attention_interval = 6; // Gemma 4 hardcoded pattern

        // Qwen3.5 hybrid SSM fields (zero for pure-transformer models)
        c.full_attention_interval = geti("full_attention_interval", c.full_attention_interval);
        c.ssm_num_v_heads  = geti("ssm.time_step_rank", 0);
        c.ssm_num_k_heads  = geti("ssm.group_count",    0);
        c.ssm_head_v_dim   = geti("ssm.state_size",     0);
        c.ssm_inner_size   = geti("ssm.inner_size",     0);
        c.ssm_conv_kernel  = geti("ssm.conv_kernel",    4);

        if (c.ssm_num_k_heads > 0 && c.ssm_head_v_dim > 0)
        {
            // For Qwen3.5: head_k_dim == state_size (same as head_v_dim)
            c.ssm_head_k_dim = c.ssm_head_v_dim;
            const int key_dim  = c.ssm_num_k_heads * c.ssm_head_k_dim;
            c.ssm_conv_dim     = 2 * key_dim + c.ssm_inner_size;
        }

        std::cerr << "[GgufLoader] Config:"
                  << " layers=" << c.num_layers
                  << " hidden=" << c.hidden_size
                  << " heads=" << c.num_attention_heads
                  << " kv_heads=" << c.num_key_value_heads
                  << " head_dim=" << c.head_dim
                  << " rope_dim=" << c.rope_dim
                  << " vocab=" << c.vocab_size
                  << " rope_theta=" << c.rope_theta
                  << " rms_eps=" << c.rms_norm_eps
                  << " tie_emb=" << c.tie_word_embeddings
                  << "\n";

        return c;
    }

    // ------------------------------------------------------------------
    // Read general.architecture from GGUF metadata.
    // Used by ModelRunnerFactory to dispatch to the correct runner.
    // ------------------------------------------------------------------
    static std::string ReadArchitecture(const std::string& path)
    {
        auto pr = Parse(path);
        return pr.kv_str.count("general.architecture")
                   ? pr.kv_str.at("general.architecture")
                   : "unknown";
    }

    // ------------------------------------------------------------------
    // Utility: check whether path is a .gguf file
    // ------------------------------------------------------------------
    static bool IsGguf(const std::string& path)
    {
        return path.size() >= 5 &&
               path.compare(path.size() - 5, 5, ".gguf") == 0;
    }

    // ------------------------------------------------------------------
    // Utility: return the parent directory of a .gguf file path.
    // Used to locate tokenizer files stored alongside the GGUF.
    // ------------------------------------------------------------------
    static std::string ParentDir(const std::string& gguf_path)
    {
        size_t slash = gguf_path.find_last_of("/\\");
        if (slash == std::string::npos) return ".";
        return gguf_path.substr(0, slash);
    }
};

} // namespace mllm
