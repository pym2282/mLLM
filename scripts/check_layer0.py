"""
Manual layer-0 forward pass comparison.
Compare mLLM's per-step hidden state with Python reference.
"""
import struct, torch, numpy as np

CACHE = "models/gemma-4-E2B-it-Q4_K_M.gguf.mlm"

def load_cache(path):
    with open(path, 'rb') as f:
        f.read(8); f.read(16)
        n = struct.unpack('<I', f.read(4))[0]
        c10 = {6: torch.float32, 5: torch.float16, 15: torch.bfloat16,
               3: torch.int32, 4: torch.int64, 0: torch.uint8, 1: torch.int8}
        W = {}
        sizes = {6: 4, 5: 2, 15: 2, 3: 4, 4: 8, 0: 1, 1: 1}
        for _ in range(n):
            nl = struct.unpack('<I', f.read(4))[0]
            name = f.read(nl).decode()
            nd = struct.unpack('<I', f.read(4))[0]
            shape = list(struct.unpack(f'<{nd}q', f.read(nd*8)))
            did = f.read(1)[0]
            numel = 1
            for d in shape: numel *= d
            raw = f.read(numel * sizes.get(did, 2))
            t = torch.frombuffer(bytearray(raw), dtype=c10.get(did, torch.bfloat16)).reshape(shape)
            W[name] = t.float()
    return W

print("Loading weights...")
W = load_cache(CACHE)
print(f"Loaded {len(W)} tensors")

def gemma_rms_norm(x, w, eps=1e-6):
    xf = x.float()
    wf = w.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, True) + eps)
    return (xf * rms * (1.0 + wf)).to(x.dtype)

# BOS token = 2
input_ids = torch.tensor([[2]], dtype=torch.long)
H = 1536
eps = 1e-6
sqrt_H = H ** 0.5

# Step 1: Embedding
embed = W["model.embed_tokens.weight"]  # [262144, 1536]
hidden = embed[input_ids[0]].unsqueeze(0).float()  # [1, 1, 1536]
hidden = hidden * sqrt_H
print(f"After embedding: norm={hidden[0,0].norm().item():.4f}")

# Step 2: Layer 0
p = "model.layers.0"
lw_input_ln = W[f"{p}.input_layernorm.weight"]
lw_post_attn_norm = W[f"{p}.post_attn_norm.weight"]
lw_post_attention_ln = W[f"{p}.post_attention_layernorm.weight"]
lw_post_ffn_norm = W[f"{p}.post_ffn_norm.weight"]
w_q = W[f"{p}.self_attn.q_proj.weight"]  # [2048, 1536]
w_k = W[f"{p}.self_attn.k_proj.weight"]  # [256, 1536]
w_v = W[f"{p}.self_attn.v_proj.weight"]  # [256, 1536]
w_o = W[f"{p}.self_attn.o_proj.weight"]  # [1536, 2048]
w_qn = W[f"{p}.self_attn.q_norm.weight"]  # [256]
w_kn = W[f"{p}.self_attn.k_norm.weight"]  # [256]
w_gate = W[f"{p}.mlp.gate_proj.weight"]  # [6144, 1536]
w_up   = W[f"{p}.mlp.up_proj.weight"]    # [6144, 1536]
w_down = W[f"{p}.mlp.down_proj.weight"]  # [1536, 6144]

# Attention
residual = hidden.clone()
h = gemma_rms_norm(hidden, lw_input_ln, eps)

n_kv = 1; n_h = 8; hd = 256
q_raw = torch.nn.functional.linear(h, w_q)  # [1,1,2048]
k_raw = torch.nn.functional.linear(h, w_k)  # [1,1,256]
v_raw = torch.nn.functional.linear(h, w_v)  # [1,1,256]

q = q_raw.view(1, 1, n_h, hd).transpose(1, 2)  # [1,8,1,256]
k = k_raw.view(1, 1, n_kv, hd).transpose(1, 2)  # [1,1,1,256]
v = v_raw.view(1, 1, n_kv, hd).transpose(1, 2)  # [1,1,1,256]

# QK-norm
q = gemma_rms_norm(q, w_qn, eps)
k = gemma_rms_norm(k, w_kn, eps)

# RoPE (position 0 → cos=1, sin=0 → no rotation for first token)
# For position 0, RoPE is identity → skip

# GQA expand
k = k.repeat_interleave(n_h // n_kv, 1)  # [1,8,1,256]
v = v.repeat_interleave(n_h // n_kv, 1)

# SDPA (causal, scale=1/sqrt(256))
scale = 1.0 / (256 ** 0.5)
attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

h = attn_out.transpose(1, 2).contiguous().view(1, 1, n_h * hd)  # [1,1,2048]
h = torch.nn.functional.linear(h, w_o)  # [1,1,1536]
h = gemma_rms_norm(h, lw_post_attn_norm, eps)
hidden = residual + h
print(f"After layer0 attn: norm={hidden[0,0].norm().item():.4f}")

# FFN
residual = hidden.clone()
h = gemma_rms_norm(hidden, lw_post_attention_ln, eps)
gate = torch.nn.functional.gelu(torch.nn.functional.linear(h, w_gate), approximate='tanh')
up   = torch.nn.functional.linear(h, w_up)
h    = torch.nn.functional.linear(gate * up, w_down)
h = gemma_rms_norm(h, lw_post_ffn_norm, eps)
hidden = residual + h
print(f"After layer0 FFN: norm={hidden[0,0].norm().item():.4f}")

# Final for comparison
print(f"\nlayer0 expected (from mLLM per-layer print):")
print("After attn ~2054, After FFN ~2167")
