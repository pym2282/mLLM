"""
Load Gemma4 weights from mlm cache into HF Gemma4ForCausalLM and run forward.
This gives the "ground truth" argmax to compare against C++.
"""
import struct, torch, numpy as np
from transformers import Gemma4TextConfig, Gemma4ForCausalLM

CACHE_PATH = "models/gemma-4-E2B-it-Q4_K_M.gguf.mlm"

def load_mlm_cache(path):
    with open(path, 'rb') as f:
        data = f.read()
    p = 8  # skip magic+ver
    p += 16  # skip src_sz, src_mt
    n, = struct.unpack_from('<I', data, p); p += 4
    c10 = {6: torch.float32, 5: torch.float16, 15: torch.bfloat16,
           3: torch.int32, 4: torch.int64, 0: torch.uint8, 1: torch.int8}
    tensors = {}
    for _ in range(n):
        nl, = struct.unpack_from('<I', data, p); p += 4
        name = data[p:p+nl].decode(); p += nl
        nd, = struct.unpack_from('<I', data, p); p += 4
        shape = list(struct.unpack_from(f'<{nd}q', data, p)); p += nd*8
        dtype = c10.get(data[p], torch.bfloat16); p += 1
        numel = 1
        for d in shape: numel *= d
        nbytes = numel * torch._utils._element_size(dtype)
        arr = np.frombuffer(data[p:p+nbytes], dtype=np.uint8).copy()
        tensors[name] = torch.frombuffer(arr, dtype=dtype).reshape(shape)
        p += nbytes
    return tensors

print("Loading weights...")
W = load_mlm_cache(CACHE_PATH)
print(f"Loaded {len(W)} tensors")

# Build HF config matching the GGUF
# layer_types pattern: 5 local + 1 global, repeated
# layers 4,9,14,19,24,29,34 are global → every 5th starting from idx 4
layer_types = []
for i in range(35):
    kd = W[f"model.layers.{i}.self_attn.k_proj.weight"].shape[0]
    layer_types.append("full_attention" if kd == 512 else "sliding_attention")

print(f"layer_types: {layer_types[:10]}...")

# Detect per-layer intermediate sizes
layer_ffn_sizes = []
for i in range(35):
    gk = f"model.layers.{i}.mlp.gate_proj.weight"
    if gk in W:
        layer_ffn_sizes.append(W[gk].shape[0])
    else:
        layer_ffn_sizes.append(6144)
print("FFN sizes:", sorted(set(layer_ffn_sizes)))

# Use max intermediate_size; shape mismatches handled by manual load below
max_ffn = max(layer_ffn_sizes)
cfg = Gemma4TextConfig(
    hidden_size=1536,
    num_hidden_layers=35,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    intermediate_size=max_ffn,
    vocab_size=262144,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    sliding_window=512,
    hidden_size_per_layer_input=256,
    final_logit_softcapping=30.0,
    layer_types=layer_types,
    attention_bias=False,
    hidden_activation="gelu_pytorch_tanh",
    tie_word_embeddings=True,
)

print("Creating model (cpu)...")
model = Gemma4ForCausalLM(cfg)

# Remap our weight names to HF names
hf_state = {}
for k, v in W.items():
    k2 = k
    # Add "model." prefix only for sub-model weights (not lm_head which sits at top level)
    # Gemma4ForCausalLM: self.model = Gemma4TextModel, weights under "model.*"
    # Our keys are already "model.layers.*", "model.embed_tokens.*", "model.norm.*"
    # HF keys are also "model.layers.*", so keep as-is
    k2 = k2.replace(".post_attn_norm.", ".post_attention_layernorm.")
    if ".post_attention_layernorm." in k and ".post_attn_norm." not in k:
        k2 = k2.replace(".post_attention_layernorm.", ".pre_feedforward_layernorm.")
    k2 = k2.replace(".post_ffn_norm.", ".post_feedforward_layernorm.")
    k2 = k2.replace(".per_layer_inp_gate.", ".per_layer_input_gate.")
    k2 = k2.replace(".per_layer_proj.", ".per_layer_projection.")
    k2 = k2.replace(".per_layer_post_norm.", ".post_per_layer_input_norm.")
    k2 = k2.replace("model.per_layer_model_proj.", "model.per_layer_model_projection.")
    k2 = k2.replace("model.per_layer_proj_norm.", "model.per_layer_projection_norm.")
    k2 = k2.replace("model.per_layer_token_embd.", "model.embed_tokens_per_layer.")
    hf_state[k2] = v.float()

# Tied weights
hf_state["lm_head.weight"] = W["model.embed_tokens.weight"].float()

print("Sample HF keys:", list(hf_state.keys())[:5])

# Filter to only matching shapes before load_state_dict
model_sd = dict(model.named_parameters())
filtered = {k: v for k, v in hf_state.items()
            if k in model_sd and model_sd[k].shape == v.shape}
skipped = {k: v.shape for k, v in hf_state.items()
           if k not in filtered and k in model_sd}
print(f"Matching: {len(filtered)}, Shape-mismatch skipped: {len(skipped)}")
missing, unexpected = model.load_state_dict(filtered, strict=False)
model = model.float().eval()

# <bos><turn|>user\nHi<turn|>\n<turn|>model\n  (same as original script)
input_ids = torch.tensor([[2, 106, 2430, 106, 108, 106, 4176, 108]])
print(f"Input IDs: {input_ids.tolist()}")
with torch.no_grad():
    out = model(input_ids)

logits = out.logits[0, -1]
topk = torch.topk(logits, 5)
print(f"\nHF forward argmax: {logits.argmax().item()}  logit={logits.max().item():.4f}")
print(f"top-5: {list(zip(topk.indices.tolist(), [f'{v:.2f}' for v in topk.values.tolist()]))}")
