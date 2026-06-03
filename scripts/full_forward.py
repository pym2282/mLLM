"""Full BOS-only forward pass in Python (float32, BF16 weights from cache)."""
import struct, torch

CACHE = "models/gemma-4-E2B-it-Q4_K_M.gguf.mlm"

def load_cache(path):
    with open(path, 'rb') as f:
        f.read(8); f.read(16)
        n = struct.unpack('<I', f.read(4))[0]
        c10 = {6: torch.float32, 5: torch.float16, 15: torch.bfloat16,
               3: torch.int32, 4: torch.int64, 0: torch.uint8, 1: torch.int8}
        sizes = {6:4, 5:2, 15:2, 3:4, 4:8, 0:1, 1:1}
        W = {}
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

print("Loading..."); W = load_cache(CACHE); print(f"{len(W)} tensors")

def gn(x, w, eps=1e-6):
    xf = x.float(); rms = torch.rsqrt(xf.pow(2).mean(-1,True)+eps)
    return (xf * rms * (1+w.float())).to(x.dtype)

ids = torch.tensor([[2]], dtype=torch.long)
hidden = W['model.embed_tokens.weight'][ids[0]].unsqueeze(0).float() * (1536**0.5)
print(f"Embedding norm: {hidden[0,0].norm().item():.4f}")

for i in range(35):
    p = f"model.layers.{i}"
    kd = W[f"{p}.self_attn.k_proj.weight"].size(0)
    qd = W[f"{p}.self_attn.q_proj.weight"].size(0)
    n_kv = 1; hd = kd; n_h = qd // hd  # handles global (hd=512, n_h=8) and local (hd=256, n_h=8)

    res = hidden.clone()
    h = gn(hidden, W[f"{p}.input_layernorm.weight"])
    q = torch.nn.functional.linear(h, W[f"{p}.self_attn.q_proj.weight"]).view(1,1,n_h,hd).transpose(1,2)
    k = torch.nn.functional.linear(h, W[f"{p}.self_attn.k_proj.weight"]).view(1,1,n_kv,hd).transpose(1,2)
    v = torch.nn.functional.linear(h, W[f"{p}.self_attn.v_proj.weight"]).view(1,1,n_kv,hd).transpose(1,2)
    q = gn(q, W[f"{p}.self_attn.q_norm.weight"])
    k = gn(k, W[f"{p}.self_attn.k_norm.weight"])
    reps = n_h // n_kv
    k = k.repeat_interleave(reps, 1); v = v.repeat_interleave(reps, 1)
    ao = torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=1.0/(256**0.5),is_causal=True)
    h = ao.transpose(1,2).contiguous().view(1,1,n_h*hd)
    h = torch.nn.functional.linear(h, W[f"{p}.self_attn.o_proj.weight"])
    h = gn(h, W[f"{p}.post_attn_norm.weight"])
    hidden = res + h

    res = hidden.clone()
    h = gn(hidden, W[f"{p}.post_attention_layernorm.weight"])
    gate = torch.nn.functional.gelu(torch.nn.functional.linear(h, W[f"{p}.mlp.gate_proj.weight"]), approximate='tanh')
    up = torch.nn.functional.linear(h, W[f"{p}.mlp.up_proj.weight"])
    h = torch.nn.functional.linear(gate*up, W[f"{p}.mlp.down_proj.weight"])
    h = gn(h, W[f"{p}.post_ffn_norm.weight"])
    hidden = res + h

hidden = gn(hidden, W['model.norm.weight'])
logits = torch.nn.functional.linear(hidden[0,-1], W['model.embed_tokens.weight'])
cap = 30.0
logits_raw = logits.clone()
logits = torch.tanh(logits/cap)*cap

print(f"\nPython (float32) BOS-only:")
print(f"  pre-softcap: top1={logits_raw.argmax().item()} val={logits_raw.max().item():.2f}")
v,idx = logits_raw.topk(5)
print(f"  top5-raw: {list(zip(idx.tolist(), [f'{x:.1f}' for x in v.tolist()]))}")
print(f"  post-softcap: top1={logits.argmax().item()} val={logits.max().item():.4f}")
