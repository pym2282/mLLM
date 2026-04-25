"""HF TinyLlama reference forward — used as ground truth for C++ LlamaRunner.

Run from project root:
    python scripts/verify_full_forward.py

Compares against C++ logs:
    [LlamaRunner] logits[0,-1,:5]: ...
    [LlamaRunner] last-token argmax token_id=...

Tolerance: ~5e-2 absolute on logits in BF16. argmax must match exactly.
"""

import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "models/TinyLlama"

m = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
m.eval()

ids = torch.tensor([[15043, 6796, 263, 1243]], dtype=torch.long)

with torch.no_grad():
    logits = m(ids).logits

last = logits[0, -1].float()
print(f"logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}")
print(f"logits[0,-1,:5]: {last[:5].tolist()}")
print(f"last-token argmax token_id: {int(last.argmax().item())}")
