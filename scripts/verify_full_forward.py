"""HF reference forward used as ground truth for C++ parity checks.

Run from project root:
    python scripts/verify_full_forward.py --model-path models/TinyLlama
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/TinyLlama"))
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=[15043, 6796, 263, 1243],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    ids = torch.tensor([args.ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(ids).logits

    last = logits[0, -1].float()
    print(f"logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}")
    print(f"logits[0,-1,:5]: {last[:5].tolist()}")
    print(f"last-token argmax token_id: {int(last.argmax().item())}")


if __name__ == "__main__":
    main()
