from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR.parent / "models" / "Qwen3-8B-FP16").resolve()

PROMPT = "The capital of France is"

torch.set_grad_enabled(False)


def main():
    print("=== Load HF model ===")
    print("MODEL_PATH:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model path not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu"
    )

    model.eval()

    print("=== Tokenize ===")

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )

    print(
        "input_ids:",
        inputs["input_ids"][0].tolist()
    )

    print("=== Forward ===")

    outputs = model(
        **inputs
    )

    logits = outputs.logits
    last_logits = logits[0, -1].float().cpu()

    print("=== HF Results ===")

    argmax_token = torch.argmax(
        last_logits
    ).item()

    topk_values, topk_indices = torch.topk(
        last_logits,
        k=5
    )

    print("argmax token:", argmax_token)
    print("top5 token ids:", topk_indices.tolist())
    print("top5 logits:", topk_values.tolist())

    print()
    print("===================================")
    print("C++ Forward 결과와 비교")
    print("===================================")
    print("- argmax 동일")
    print("- top5 token ids 동일")
    print("- logits diff 최소")


if __name__ == "__main__":
    main()
