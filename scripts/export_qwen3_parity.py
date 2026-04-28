# Exports HuggingFace Qwen3 reference tensors for C++ --parity comparisons.

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (BASE_DIR.parent / "models" / "Qwen3-8B-FP16").resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "parity").resolve()

PROMPT = """<|im_start|>system
You are a concise assistant.
Answer with only the final answer.
Do not explain.
Do not repeat.
One short sentence only.
<|im_end|>
<|im_start|>user
The capital of France is
<|im_end|>
<|im_start|>assistant

"""

torch.set_grad_enabled(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=PROMPT)
    return parser.parse_args()


def save_tensor_txt(tensor: torch.Tensor, path: Path) -> None:
    values = tensor.detach().float().cpu().flatten()

    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{float(value):.10f}\n")

    print(f"saved: {path.name}")


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()

    if not model_path.exists():
        raise RuntimeError(f"Model path not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Load HF Qwen3 model ===")
    print("MODEL_PATH:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )
    model.eval()

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]

    print("input_ids:", input_ids[0].tolist())
    print("Input token count:", input_ids.size(1))

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("hidden_states is None")

    save_tensor_txt(hidden_states[0], output_dir / "embedding.txt")

    num_layers = len(hidden_states) - 1
    for i in range(num_layers):
        save_tensor_txt(hidden_states[i + 1], output_dir / f"layer_{i:02d}.txt")

    final_hidden = hidden_states[-1]
    final_norm_output = model.model.norm(final_hidden)
    save_tensor_txt(final_norm_output, output_dir / "final_norm_output.txt")

    last_layer_output = None

    def save_last_layer_output(module, inputs, output):
        nonlocal last_layer_output
        del module, inputs
        last_layer_output = output.detach()

    hook = model.model.layers[-1].register_forward_hook(save_last_layer_output)
    _ = model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )
    hook.remove()

    if last_layer_output is None:
        raise RuntimeError("Failed to capture last layer output")

    save_tensor_txt(last_layer_output, output_dir / "last_layer_output.txt")

    last_logits = logits[0, -1]
    save_tensor_txt(last_logits, output_dir / "final_logits.txt")

    argmax_token = torch.argmax(last_logits).item()
    topk_values, topk_indices = torch.topk(last_logits, k=5)

    print("argmax token:", argmax_token)
    print("top5 token ids:", topk_indices.tolist())
    print("top5 logits:", topk_values.tolist())
    print(f"Saved files: embedding, {num_layers} layers, final_norm, final_logits")


if __name__ == "__main__":
    main()
