from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

"""
export_onnx.py
----------------
목표:
- Hugging Face 모델을 ONNX로 변환
- C++ ONNX Runtime에서 사용할 model.onnx 생성

실행 전:
pip install torch transformers onnx

추천 시작 모델:
Qwen/Qwen2.5-1.5B-Instruct
또는
TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_PATH = "./models/model.onnx"
MAX_LENGTH = 32


def main():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model.eval()

    sample_text = "1 + 1 ="
    print(f"Sample input: {sample_text}")

    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("Exporting ONNX...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    print("Done!")
    print(f"ONNX saved to: {OUTPUT_PATH}")
    print("Next step: load this file from C++ with ONNX Runtime")


if __name__ == "__main__":
    main()
