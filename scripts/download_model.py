import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--save-path", type=Path, default=Path("./models/TinyLlama"))
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)

    print(f"Download complete: {args.save_path}")


if __name__ == "__main__":
    main()
