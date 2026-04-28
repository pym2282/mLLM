import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "TinyLlama")
    sub = parser.add_subparsers(dest="mode", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("text")
    dec = sub.add_parser("decode")
    dec.add_argument("ids", nargs="+", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    if args.mode == "encode":
        token_ids = tokenizer.encode(
            args.text,
            add_special_tokens=True
        )

        print(
            " ".join(
                map(str, token_ids)
            )
        )
        return

    if args.mode == "decode":
        text = tokenizer.decode(
            args.ids,
            skip_special_tokens=True
        )

        print(text)


if __name__ == "__main__":
    main()
