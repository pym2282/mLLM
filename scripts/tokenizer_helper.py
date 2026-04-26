import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

def main():
    if len(sys.argv) < 3:
        print("usage:")
        print('python tokenizer_helper.py encode "hello world"')
        print("python tokenizer_helper.py decode 1 22172")
        return

    mode = sys.argv[1]

    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / "models" / "TinyLlama"

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        local_files_only=True
    )

    if mode == "encode":
        text = sys.argv[2]

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True
        )

        print(
            " ".join(
                map(str, token_ids)
            )
        )
        return

    if mode == "decode":
        token_ids = list(
            map(int, sys.argv[2:])
        )

        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )

        print(text)
        return

    print("unknown mode:", mode)


if __name__ == "__main__":
    main()
