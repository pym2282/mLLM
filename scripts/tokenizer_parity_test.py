"""
Tokenizer parity harness.

Compares C++ BPE tokenizer output against HuggingFace reference
for a battery of test cases. Run from project root after building
Release:

    python scripts/tokenizer_parity_test.py

Exit code 0 = all pass, 1 = any failure.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

TEST_CASES = [
    # basic
    "hello",
    "Hello",
    "hello world",
    " hello",               # leading space → prepend_scheme edge case
    "Hello World",

    # punctuation / digits
    "This is a test.",
    "1234567890",
    "!@#$%^&*()",
    "foo, bar; baz.",

    # longer
    "The quick brown fox jumps over the lazy dog",

    # non-ASCII → byte fallback
    "안녕하세요",            # Korean
    "こんにちは",            # Japanese
    "hello 안녕",           # mixed ASCII + Korean

    # edge
    "",
    " ",
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_python_tokenizer_batch(texts: list[str], model_path: Path) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    return [
        tokenizer.encode(text, add_special_tokens=True)
        for text in texts
    ]


def find_exe(build_dir: Path, exe: Path | None) -> Path:
    if exe is not None:
        if exe.exists():
            return exe.resolve()
        raise RuntimeError(f"mLLM.exe not found: {exe}")

    for candidate in [
        build_dir / "Release" / "mLLM.exe",
        build_dir / "mLLM.exe",
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"mLLM.exe not found under {build_dir}.\n"
        "Build with: cmake --build cmake-build-release --config Release"
    )


def run_cpp_tokenizer_batch(texts: list[str], exe: Path, model_path: Path) -> list[list[int]]:
    build_dir = exe.parent
    # Send text via stdin (UTF-8) to avoid Windows argv code-page conversion.
    result = subprocess.run(
        [str(exe), str(model_path), "--tokenize-batch"],
        input="\n".join(texts) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(build_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mLLM --tokenize-batch failed:\n{result.stderr}"
        )

    lines = result.stdout.splitlines()
    if len(lines) != len(texts):
        raise RuntimeError(
            f"mLLM returned {len(lines)} lines for {len(texts)} inputs.\n"
            f"stderr:\n{result.stderr}"
        )

    return [
        [] if not line.strip() else list(map(int, line.split()))
        for line in lines
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "TinyLlama")
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "cmake-build-release")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    try:
        exe = find_exe(args.build_dir, args.exe)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    passed = 0
    failed = 0

    try:
        py_batch = run_python_tokenizer_batch(TEST_CASES, model_path)
        cpp_batch = run_cpp_tokenizer_batch(TEST_CASES, exe, model_path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for text, py_ids, cpp_ids in zip(TEST_CASES, py_batch, cpp_batch):
        label = repr(text)[:50]
        if py_ids == cpp_ids:
            print(f"PASS  {label}")
            passed += 1
        else:
            print(f"FAIL  {label}")
            print(f"  py : {py_ids[:25]}"
                  f"{'...' if len(py_ids)  > 25 else ''}")
            print(f"  cpp: {cpp_ids[:25]}"
                  f"{'...' if len(cpp_ids) > 25 else ''}")
            for i, (p, c) in enumerate(zip(py_ids, cpp_ids)):
                if p != c:
                    print(f"  first diff @ [{i}]: py={p} cpp={c}")
                    break
            else:
                print(
                    f"  length: py={len(py_ids)} cpp={len(cpp_ids)}"
                )
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(TEST_CASES)} cases")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
