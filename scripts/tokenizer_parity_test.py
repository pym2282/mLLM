"""
Tokenizer parity harness.

Compares C++ BPE tokenizer output against HuggingFace reference
for a battery of test cases. Run from project root after building
Release:

    python scripts/tokenizer_parity_test.py

Exit code 0 = all pass, 1 = any failure.
"""

import subprocess
import sys
from pathlib import Path

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


def run_python_tokenizer(text: str) -> list[int]:
    result = subprocess.run(
        ["python", "scripts/tokenizer_helper.py", "encode", text],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tokenizer_helper.py failed:\n{result.stderr}"
        )
    raw = result.stdout.strip()
    if not raw:
        return []
    return list(map(int, raw.split()))


def find_exe() -> Path:
    build_dir = PROJECT_ROOT / "cmake-build-release"
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


def run_cpp_tokenizer(text: str, exe: Path) -> list[int]:
    build_dir = exe.parent
    # Send text via stdin (UTF-8) to avoid Windows argv code-page conversion.
    result = subprocess.run(
        [str(exe), "--tokenize"],
        input=text + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(build_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mLLM --tokenize failed:\n{result.stderr}"
        )
    raw = result.stdout.strip()
    if not raw:
        return []
    return list(map(int, raw.split()))


def main() -> None:
    try:
        exe = find_exe()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    passed = 0
    failed = 0

    for text in TEST_CASES:
        label = repr(text)[:50]
        try:
            py_ids  = run_python_tokenizer(text)
            cpp_ids = run_cpp_tokenizer(text, exe)

            if py_ids == cpp_ids:
                print(f"PASS  {label}")
                passed += 1
            else:
                print(f"FAIL  {label}")
                print(f"  py : {py_ids[:25]}"
                      f"{'…' if len(py_ids)  > 25 else ''}")
                print(f"  cpp: {cpp_ids[:25]}"
                      f"{'…' if len(cpp_ids) > 25 else ''}")
                for i, (p, c) in enumerate(zip(py_ids, cpp_ids)):
                    if p != c:
                        print(f"  first diff @ [{i}]: py={p} cpp={c}")
                        break
                else:
                    print(
                        f"  length: py={len(py_ids)} cpp={len(cpp_ids)}"
                    )
                failed += 1

        except Exception as e:
            print(f"ERROR {label}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(TEST_CASES)} cases")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
