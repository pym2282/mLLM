"""
mLLM regression test suite.

Usage:
    python scripts/regression_test.py                       # all suites (Qwen path auto-detected)
    python scripts/regression_test.py --suite gemma-forward
    python scripts/regression_test.py --suite qwen-forward --model-path models/Qwen3-8B-FP16
    python scripts/regression_test.py --suite all           # requires both models

Suites:
    tokenizer      - Qwen tokenizer round-trip parity
    qwen-forward   - Qwen3 last-token argmax vs reference
    gemma-forward  - Gemma 4 last-token argmax = 4176 (HF BF16 reference)
    all            - tokenizer + qwen-forward + gemma-forward
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QWEN_PATH  = PROJECT_ROOT / "models" / "Qwen3-8B-FP16"
DEFAULT_GEMMA_PATH = PROJECT_ROOT / "models" / "gemma-4-E2B-it-Q4_K_M.gguf"

# Gemma 4 parity: token 4176 ("vi") matches HF BF16 forward on the 8-token
# chat-template prompt [2, 106, 2430, 106, 108, 106, 4176, 108].
GEMMA_EXPECTED_ARGMAX = 4176


def find_exe(build_dir: Path, exe: Path | None) -> Path:
    if exe is not None:
        if exe.exists():
            return exe.resolve()
        raise RuntimeError(f"Executable not found: {exe}")
    for candidate in [build_dir / "Release" / "mLLM.exe", build_dir / "mLLM.exe"]:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        f"Executable not found under {build_dir}\n"
        "Build with: cmake --build cmake-build-release"
    )


def run_checked(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=str(cwd))
    if result.returncode != 0:
        print("===== STDOUT =====")
        print(result.stdout)
        print("===== STDERR =====")
        print(result.stderr)
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def parse_argmax(output: str) -> int:
    m = re.search(r"argmax token_id[:=]\s*(\d+)", output)
    if not m:
        raise RuntimeError("Failed to parse argmax token_id from output")
    return int(m.group(1))


def setup_path(args: argparse.Namespace) -> None:
    if args.libtorch_dir is not None:
        os.environ["PATH"] = str(args.libtorch_dir) + os.pathsep + os.environ["PATH"]


def run_tokenizer_suite(args: argparse.Namespace, exe: Path, model_path: Path) -> None:
    print("Running tokenizer parity...")
    result = run_checked(
        [args.python, "scripts/tokenizer_parity_test.py",
         "--exe", str(exe), "--model-path", str(model_path)],
        PROJECT_ROOT,
    )
    print(result.stdout.strip())
    print("  PASS: tokenizer")


def run_qwen_forward_suite(args: argparse.Namespace, exe: Path, model_path: Path) -> None:
    print(f"Running Qwen forward parity ({model_path.name})...")
    setup_path(args)
    result = run_checked(
        [str(exe), str(model_path), "--parity", "--parity-dir", str(args.parity_dir.resolve())],
        exe.parent,
    )
    argmax = parse_argmax(result.stdout + result.stderr)
    print(f"  PASS: Qwen argmax={argmax}")


def run_gemma_forward_suite(args: argparse.Namespace, exe: Path, model_path: Path) -> None:
    print(f"Running Gemma 4 forward parity ({model_path.name})...")
    setup_path(args)
    result = run_checked([str(exe), str(model_path), "--parity"], exe.parent)
    argmax = parse_argmax(result.stdout + result.stderr)
    if argmax != GEMMA_EXPECTED_ARGMAX:
        raise RuntimeError(
            f"Gemma argmax mismatch: got {argmax}, expected {GEMMA_EXPECTED_ARGMAX}"
        )
    print(f"  PASS: Gemma 4 argmax={argmax} (matches HF BF16)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mLLM regression suite")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH,
                        help="Qwen model directory")
    parser.add_argument("--gemma-path", type=Path, default=DEFAULT_GEMMA_PATH,
                        help="Gemma 4 GGUF file")
    parser.add_argument("--build-dir", type=Path,
                        default=PROJECT_ROOT / "cmake-build-release")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--libtorch-dir", type=Path, default=Path(r"C:\libtorch-cuda\lib"))
    parser.add_argument("--parity-dir", type=Path,
                        default=PROJECT_ROOT / "scripts" / "parity")
    parser.add_argument(
        "--suite",
        choices=["all", "tokenizer", "qwen-forward", "gemma-forward"],
        default="gemma-forward",
        help="Test suite to run (default: gemma-forward)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exe = find_exe(args.build_dir, args.exe)

    if args.suite in ("all", "tokenizer", "qwen-forward"):
        qwen_path = args.model_path.resolve()
        if not qwen_path.exists():
            print(f"SKIP: Qwen model not found at {qwen_path}")
        else:
            if args.suite in ("all", "tokenizer"):
                run_tokenizer_suite(args, exe, qwen_path)
            if args.suite in ("all", "qwen-forward"):
                run_qwen_forward_suite(args, exe, qwen_path)

    if args.suite in ("all", "gemma-forward"):
        gemma_path = args.gemma_path.resolve()
        if not gemma_path.exists():
            if args.suite == "all":
                print(f"SKIP: Gemma model not found at {gemma_path}")
            else:
                raise RuntimeError(f"Gemma model not found: {gemma_path}")
        else:
            run_gemma_forward_suite(args, exe, gemma_path)

    print("\nPASS: all suites completed")


if __name__ == "__main__":
    main()
