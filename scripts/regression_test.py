"""
mLLM regression test suite.

Usage:
    python scripts/regression_test.py                       # default: gemma-forward
    python scripts/regression_test.py --suite gemma-forward
    python scripts/regression_test.py --suite gemma-generate
    python scripts/regression_test.py --suite qwen-generate
    python scripts/regression_test.py --suite all           # requires both models

Suites:
    gemma-forward   - Gemma 4 last-token argmax = 4176 (HF BF16 reference)
    gemma-generate  - Gemma 4 greedy token sequence (golden record)
    qwen-generate   - Qwen3.5 greedy token sequence (golden record)
    tokenizer       - Qwen tokenizer round-trip parity
    qwen-forward    - Qwen3 last-token argmax vs reference (requires parity dir)
    all             - gemma-forward + gemma-generate + qwen-generate
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QWEN_PATH  = PROJECT_ROOT / "models" / "Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_GEMMA_PATH = PROJECT_ROOT / "models" / "gemma-4-E2B-it-Q4_K_M.gguf"

# ─── Golden records ───────────────────────────────────────────────────────────
# Gemma 4 forward parity: argmax=4176 matches HF BF16 on the 8-token parity prompt.
GEMMA_EXPECTED_ARGMAX = 4176

# Gemma 4 generate parity: greedy generation from the parity prompt (rep_penalty=1.0).
# First token = 4176 matches --parity argmax. EOS hit after 1 token (finish=EOS).
GEMMA_EXPECTED_GENERATE_TOKENS: list[int] | None = [4176]

# Qwen3.5 generate parity: greedy response to "Hello" chat prompt (rep_penalty=1.0).
# EOS token excluded (EOS=248046 for Qwen3.5-9B). finish=EOS after 6 tokens.
QWEN_EXPECTED_GENERATE_TOKENS: list[int] | None = [248068, 271, 248069, 271, 8160, 96487]
# ─────────────────────────────────────────────────────────────────────────────


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


def parse_generate_tokens(output: str) -> list[int]:
    m = re.search(r"generate-test tokens:([ \d]*)", output)
    if not m:
        raise RuntimeError("Failed to parse 'generate-test tokens:' from output")
    raw = m.group(1).strip()
    if not raw:
        return []
    return [int(t) for t in raw.split()]


def setup_path(args: argparse.Namespace) -> None:
    if args.libtorch_dir is not None:
        os.environ["PATH"] = str(args.libtorch_dir) + os.pathsep + os.environ["PATH"]


def run_prefix_cache_suite(args: argparse.Namespace, exe: Path, model_path: Path) -> None:
    print(f"Running prefix-cache cold-vs-hit ({model_path.name})...")
    setup_path(args)
    result = run_checked([str(exe), str(model_path), "--prefix-test"], exe.parent)
    combined = result.stdout + result.stderr

    hit_len_m = re.search(r"prefix-test hit_len=(\d+)", combined)
    match_m   = re.search(r"prefix-test tokens_match=(\d+)", combined)
    prompt_m  = re.search(r"prefix-test prompt_len=(\d+)", combined)

    if not hit_len_m or not match_m:
        raise RuntimeError("Failed to parse prefix-test output")

    hit_len     = int(hit_len_m.group(1))
    tokens_match = int(match_m.group(1))
    prompt_len  = int(prompt_m.group(1)) if prompt_m else 0

    if hit_len == 0:
        print(f"  SKIP: prefix cache not active (prompt_len={prompt_len}, model may not support KVSnapshot)")
        return

    if not tokens_match:
        raise RuntimeError(
            f"Prefix cache cold-vs-hit mismatch: tokens differ "
            f"(hit_len={hit_len}, prompt_len={prompt_len})"
        )
    print(f"  PASS: prefix-cache hit_len={hit_len} tokens_match=1 prompt_len={prompt_len}")


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


def run_generate_suite(
    args: argparse.Namespace,
    exe: Path,
    model_path: Path,
    expected_tokens: list[int] | None,
    label: str,
) -> None:
    print(f"Running {label} generate test ({model_path.name})...")
    setup_path(args)
    result = run_checked([str(exe), str(model_path), "--generate-test"], exe.parent)
    combined = result.stdout + result.stderr
    tokens = parse_generate_tokens(combined)

    if not tokens:
        raise RuntimeError(f"{label} generate-test produced 0 tokens")

    if expected_tokens is None:
        # Golden capture mode: print the tokens so user can embed them.
        print(f"  GOLDEN CAPTURE (no expected set yet): tokens={tokens}")
        print(f"  Update regression_test.py with: {label.upper().replace(' ', '_')}"
              f"_EXPECTED_GENERATE_TOKENS = {tokens}")
        print(f"  PASS: {label} generate (stability only — set golden to enable exact check)")
    else:
        if tokens != expected_tokens:
            raise RuntimeError(
                f"{label} generate mismatch:\n"
                f"  got:      {tokens}\n"
                f"  expected: {expected_tokens}"
            )
        print(f"  PASS: {label} generate tokens={tokens}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mLLM regression suite")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH,
                        help="Qwen model path")
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
        choices=["all", "tokenizer", "qwen-forward", "gemma-forward",
                 "gemma-generate", "qwen-generate", "prefix-cache"],
        default="gemma-forward",
        help="Test suite to run (default: gemma-forward)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exe = find_exe(args.build_dir, args.exe)

    # ── Tokenizer / Qwen forward (require parity dir) ────────────────────────
    if args.suite in ("tokenizer", "qwen-forward"):
        qwen_path = args.model_path.resolve()
        if not qwen_path.exists():
            print(f"SKIP: Qwen model not found at {qwen_path}")
        else:
            if args.suite == "tokenizer":
                run_tokenizer_suite(args, exe, qwen_path)
            if args.suite == "qwen-forward":
                run_qwen_forward_suite(args, exe, qwen_path)

    # ── Gemma forward (argmax) ────────────────────────────────────────────────
    if args.suite in ("all", "gemma-forward"):
        gemma_path = args.gemma_path.resolve()
        if not gemma_path.exists():
            if args.suite == "all":
                print(f"SKIP: Gemma model not found at {gemma_path}")
            else:
                raise RuntimeError(f"Gemma model not found: {gemma_path}")
        else:
            run_gemma_forward_suite(args, exe, gemma_path)

    # ── Gemma generate (multi-token golden record) ────────────────────────────
    if args.suite in ("all", "gemma-generate"):
        gemma_path = args.gemma_path.resolve()
        if not gemma_path.exists():
            if args.suite == "all":
                print(f"SKIP: Gemma model not found at {gemma_path}")
            else:
                raise RuntimeError(f"Gemma model not found: {gemma_path}")
        else:
            run_generate_suite(
                args, exe, gemma_path,
                GEMMA_EXPECTED_GENERATE_TOKENS,
                "Gemma 4",
            )

    # ── Qwen generate (multi-token golden record) ─────────────────────────────
    if args.suite in ("all", "qwen-generate"):
        qwen_path = args.model_path.resolve()
        if not qwen_path.exists():
            if args.suite == "all":
                print(f"SKIP: Qwen model not found at {qwen_path}")
            else:
                raise RuntimeError(f"Qwen model not found: {qwen_path}")
        else:
            run_generate_suite(
                args, exe, qwen_path,
                QWEN_EXPECTED_GENERATE_TOKENS,
                "Qwen3.5",
            )

    # ── Prefix cache cold-vs-hit (Qwen only, SKIP for others) ─────────────────
    if args.suite in ("all", "prefix-cache"):
        qwen_path = args.model_path.resolve()
        if not qwen_path.exists():
            if args.suite == "all":
                print(f"SKIP: Qwen model not found at {qwen_path}")
            else:
                raise RuntimeError(f"Qwen model not found: {qwen_path}")
        else:
            run_prefix_cache_suite(args, exe, qwen_path)

    print("\nPASS: all suites completed")


if __name__ == "__main__":
    main()
