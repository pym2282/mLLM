import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QWEN_PATH = PROJECT_ROOT / "models" / "Qwen3-8B-FP16"


def find_exe(build_dir: Path, exe: Path | None) -> Path:
    if exe is not None:
        if exe.exists():
            return exe.resolve()
        raise RuntimeError(f"Executable not found: {exe}")

    for candidate in [
        build_dir / "Release" / "mLLM.exe",
        build_dir / "mLLM.exe",
    ]:
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(
        f"Executable not found under {build_dir}\n"
        f"Build with: cmake --build {build_dir} --config Release"
    )


def run_checked(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )

    if result.returncode != 0:
        print("===== STDOUT =====")
        print(result.stdout)
        print("===== STDERR =====")
        print(result.stderr)
        raise RuntimeError(
            f"Command failed with code {result.returncode}: {' '.join(cmd)}"
        )

    return result


def run_tokenizer_suite(args: argparse.Namespace, exe: Path, model_path: Path) -> None:
    print("Running tokenizer parity...")
    result = run_checked(
        [
            args.python,
            "scripts/tokenizer_parity_test.py",
            "--exe",
            str(exe),
            "--model-path",
            str(model_path),
        ],
        PROJECT_ROOT,
    )
    print(result.stdout)


def run_qwen_forward_suite(args: argparse.Namespace, exe: Path, model_path: Path) -> None:
    print("Running Qwen forward parity...")

    if args.libtorch_dir is not None:
        os.environ["PATH"] = str(args.libtorch_dir) + os.pathsep + os.environ["PATH"]

    result = run_checked(
        [
            str(exe),
            str(model_path),
            "--parity",
            "--parity-dir",
            str(args.parity_dir.resolve()),
        ],
        exe.parent,
    )

    argmax = parse_argmax(result.stdout)
    print(result.stdout)
    print(f"Qwen C++ argmax: {argmax}")


def parse_argmax(output: str) -> int:
    match = re.search(
        r"argmax token_id[:=]\s*(\d+)",
        output,
    )

    if not match:
        raise RuntimeError("Failed to parse argmax token_id")

    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_QWEN_PATH,
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "cmake-build-release",
    )
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--libtorch-dir",
        type=Path,
        default=Path(r"C:\libtorch\lib"),
    )
    parser.add_argument(
        "--parity-dir",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "parity",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "tokenizer", "qwen-forward"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exe = find_exe(args.build_dir, args.exe)
    model_path = args.model_path.resolve()

    if args.suite in ("all", "tokenizer"):
        run_tokenizer_suite(args, exe, model_path)

    if args.suite in ("all", "qwen-forward"):
        run_qwen_forward_suite(args, exe, model_path)

    print("PASS: regression suite completed")


if __name__ == "__main__":
    main()
