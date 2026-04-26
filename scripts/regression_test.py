import os
import subprocess
import re
import sys
from pathlib import Path


def run_cpp():
    # LibTorch DLL 경로 등록
    os.environ["PATH"] = (
        r"C:\libtorch\lib;"
        + os.environ["PATH"]
    )

    # regression_test.py 기준으로 프로젝트 루트 계산
    # scripts/regression_test.py -> parent.parent = mLLM/
    project_root = Path(__file__).resolve().parent.parent
    build_dir = project_root / "cmake-build-debug"

    exe_path = build_dir / "mLLM.exe"

    if not exe_path.exists():
        raise RuntimeError(
            f"Executable not found: {exe_path}"
        )

    # 반드시 build 폴더에서 실행해야
    # ../models/TinyLlama 경로가 정상 동작함
    result = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        cwd=str(build_dir)
    )

    if result.returncode != 0:
        print("===== C++ STDOUT =====")
        print(result.stdout)

        print("===== C++ STDERR =====")
        print(result.stderr)

        raise RuntimeError(
            f"C++ runtime execution failed "
            f"(code={result.returncode})"
        )

    return result.stdout


def run_python_reference():
    project_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        ["python", "scripts/verify_full_forward.py"],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )

    if result.returncode != 0:
        print("===== PYTHON STDOUT =====")
        print(result.stdout)

        print("===== PYTHON STDERR =====")
        print(result.stderr)

        raise RuntimeError(
            f"Python reference execution failed "
            f"(code={result.returncode})"
        )

    return result.stdout


def parse_argmax(output):
    match = re.search(
        r"argmax token_id[:=]\s*(\d+)",
        output
    )

    if not match:
        raise RuntimeError(
            "Failed to parse argmax token_id"
        )

    return int(match.group(1))


def main():
    print("Running C++ runtime...")
    cpp_output = run_cpp()

    print("Running Python HF reference...")
    py_output = run_python_reference()

    cpp_argmax = parse_argmax(cpp_output)
    py_argmax = parse_argmax(py_output)

    print(f"C++ argmax: {cpp_argmax}")
    print(f"Python argmax: {py_argmax}")

    if cpp_argmax != py_argmax:
        print("\nFAIL: argmax mismatch")
        sys.exit(1)

    print("\nPASS: forward parity verified")


if __name__ == "__main__":
    main()
