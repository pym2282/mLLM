"""
GGUF 모델 다운로드 스크립트.

사용법:
    python scripts\download_gguf.py                   # 권장 모델 목록 표시
    python scripts\download_gguf.py --model qwen3.5   # Qwen3.5 다운로드
    python scripts\download_gguf.py --url <URL>       # 직접 URL 지정

권장 모델:
    qwen3.5   Qwen3.5-9B-Q4_K_M  (5.7GB, VRAM 8GB, 한국어 우수)
    gemma4    Gemma-4-E2B-Q4_K_M  (2.5GB, VRAM 4GB, 소형 고속)
"""
import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

RECOMMENDED = {
    "qwen3.5": {
        "repo":     "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "desc":     "Qwen3.5-9B Q4_K_M — VRAM 8GB, 한국어/영어 우수 (권장)",
        "vram_gb":  8,
    },
    "gemma4": {
        "repo":     "bartowski/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "desc":     "Gemma 4 E2B Q4_K_M — VRAM 4GB, 소형 고속",
        "vram_gb":  4,
    },
}


def print_menu() -> None:
    print("\n사용 가능한 권장 모델:")
    print("-" * 60)
    for key, m in RECOMMENDED.items():
        print(f"  --model {key:<10}  {m['desc']}")
    print("-" * 60)
    print("\n사용법: python scripts\\download_gguf.py --model qwen3.5")
    print()


def download(repo: str, filename: str, dest_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub 가 설치되지 않았습니다.")
        print("설치: pip install huggingface_hub")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"이미 존재합니다: {dest_path}")
        return

    print(f"다운로드 중: {repo}/{filename}")
    print(f"저장 위치: {dest_path}")
    print("(수백 MB~수 GB, 네트워크 속도에 따라 시간이 걸립니다...)\n")

    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    print(f"\n완료! 모델 위치: {dest_path}")
    print(f"\n실행: chat.bat {dest_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GGUF 모델 다운로드")
    parser.add_argument("--model",    type=str, help="권장 모델 키 (qwen3.5, gemma4)")
    parser.add_argument("--repo",     type=str, help="HuggingFace repo ID (직접 지정)")
    parser.add_argument("--filename", type=str, help="다운로드할 파일명 (--repo와 함께 사용)")
    parser.add_argument("--dest",     type=Path, default=MODELS_DIR, help="저장 폴더")
    args = parser.parse_args()

    if not args.model and not args.repo:
        print_menu()
        return

    if args.model:
        if args.model not in RECOMMENDED:
            print(f"알 수 없는 모델: {args.model}")
            print_menu()
            sys.exit(1)
        m = RECOMMENDED[args.model]
        download(m["repo"], m["filename"], args.dest)
    else:
        if not args.filename:
            print("--repo 사용 시 --filename 도 지정하세요.")
            sys.exit(1)
        download(args.repo, args.filename, args.dest)


if __name__ == "__main__":
    main()
