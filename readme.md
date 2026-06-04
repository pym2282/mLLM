# mLLM — 로컬 AI 채팅 서버

Windows PC에서 AI 모델을 직접 실행하는 C++ LLM 런타임입니다.
인터넷 없이, 월정액 없이 로컬에서 AI와 대화하거나 API 서버를 띄울 수 있습니다.

---

## 5분 빠른 시작

### 0. 준비물 확인

| 항목 | 필수 | 확인 방법 |
|------|------|----------|
| Windows 10/11 | ✅ | |
| NVIDIA GPU (8GB+ VRAM 권장) | 권장 | 작업 관리자 → 성능 탭 |
| [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) | GPU 사용 시 | `nvcc --version` |
| [LibTorch (CUDA)](https://pytorch.org/get-started/locally/) | ✅ | 아래 설명 참고 |
| [Visual Studio 2022](https://visualstudio.microsoft.com/ko/downloads/) (C++ 워크로드) | ✅ | |
| [CMake 3.20+](https://cmake.org/download/) | ✅ | `cmake --version` |
| [Python 3.10+](https://www.python.org/downloads/) | ✅ (모델 다운로드) | `python --version` |

### 1. LibTorch 설치

1. [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) 접속
2. 아래 설정 선택:
   - PyTorch Build: **Stable**
   - OS: **Windows**
   - Package: **LibTorch**
   - Language: **C++/Java**
   - Compute Platform: **CUDA 12.x** (GPU) 또는 **CPU**
3. 다운로드 링크에서 **Release 버전** zip 다운로드
4. `C:\libtorch-cuda\` 에 압축 해제 (CUDA) 또는 `C:\libtorch\` (CPU)

### 2. 환경 설정

```bat
setup.bat
```

이 스크립트가 LibTorch, CUDA, GPU 아키텍처를 자동으로 감지하고 설정합니다.

### 3. 빌드

```bat
build_mllm.bat
```

첫 빌드는 5~10분 정도 걸립니다.

### 4. 모델 다운로드

```bat
pip install huggingface_hub
python scripts\download_gguf.py
```

> **추천 모델**: `Qwen3.5-9B-Q4_K_M.gguf` — VRAM 8GB, 한국어/영어 우수

### 5. 실행

**채팅 모드:**
```bat
chat.bat
```

**API 서버 모드:**
```bat
serve.bat
```

---

## 모델 선택 가이드

| 모델 | 파일 크기 | VRAM | 품질 | 속도 | 특징 |
|------|-----------|------|------|------|------|
| **Qwen3.5-9B Q4_K_M** ⭐ | 5.7 GB | 8 GB | ★★★★ | ★★★ | 권장, 한국어 우수 |
| Gemma 4 E2B Q4_K_M | 2.5 GB | 4 GB | ★★★ | ★★★★★ | 빠름, 소형 |
| Qwen3-8B FP16 | 16 GB | 16 GB | ★★★★★ | ★★ | 최고 품질, VRAM 많이 필요 |

---

## 실행 방법

### 채팅 모드

```bat
chat.bat
chat.bat models\Qwen3.5-9B-Q4_K_M.gguf
```

터미널에서 직접 AI와 대화합니다. `q` 입력 후 엔터로 종료.

### API 서버 모드

```bat
serve.bat
serve.bat models\Qwen3.5-9B-Q4_K_M.gguf 8080
```

서버 시작 후 `http://localhost:8080` 에서 OpenAI 호환 API를 사용할 수 있습니다.

**curl 예시:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"안녕하세요!\"}]}"
```

**Python 예시:**
```python
import requests
res = requests.post("http://localhost:8080/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "파이썬으로 hello world 짜줘"}]
})
print(res.json()["choices"][0]["message"]["content"])
```

### CLI 직접 실행

```bat
cmake-build-release\mLLM.exe --help
cmake-build-release\mLLM.exe models\Qwen3.5-9B-Q4_K_M.gguf
cmake-build-release\mLLM.exe models\Qwen3.5-9B-Q4_K_M.gguf --serve --port 8080
cmake-build-release\mLLM.exe models\Qwen3.5-9B-Q4_K_M.gguf --thinking
```

---

## API 레퍼런스

### POST /v1/chat/completions (권장)

OpenAI 호환 엔드포인트. ChatGPT API와 동일한 형식입니다.

```json
{
  "messages": [
    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
    {"role": "user",   "content": "질문 내용"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512
}
```

응답:
```json
{
  "choices": [{"message": {"role": "assistant", "content": "응답 내용"}}],
  "usage": {"prompt_tokens": 20, "completion_tokens": 50}
}
```

스트리밍: `"stream": true` 설정 시 Server-Sent Events(SSE)로 토큰을 실시간 수신.

### GET /health

서버 상태 확인. `200 OK` 반환.

---

## 문제 해결

### "DLL을 찾을 수 없습니다" 오류
→ `chat.bat` 또는 `serve.bat` 사용 (PATH 자동 설정). 직접 실행 시 PATH에 `cmake-build-release\` 추가.

### "CUDA out of memory" 오류
→ 더 작은 모델 사용 (Q4_K_M GGUF 권장), 다른 GPU 사용 프로그램 종료.

### 모델 파일을 찾을 수 없음
→ 모델이 `models\` 폴더에 있는지 확인. `chat.bat models\모델파일명.gguf` 처럼 직접 경로 지정.

### 빌드 실패 — "LibTorch를 찾을 수 없습니다"
→ `setup.bat` 재실행. 또는 `local.cmake` 파일에서 `Torch_DIR` 경로 직접 수정.

### 빌드 실패 — "Visual Studio를 찾을 수 없습니다"
→ VS 2022 설치 확인. "C++ 데스크톱 개발" 워크로드가 선택되어 있어야 합니다.

---

## 기술 문서

개발자용 아키텍처 설명: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 지원 모델

| 계열 | 모델 | 포맷 |
|------|------|------|
| Qwen | Qwen3, Qwen3.5 (8B~) | GGUF, FP16, FP8 |
| Gemma | Gemma 4 | GGUF, SafeTensors |
| Llama | TinyLlama, Llama 계열 | SafeTensors |
