# MiniVLLM

C++ 기반의 mini inference engine 프로젝트입니다.

목표:
- LLM inference 흐름 이해
- tokenizer → token ids → logits → sampling 구조 학습
- ONNX Runtime 기반 실제 모델 추론 연결
- OpenAI-compatible local runtime으로 확장

---

## 프로젝트 구조

```text
mini-vllm/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── export_onnx.py
│
├── src/
│   └── main.cpp
│
├── include/
│
├── models/
│   └── model.onnx
│
├── build/
└── .venv/
```

---

## Python 환경 설정

### 1. venv 생성

```bash
python -m venv .venv
```

### 2. venv 활성화

### Windows

```bash
.venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## ONNX 모델 생성

```bash
python export_onnx.py
```

실행 후:

```text
models/model.onnx
```

파일이 생성됩니다.

---

## C++ 빌드

### build 폴더 생성

```bash
mkdir build
cd build
```

### CMake configure

```bash
cmake ..
```

### Build

```bash
cmake --build .
```

---

## 실행

```bash
./MiniVLLM.exe
```

또는 Visual Studio / VSCode / CLion에서 CMake target 실행

---

## 현재 진행 단계

- [x] mini tokenizer
- [x] CMake 프로젝트 구성
- [x] Python → ONNX export
- [ ] ONNX Runtime C++ inference
- [ ] KV Cache
- [ ] batch inference
- [ ] OpenAI-compatible API
- [ ] mini-vLLM 완성

---

## 핵심 목표

이 프로젝트의 핵심은

**모델을 사용하는 것**이 아니라
**모델이 실제로 어떻게 동작하는지 이해하는 것**입니다.

특히:

- tokenizer
- logits
- sampling
- KV cache
- GPU memory
- inference optimization

을 직접 구현하고 이해하는 것이 목표입니다.

