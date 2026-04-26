# mLLM

미니 vLLM 스타일의 C++ LLM 추론 런타임.

HuggingFace의 `config.json` + `model.safetensors`를
**ONNX / TorchScript 없이 직접 로드**하여
forward / generation / KV cache decode를 수행합니다.

이 프로젝트의 목표는 단순 inference demo가 아니라,

# "mini-vLLM / mini-llama.cpp 수준의 standalone runtime"

입니다.

즉:

* safetensors 직접 로드
* full transformer forward
* sampler
* KV cache
* prefill / decode split
* tokenizer abstraction
* native tokenizer
* continuous batching
* scheduler
* OpenAI-compatible serving

까지를 단계적으로 구현합니다.

---

# 현재 프로젝트 상태 (중요)

## 현재는 어디까지 왔는가

현재는

# Core Runtime은 거의 완료

상태입니다.

즉:

* full forward parity 완료
* generation loop 완료
* KV cache decode 완료
* prefill / decode split 완료
* tokenizer abstraction 완료

까지 끝났습니다.

이제 남은 것은

# tokenizer parity + serving layer

입니다.

즉 프로젝트는

# "LLM 구현 중"

이 아니라

# "mini-vLLM finishing stage"

입니다.

---

# 기술 스택

## Core

* C++17
* CMake
* LibTorch
* nlohmann/json

## Platform

* Windows
* MSVC
* CLion

## Reference Model

### TinyLlama

* 22 layers
* hidden size: 2048
* attention heads: 32
* KV heads: 4 (GQA)
* BF16

## Expansion Target

향후 지원 예정:

* Qwen
* Mistral
* Gemma

---

# 중요한 설계 원칙 (매우 중요)

## 1. Stateless Runtime

위치:

```text
src/core/runtime/*
```

여기는

# 상태 없는 pure function

만 존재합니다.

예:

* Attention
* RMSNorm
* Sampler
* MLP
* RoPE

모델별 구현과 완전히 분리합니다.

---

## 2. Model-specific Logic 분리

위치:

```text
src/models/llama/
```

여기는

# TinyLlama 전용 구현

만 존재합니다.

즉:

* weight loading
* layer registry
* generation loop
* model-specific config

향후

* QwenRunner
* MistralRunner

추가 시 runtime 재사용이 가능해야 합니다.

---

## 3. Tokenizer Abstraction

위치:

```text
src/tokenizer/
```

구조:

```cpp
ITokenizer
↓
LlamaTokenizer
```

현재는:

* native decode (partial)
* native encode (MVP)
* 일부 Python bridge 유지

최종 목표는:

# Python tokenizer 완전 제거

입니다.

즉:

```text
모델만 있으면 실행
```

이 목표입니다.

---

## 4. Weight Registry Pattern

구조:

```cpp
unordered_map<string, Tensor>
```

원칙:

# HF tensor naming 그대로 유지

즉:

```text
safetensors key == internal key
```

rename / remap 하지 않습니다.

---

## 5. Regression First

가장 중요한 규칙입니다.

코드 수정 후 반드시:

```bash
python scripts/regression_test.py
```

를 실행합니다.

PASS가 뜨지 않으면

# 다음 작업 금지

입니다.

---

# 현재 구현 완료 상태

## Core Runtime

| 영역                                             | 상태 |
| ---------------------------------------------- | -- |
| Config / safetensors 로딩                        | 완료 |
| BF16 / F16 / F32 tensor 디코딩                    | 완료 |
| Embedding / RMSNorm                            | 완료 |
| Attention (GQA + RoPE + causal)                | 완료 |
| MLP (SwiGLU)                                   | 완료 |
| Transformer Block                              | 완료 |
| Full 22-layer forward parity                   | 완료 |
| Regression Test 자동화                            | 완료 |
| Sampler (greedy / temperature / top-k / top-p) | 완료 |
| Repetition penalty                             | 완료 |
| Multi-token generation loop                    | 완료 |
| KV cache                                       | 완료 |
| Prefill / Decode split                         | 완료 |
| Persistent Chat History                        | 완료 |
| Sliding Window                                 | 완료 |
| Interactive CLI                                | 완료 |

---

## Tokenizer

| 영역                         | 상태 |
| -------------------------- | -- |
| ITokenizer abstraction     | 완료 |
| BpeTokenizer (공통 BPE 엔진)  | 완료 |
| LlamaTokenizer (Metaspace) | 완료 |
| tokenizer.json load        | 완료 |
| vocab / special token parse | 완료 |
| BOS / EOS handling         | 완료 |
| Native Encode (full BPE)   | 완료 |
| BPE merge rules            | 완료 |
| byte fallback              | 완료 |
| unicode handling           | 완료 |
| Full HF parity             | 완료 |
| Native Decode              | 완료 |

HF parity 검증 완료 (`tokenizer_parity_test.py` PASS).

---

## Serving Layer

| 영역                    | 상태      |
| --------------------- | ------- |
| Streaming output      | 임시 비활성화 |
| Request abstraction   | 완료      |
| Request Queue         | 완료      |
| Scheduler (sequential)| 완료      |
| Continuous batching   | 예정      |
| OpenAI-compatible API | 예정      |

---

# 검증 결과

## Full Forward Parity

TinyLlama 기준:

```text
C++ argmax: 2643
Python argmax: 2643
PASS: forward parity verified
```

즉:

* HF logits parity 통과
* argmax 일치
* multi-token generation 정상 동작
* KV cache 기반 decode 정상 동작

현재는 단순 forward 테스트가 아니라
실제 generation 단계까지 정상 동작합니다.

---

# 현재 가능한 실행 흐름

## Example Flow

```text
User text input
→ tokenizer encode
→ full prefill
→ decode loop (KV cache)
→ sampler
→ next token 선택
→ EOS 종료
→ decode output
```

예시:

```text
Input:
hello

Output:
Hello! How can I help you today?
```

주의:

현재 streaming output은
성능상 이유로 임시 비활성화했습니다.

개발 단계에서는

```text
Generate()
→ 마지막에 한 번 decode
```

방식을 유지합니다.

---

# 현재 프로젝트 구조

```text
src/
├── main.cpp
│   └── interactive CLI runtime
│
├── core/runtime/
│   ├── EmbeddingLookup.h
│   ├── RMSNorm.h
│   ├── Linear.h
│   ├── RoPE.h
│   ├── MLP.h
│   ├── Attention.h
│   ├── TransformerBlock.h
│   ├── Sampler.h
│   ├── Sampler.cpp
│   └── KVCache.h
│
├── tokenizer/
│   ├── ITokenizer.h
│   ├── LlamaTokenizer.h
│   ├── LlamaTokenizer.cpp
│   ├── TokenizerJsonLoader.h
│   └── TokenizerJsonLoader.cpp
│
├── models/
│   ├── base/
│   │   ├── IModelRunner.h
│   │   ├── GenerateOptions.h
│   │   ├── ModelConfigLoader.h
│   │   ├── SafeTensorHeaderParser.h
│   │   └── SafeTensorTensorLoader.h
│   │
│   └── llama/
│       ├── LlamaRunner.h
│       └── LlamaRunner.cpp
│
├── serving/
│   ├── GenerationRequest.h
│   ├── RequestQueue.h
│   ├── Scheduler.h
│   └── Scheduler.cpp
│
└── scripts/
    ├── verify_full_forward.py
    └── regression_test.py
```

---

# 다음 작업 (AI가 이어서 해야 할 작업)

## 최우선 작업

# Request abstraction + Request Queue

즉:

```text
single blocking Generate()
```

에서

```text
request 생성
→ queue 저장
→ scheduler가 처리
```

구조로 전환해야 합니다.

---

## 다음 구현 순서

```text
1. GenerationRequest      ✓ 완료
2. RequestQueue           ✓ 완료
3. Simple Scheduler       ✓ 완료
4. Continuous batching    → 다음 작업
5. OpenAI-compatible API  → 예정
```

즉

# mini-vLLM serving layer

구현 단계입니다.

---

## tokenizer 완료

Python tokenizer 완전 제거 완료.
BPE parity 달성.

---

# 빌드 방법

## 반드시 Release Build 권장

Debug는 매우 느립니다.

```bash
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release
```

실행:

```powershell
cd cmake-build-release
./mLLM.exe
```

---

# 주의 사항

---

## Streaming Output는 지금 비활성화 유지

이유:

per-token decode + flush는
개발 단계에서 매우 느립니다.

streaming은

# 성능 개선이 아니라 UX 개선

입니다.

현재 우선순위가 아닙니다.

---