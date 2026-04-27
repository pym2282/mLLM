# mLLM

미니 vLLM 스타일의 C++ LLM 추론 런타임.

HuggingFace의 `config.json` + `model.safetensors`를
**ONNX / TorchScript 없이 직접 로드**하여

* full forward
* generation
* KV cache decode
* tokenizer encode/decode
* model-specific runtime

를 수행합니다.

목표는 단순 inference demo가 아니라

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

까지 단계적으로 구현합니다.

---

# 현재 프로젝트 상태 (매우 중요)

현재 프로젝트는

# Core Runtime 완료 + Multi-model 구조 정리 완료

상태입니다.

즉:

* full forward parity 완료
* generation loop 완료
* KV cache decode 완료
* prefill / decode split 완료
* tokenizer abstraction 완료
* Qwen3 FP16 parity 완료
* QwenRunner 분리 완료
* LlamaRunner / QwenRunner 구조 분리 완료
* ModelRunnerFactory 자동 분기 완료

까지 끝났습니다.

현재는

# "LLM 구현 중"

이 아니라

# "실사용 가능한 mini-vLLM 엔진 완성 단계"

입니다.

이제 남은 핵심은

# AWQ 지원 + Serving Layer + Continuous Batching

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

---

# 현재 지원 모델

## Llama Family

* TinyLlama
* Llama 계열
* 기본 Llama architecture

## Qwen Family

* Qwen3 FP16
* Qwen3-8B-FP16 parity 완료

## 다음 목표

* Qwen3-14B-AWQ
* Qwen3-8B-AWQ
* GGUF loader (후순위)
* Mistral
* Gemma

---

# 중요한 설계 원칙

---

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
* Linear
* RoPE
* MLP
* Sampler
* TransformerBlock

모델별 구현과 완전히 분리합니다.

---

## 2. Model-specific Logic 분리

위치:

```text
src/models/
```

구조:

```text
IModelRunner
 ├── LlamaRunner
 └── QwenRunner
```

즉:

* weight loading
* layer registry
* generation loop
* model-specific config

는 모델별 Runner가 담당합니다.

runtime은 재사용합니다.

이 구조가 매우 중요합니다.

---

## 3. Tokenizer Abstraction

위치:

```text
src/tokenizer/
```

구조:

```text
ITokenizer
 ├── BpeTokenizer
 ├── LlamaTokenizer
 └── QwenTokenizer
```

현재:

* native encode 완료
* native decode 완료
* EOS/BOS handling 완료
* HF parity 확보

즉:

# Python tokenizer 제거 완료

입니다.

목표:

```text
모델만 있으면 실행 가능
```

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

이게 유지보수 핵심입니다.

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

---

## Core Runtime

| 영역                                             | 상태 |
| ---------------------------------------------- | -- |
| Config / safetensors 로딩                        | 완료 |
| sharded safetensors 지원                         | 완료 |
| BF16 / F16 / F32 tensor decode                 | 완료 |
| Embedding / RMSNorm                            | 완료 |
| Attention (GQA + RoPE + causal)                | 완료 |
| MLP (SwiGLU)                                   | 완료 |
| Transformer Block                              | 완료 |
| Full forward parity                            | 완료 |
| Sampler (greedy / top-k / top-p / temperature) | 완료 |
| Repetition penalty                             | 완료 |
| Multi-token generation                         | 완료 |
| KV cache                                       | 완료 |
| Prefill / Decode split                         | 완료 |
| Interactive CLI                                | 완료 |

---

## Tokenizer

| 영역                       | 상태 |
| ------------------------ | -- |
| ITokenizer abstraction   | 완료 |
| BpeTokenizer             | 완료 |
| LlamaTokenizer           | 완료 |
| QwenTokenizer            | 완료 |
| tokenizer.json load      | 완료 |
| vocab / merges parse     | 완료 |
| BOS / EOS handling       | 완료 |
| Native Encode            | 완료 |
| Native Decode            | 완료 |
| Qwen chat template       | 완료 |
| GPT2 byte decode cleanup | 완료 |
| HF parity                | 완료 |

---

## Qwen Support

| 영역                        | 상태 |
| ------------------------- | -- |
| Qwen3 config parse        | 완료 |
| Qwen3 FP16 load           | 완료 |
| Qwen3 sharded safetensors | 완료 |
| QK Norm                   | 완료 |
| Qwen chat template        | 완료 |
| Qwen tokenizer            | 완료 |
| greedy decode parity      | 완료 |
| generation 정상 동작          | 완료 |
| QwenRunner 분리             | 완료 |

검증 결과:

```text
Assistant: Paris.
```

정상 출력 완료.

즉:

# Qwen3 FP16 runtime 성공

입니다.

---

## Architecture

| 영역                 | 상태 |
| ------------------ | -- |
| QwenRunner 분리      | 완료 |
| LlamaRunner 정리     | 완료 |
| ModelRunnerFactory | 완료 |
| config 기반 자동 분기    | 완료 |

현재:

```cpp
auto bundle =
    ModelRunnerFactory::Create(model_path);
```

만으로

* runner 선택
* tokenizer 선택

이 자동 처리됩니다.

---

# 검증 결과

---

## TinyLlama Forward Parity

```text
C++ argmax == Python argmax
PASS
```

---

## Qwen3 Forward + Generation

```text
Assistant: Paris.
```

검증 완료.

즉:

* HF logits parity
* argmax 일치
* multi-token generation 정상
* KV cache decode 정상
* tokenizer parity 정상

까지 완료되었습니다.

---

# 현재 프로젝트 구조

```text
src/
├── main.cpp
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
│   ├── BpeTokenizer.h
│   ├── LlamaTokenizer.*
│   ├── QwenTokenizer.*
│   └── TokenizerJsonLoader.*
│
├── models/
│   ├── base/
│   │   ├── IModelRunner.h
│   │   ├── GenerateOptions.h
│   │   ├── ModelRunnerFactory.h
│   │   ├── ModelConfigLoader.h
│   │   ├── SafeTensorHeaderParser.h
│   │   └── SafeTensorTensorLoader.h
│   │
│   ├── llama/
│   │   ├── LlamaRunner.h
│   │   └── LlamaRunner.cpp
│   │
│   └── qwen/
│       ├── QwenRunner.h
│       └── QwenRunner.cpp
│
├── serving/
│   ├── GenerationRequest.h
│   ├── RequestQueue.h
│   └── Scheduler.*
│
└── scripts/
    ├── regression_test.py
    ├── test_qwen3_parity.py
    └── verify_full_forward.py
```

---

# 다음 작업 (매우 중요)

---

# 최우선 작업

# Qwen3-14B-AWQ 지원

현재 가장 중요한 다음 단계입니다.

이유:

FP16은 검증용이고

실사용은

# AWQ

입니다.

목표:

```text
Qwen3-14B-AWQ
```

특히 3060 12GB 기준으로

가장 좋은 sweet spot입니다.

---

# 구현 순서

```text
1. AWQ tensor naming 확인
2. qweight / qzeros / scales / g_idx 로드
3. dequantize → fp16 parity 확보
4. 이후 fused int4 matmul 고려
```

주의:

GGUF는 후순위입니다.

지금은

# AWQ 먼저

입니다.

---

# 그 다음

```text
Continuous batching
→ Scheduler 고도화
→ OpenAI-compatible API
```

즉

# mini-vLLM serving layer

완성 단계입니다.

---

# 빌드

## Release Build 권장

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

## Streaming Output는 우선순위 낮음

streaming은

# 성능 개선이 아니라 UX 개선

입니다.

현재 우선순위는

```text
AWQ
runtime
batching
serving
```

입니다.

streaming 최적화는 후순위입니다.

---