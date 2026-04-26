# mLLM

미니 vLLM 스타일의 C++ LLM 추론 런타임입니다.
HuggingFace의 safetensors / config.json을 **ONNX / TorchScript 없이 직접 로드**하여 forward 및 generation을 수행합니다.

* **Stack:** C++17, CMake, LibTorch, nlohmann/json
* **Target Platform:** Windows / MSVC / CLion
* **Reference Model:** TinyLlama (22 layers, 2048 hidden, 32/4 GQA, BF16)
* **Expansion Target:** Qwen, Mistral

> ONNX / TorchScript 경로는 폐기했습니다.
> transformers DynamicCache export 불안정 문제로 인해 safetensors 직접 로드 방식만 유지합니다.

---

# 현재 상태

## Inference Engine MVP 완료

프로젝트는 이제 단순 모델 로더가 아니라
실제로 텍스트 생성이 가능한 로컬 추론 엔진 단계까지 도달했습니다.

구현 완료:

* Config / safetensors 로딩
* BF16 / F16 / F32 tensor 디코딩
* Embedding Lookup
* RMSNorm
* Attention (GQA + RoPE + causal mask)
* MLP (SwiGLU)
* TransformerBlock
* Full 22-layer forward parity
* Regression Test 자동화
* Sampler (greedy / temperature / top-k)
* EOS stop
* Multi-token generation loop
* Python tokenizer bridge
* Text input → token ids → generation → decode → text output

즉,

```text
prompt 입력
→ tokenize
→ forward
→ sampler
→ next token 생성
→ decode
→ text output
```

까지 정상 동작합니다.

이제 프로젝트는

```text
loader prototype
→ inference engine MVP
```

단계로 올라왔습니다.

---

# 주요 업그레이드: KV Cache

## 왜 KV Cache가 중요한가

기존 방식은 매 토큰 생성마다
전체 prompt를 처음부터 다시 forward 했습니다.

예:

```text
hello
→ token1

hello + token1
→ token2

hello + token1 + token2
→ token3
```

즉,

```text
매 step마다 전체 sequence 재계산
```

이라 매우 느립니다.

---

## KV Cache란

Attention의 이전 K / V tensor를 저장하고
다음 step에서 재사용하는 방식입니다.

이로 인해:

* generation 속도 향상
* decode-only inference 가능
* 실제 LLM runtime 구조 구현
* continuous batching 기반 마련

이 가능해집니다.

즉,

```text
장난감 inference
→ 실제 inference engine
```

로 넘어가는 핵심 단계입니다.

---

# KV Cache 구현 완료

## 추가된 것

* `KVCache` struct
* layer별 cache storage
* Attention 내부 cache append + reuse
* TransformerBlock cache propagation
* LlamaRunner layer-wise cache ownership
* Prefill / Decode split
* decode용 position_ids 처리

---

## Prefill + Decode 분리

이제 generation은 다음처럼 동작합니다.

### Step 1: Prefill

처음 한 번만
전체 prompt를 계산합니다.

```text
full prompt forward
```

---

### Step 2+: Decode

이후에는
새 token 1개만 계산합니다.

```text
last token only
```

즉,

```text
[1, 31, vocab]
→ 매번 전체 계산
```

에서

```text
[1, 1, vocab]
→ incremental decode
```

로 바뀌었습니다.

이제 KV Cache가 실제로 속도 개선을 만들기 시작합니다.

---

# 출력 품질 개선

## Repetition Penalty

반복 생성 방지를 위해
repetition penalty를 추가했습니다.

예:

기존:

```text
hello hello hello hello
```

개선 후:

```text
hello world ...
```

현재 Sampler 지원:

* greedy
* temperature
* top-k
* repetition penalty

예정:

* top-p (nucleus sampling)
* streaming output

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

현재는 단순 forward 테스트가 아니라
실제 generation 단계까지 검증되었습니다.

---

# 현재 가능한 것

## Example Flow

```text
text input
→ tokenizer
→ generation
→ decode
→ output
```

예시:

```text
Model output:
I'm excited to share a quick and easy homemade vegan alternative...
```

즉,
Tokenizer를 포함한 실제 로컬 챗봇의 핵심 흐름이 완성되었습니다.

---

# 프로젝트 구조

```text
src/
├── main.cpp
│   └── interactive CLI entry
│
├── core/runtime/
│   ├── EmbeddingLookup.h
│   ├── RMSNorm.h
│   ├── Linear.h
│   ├── RoPE.h
│   ├── MLP.h
│   ├── Attention.h
│   ├── KVCache.h
│   ├── TransformerBlock.h
│   ├── Sampler.h
│   └── Sampler.cpp
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
└── scripts/
    ├── verify_full_forward.py
    └── regression_test.py
```

---

# 중요한 설계 원칙

## 1. Stateless Runtime

`core/runtime/*`

→ 상태 없는 pure function 유지

예:

* Attention
* RMSNorm
* Sampler

모델별 구현과 완전히 분리합니다.

---

## 2. Model-specific Logic 분리

`models/llama/`

→ TinyLlama 전용 로직만 존재

향후 Qwen / Mistral 추가 시
runtime 재사용 가능하도록 설계했습니다.

---

## 3. Weight Registry Pattern

```cpp
unordered_map<string, Tensor>
```

HF naming 그대로 유지합니다.

safetensors key = internal key

rename / remap 하지 않습니다.

---

## 4. Regression First

코드 수정 후 반드시:

```bash
python scripts/regression_test.py
```

PASS가 뜨지 않으면
다음 작업을 진행하지 않습니다.

이 프로젝트의 가장 중요한 개발 규칙입니다.

---

# 실행 방법

## Build

```powershell
cmake --build cmake-build-debug
```

---

## Regression Test

```bash
python scripts/regression_test.py
```

정상 결과:

```text
PASS: forward parity verified
```

---

## Direct Runtime Run

```powershell
cd cmake-build-debug
.\mLLM.exe
```

---

# 다음 단계

우선순위:

```text
Top-p
↓
Streaming output
↓
Persistent chat history
↓
Continuous batching
↓
Scheduler / batching
↓
OpenAI-compatible API
```

주의:

다음 작업 전 반드시 regression PASS 유지.

---
