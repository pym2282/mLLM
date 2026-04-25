# mLLM

미니 vLLM 스타일 C++ LLM 추론 런타임. HuggingFace safetensors / config.json 을 **ONNX / TorchScript 없이** 직접 로드해 forward 수행.

- **Stack:** C++17, CMake, LibTorch, nlohmann/json
- **타겟 플랫폼:** Windows / MSVC / CLion (현재 개발 환경)
- **모델:** TinyLlama (22 layers, 2048 hidden, 32/4 GQA, BF16) — 최초 레퍼런스
- **확장 대상:** Qwen, Mistral (같은 pre-norm + SwiGLU + RoPE 계열)

> **ONNX / TorchScript 폐기** — transformers DynamicCache export 불안정 문제로 직접 safetensors 로딩 경로로 전환함. 이 경로 복귀하지 말 것.

---

## ⚡ 현재 상태 (스냅샷)

| 영역 | 상태 |
|---|---|
| Config / safetensors 로딩 | ✅ |
| BF16 / F16 / F32 tensor 디코딩 | ✅ |
| Embedding → Layer0 input_layernorm 패리티 | ✅ |
| **Full forward (22 layers) 패리티** | ✅ **HF 와 logits 일치, argmax 일치** |
| Sampler | ⏳ 다음 단계 |
| Tokenizer runtime | ⏳ 미구현 |
| KV cache / scheduler / API | ⏳ 미구현 |

### ✅ 검증 결과

`input_ids = [[1, 2, 3, 4]]` 기준 (TinyLlama, BF16, eager attention):

| | C++ | Python (HF) |
|---|---|---|
| `logits[0,-1,:5]` | [-2.359, -2.625, 9.5, 13.25, 9.5] | [-2.375, -2.641, 9.5, 13.25, 9.563] |
| argmax token_id | **3** | **3** |

절대오차 < 0.07 (BF16 노이즈), top-1 일치. **Full forward parity 통과.**

**다음 단계:** Sampler (greedy / top-p / temperature) → tokenizer runtime → KV cache.

---

## 디렉토리 구조

```
src/
├── main.cpp                                # 엔트리, 더미 input [[1,2,3,4]] 로 forward
├── core/runtime/                           # stateless 수치 연산 (재사용 가능)
│   ├── EmbeddingLookup.h
│   ├── RMSNorm.h
│   ├── Linear.h                            # F.linear(x, weight, bias), HF [out,in] 레이아웃
│   ├── RoPE.h                              # rotate_half, concat-doubled cos/sin
│   ├── MLP.h                               # SwiGLU: silu(gate) * up → down
│   ├── Attention.h                         # GQA + RoPE + causal + fp32 softmax
│   └── TransformerBlock.h                  # LayerWeights struct + pre-norm residual 2단
│
├── models/
│   ├── base/
│   │   ├── IModelRunner.h                  # ModelConfig 구조체 + 런타임 인터페이스
│   │   ├── ModelConfigLoader.h             # config.json → ModelConfig
│   │   ├── SafeTensorLoader.h              # 파일 존재 확인
│   │   ├── SafeTensorHeaderParser.h        # JSON 헤더 → tensor_map
│   │   └── SafeTensorTensorLoader.h        # raw bytes → torch::Tensor (from_blob + clone)
│   │
│   └── llama/
│       ├── LlamaRunner.h                   # IModelRunner 구현, weights_ + layer_weights_
│       └── LlamaRunner.cpp                 # Load / Forward 본체 + probe 로직

scripts/
└── verify_full_forward.py                  # HF 기준값 + layer 10 sub-block hook
```

---

## 아키텍처 설계 원칙

1. **Stateless 수치 op** — `core/runtime/*.h` 는 전부 static `Forward()`. 상태 없음. 모델 간 재사용 가능.
2. **Model-specific 로직은 `models/llama/` 에만** — 런타임은 어느 아키텍처가 호출하는지 모름.
3. **Weight 레지스트리 패턴** — `std::unordered_map<string, torch::Tensor> weights_` 에 통합 저장, `std::vector<LayerWeights>` 가 per-layer view.
4. **HF 명명 규칙을 그대로 사용** — 재구성/rename 없음. safetensors 키 = C++ 내부 키.
5. **dtype 정책은 HF 패리티 기준** — RMSNorm / softmax 는 fp32 업캐스트, 나머지 (Linear, MLP, attention score matmul) 는 입력 dtype 유지.
6. **hardcoded arch 상수 금지** — `num_heads`, `rope_theta`, `head_dim` 전부 `ModelConfig` 에서 읽음.

---

## 데이터 흐름 (현재)

### `Load(model_path)`
```
config.json          → LoadModelConfigFromJson
model.safetensors    → SafeTensorHeaderParser (tensor_map_)
                     → SafeTensorTensorLoader (lazy, LoadWeight)
                     → LoadAllWeights(): embed + norm + lm_head + 22×9 per layer
                     → layer_weights_ vector 구축
```

### `Forward(input_ids, attention_mask)`
```
input_ids [B, S]
  → EmbeddingLookup::Forward(ids, embed_tokens.weight)
  → for i in 0..num_layers:
        TransformerBlock::Forward(hidden, layer_weights_[i], ...)
          = residual + Attention(RMSNorm(x, input_ln))
            residual + MLP(RMSNorm(x, post_attn_ln))
  → RMSNorm::Forward(hidden, model.norm.weight, eps)
  → Linear::Forward(hidden, lm_head.weight or embed_tokens.weight)  [logits]
```

### `TransformerBlock::Forward` (pre-norm)
```
residual = x
x = RMSNorm(x, input_layernorm)
x = Attention(x)
x = residual + x

residual = x
x = RMSNorm(x, post_attention_layernorm)
x = MLP(x)
x = residual + x
```

---

## 수치 검증 이력

### ✅ 통과 (2026-04-23)

**Embedding + Layer 0 input_layernorm**

| idx | C++ (`0.01 *` 풀기) | Python | diff |
|---|---|---|---|
| 0 | -0.001183 | -0.001174 | ~1e-5 |
| 1 | 0.003159 | 0.003143 | ~2e-5 |
| 2 | -0.030029 | -0.030029 | 0 |
| 3 | -0.028687 | -0.028564 | ~1e-4 |
| 4 | -0.001709 | -0.001701 | ~1e-5 |

→ `SafeTensorTensorLoader` BF16 디코딩, `EmbeddingLookup`, `RMSNorm` 전부 정상.

**libtorch tensor `std::cout` 주의:** 값이 작으면 `0.01 *` 같은 스케일 prefix 가 먼저 찍히고 그 아래 값이 100× 증폭된 상태로 나옴. 첫 검증 때 이걸 못 보고 "100배 차이 버그" 로 오독한 이력 있음.

### ✅ 통과 (2026-04-25) — Full 22-layer forward

| | C++ | Python | diff |
|---|---|---|---|
| `block[21]` raw [0,0,:5] | [-0.051, 0.084, -0.097, -0.264, 0.143] | [-0.053, 0.080, -0.094, -0.260, 0.143] | < 5e-3 (BF16 노이즈) |
| `post_norm` [0,0,:5] | [-0.424, 0.664, -0.820, -2.281, 1.188] | [-0.451, 0.645, -0.809, -2.297, 1.211] | < 0.03 |
| `logits[0,-1,:5]` | [-2.359, -2.625, 9.5, 13.25, 9.5] | [-2.375, -2.641, 9.5, 13.25, 9.563] | < 0.07 |
| argmax token_id | 3 | 3 | ✅ |

**bisection 진단 중 발견한 함정:** HF `output_hidden_states=True` 의 `hidden_states` 튜플은 마지막 원소가 **`norm(layer_N-1 raw output)`** 이라 raw layer 출력과 다름. layer i 의 raw 는 `hidden_states[i+1]` (i = 0..N-2), 마지막은 hook 으로 별도 캡처해야 함. 이걸 raw 인 줄 알고 비교하면 layer N-1 만 8x 차이로 보이는 가짜 버그가 잡힘.

---

## 재개 방법

이 저장소를 다시 열어서 이어서 하려면:

### 1. 빌드 (Windows + CLion 환경 전제)
```powershell
# CLion 의 cmake-build-debug 디렉토리 사용 가정.
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --build cmake-build-debug
```

### 2. 회귀 검증 — C++ vs Python 두 줄 대조
```powershell
# C++:  실행 디렉토리는 cmake-build-debug (relative path ../models 때문)
$env:PATH = "C:\libtorch\lib;" + $env:PATH
cd cmake-build-debug
.\mLLM.exe
```
```bash
# Python: 프로젝트 루트에서
python scripts/verify_full_forward.py
```

확인할 줄:
- C++ `[LlamaRunner] logits[0,-1,:5]: ...`
- C++ `[LlamaRunner] last-token argmax token_id=N`
- Python `logits[0,-1,:5]: [...]`
- Python `last-token argmax token_id: N`

**판정:** argmax 같으면 forward 정상. logits 절대오차 < 0.1 이면 BF16 노이즈 범위.

### 3. 분석 (회귀 발생 시)

| 관측 | 의미 |
|---|---|
| `WEIGHT COUNT MISMATCH` | 201 이 아닌 N 이면 weight 오로딩. `tensor_map_` 키 dump 해서 누락/중복 탐색 |
| argmax 다름 | forward 의 어느 단계 깨짐. git history 에서 직전 작업 후보부터 점검. bisection probe 임시 부활 (layer 0/1/3/5/8/10/15/21 `[0,0,:5]`) |
| argmax 같은데 logits 차이 큼 | dtype 불일치 (fp32 vs bf16) 또는 sampler 미반영 |
| Python 이 layer 21 만 8x 차이 | HF `hidden_states` 마지막 원소가 norm 후 값임을 잊은 것 (가짜 버그). raw 비교는 hook 으로 |
| `after_attn` 에서만 다름 | Attention 내부 (scale, mask, softmax, repeat_interleave 방향) |
| `after_mlp` 에서만 다름 | MLP (SiLU, gate×up 순서, down_proj) |
| 둘 다 맞고 다음 layer 에서 다름 | 잔차 덧셈 dtype 문제 |

### 5. 의존성 (Python 측)
```bash
pip install torch transformers
```

### 6. 모델 파일 위치
`models/TinyLlama/` 에 HF 저장소 구조 그대로:
```
models/TinyLlama/
├── config.json
├── model.safetensors
├── tokenizer.json         # (현재 미사용 — tokenizer runtime 단계에서 연결)
└── ...
```
C++ 실행 디렉토리는 `build/` 기준이므로 C++ 에서는 `../models/TinyLlama` 로 접근 (`src/main.cpp:12`).

---

## 로드맵

- [x] Config / safetensors 헤더 / 실제 tensor bytes 로딩
- [x] Embedding lookup
- [x] RMSNorm (검증 통과)
- [x] Linear
- [x] RoPE
- [x] MLP (SwiGLU)
- [x] Attention (GQA + causal + fp32 softmax)
- [x] Transformer Block (pre-norm residual 2단)
- [x] Full N-layer forward + LM head
- [x] Full forward 수치 패리티 검증 (HF eager 와 logits / argmax 일치)
- [ ] **Sampler (greedy / temperature / top-p)** ← 다음 단계 — fp32 cast 필요
- [ ] Tokenizer runtime (HF tokenizer.json 파싱 or 외부 라이브러리)
- [ ] KV cache (prefill vs decode 분리, position_ids 확장 `[B, 1]` 경로)
- [ ] Scheduler / continuous batching (vLLM-lite 스타일)
- [ ] OpenAI-compatible HTTP API

---

## 코딩 스타일 규칙 (이어서 작업할 때 준수)

- `core/runtime/*.h` 는 **header-only + static Forward** 형태 유지. 상태 없는 pure function.
- 예외는 `std::runtime_error` 만 사용. 경계 검증 (`x.dim()`, shape match) 최소한으로.
- `std::cout` 로깅은 **Forward 1회당 O(1)** 만. per-layer 반복 안에서 절대 안 찍음 (probe 는 특정 index 조건부).
- Comment 는 WHY 만. WHAT 은 식별자로. 제거해도 의미 유지되면 지우기.
- `weights_.at(name)` 으로 접근 — 없으면 의도적 throw. `find` 로 묵인 금지.
- 새 모델 계열 (Qwen/Mistral) 추가할 때 `core/runtime/*.h` 는 건드리지 않음. `models/qwen/QwenRunner.cpp` 에서 같은 `LayerWeights` / `TransformerBlock` 재사용.

---

## 주요 설계 결정 (gotchas)

1. **libtorch API 이식성** — `torch::pow(Scalar, Tensor)` / `torch::outer` / `torch::silu` 는 버전별 lookup 불안정. `exp(-x*log(θ))` / unsqueeze broadcast / `x * sigmoid(x)` 로 대체 구현.
2. **GQA repeat 방향** — `repeat_interleave(n_rep, dim=1)` 사용. `repeat` 은 heads 가 잘못 섞이는 silent bug.
3. **causal mask** — `triu(-inf, diagonal=1)` 로 strict upper triangle. scale 먼저, mask 그다음, softmax 마지막.
4. **BF16 byte layout** — safetensors 는 little-endian. `torch::from_blob(bytes, shape, kBFloat16).clone()` 로 ownership transfer.
5. **MSVC `_` 오염** — `<io.h>` 가 `_` 를 매크로로 점유. 구조체 binding 에서 `auto [it, _] = ...` 쓰지 말고 명명된 이름 (`inserted` 등) 사용.
6. **cos/sin 레이아웃** — Llama/NeoX 계열은 `concat([freqs, freqs], -1)` (concat-doubled). GPT-J 스타일 interleaved 와 다름. 잘못 쓰면 silent wrong.
7. **rope_theta 위치** — transformers 5.x 는 `rope_parameters` 객체 안에 중첩. ModelConfigLoader 가 top-level + nested 둘 다 확인.
8. **tie_word_embeddings** — TinyLlama = false (별도 `lm_head.weight`), Gemma = true (embedding 재사용). `ModelConfig.tie_word_embeddings` 로 분기.
9. **HF `output_hidden_states` 의 마지막 원소** — `norm(layer_N-1 raw output)` 임에 주의. raw 비교는 `m.model.layers[-1]` 에 forward_hook 으로.
10. **libtorch tensor `std::cout` 의 `0.01 *` prefix** — 작은 값을 100× 증폭해서 출력. 주석으로 보이는 숫자에 prefix 붙어 있는지 항상 확인.

---

## 의존성

```cmake
# CMakeLists.txt
set(Torch_DIR "C:/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
```
- LibTorch (Windows prebuilt, 현재 환경 `C:/libtorch/`)
- nlohmann/json (header-only, `external/` 혹은 system include)

Python 검증:
- `torch`, `transformers` (≥ 4.36, `attn_implementation` 옵션 지원)

---

## 커밋 히스토리 맥락

- `6dd2ee2` Initial commit
- `32decc0` init project (초기 스캐폴딩)
- `4d63dfb` Add safetensors tensor loading and embedding runtime flow (가짜 `torch::rand` → 실제 bytes 디코딩 수정 포함)
- `c7081a2` Implement real tensor loading and RMSNorm validation path
- **(current, uncommitted)** RoPE / Linear / MLP / Attention / TransformerBlock 추가, LlamaRunner 재배선, layer bisection probe
