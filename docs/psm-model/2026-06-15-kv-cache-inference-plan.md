# Plan 1: KV Cache for PSM 50M Production Inference

**Goal:** Make `remember()` fast enough for production on CPU (target **<5s p95**) and GPU (**<1s p95**) without changing model weights.

**Why this matters:** We moved from Qwen2.5 → custom 50M expecting better CPU inference. Today a 50M model takes **~56s/remember on CPU** because the decoder re-runs the full sequence every token. That is a runtime bug, not a model-size problem.

---

## Current state (diagnosis)

| Item | Today |
|------|--------|
| Model | Real **~50M** (`16×512`, ~631MB fp32 checkpoint) |
| Decode | `TinyDecoderModel.generate()` — **no KV cache** |
| Per token | Full forward over **entire** `input_ids` (O(n²) decode) |
| `max_new_tokens` | **384** default (`remember_server`, `PsmModelRuntime`) |
| Output format | **tagged** — usually ends at `END` (~30–80 tokens) |
| Server | `remember_server` loads once ✅ — good |
| Measured CPU | **~56s** / remember (warm) |
| Measured GPU (4060) | **~16s** / remember (warm) |

Root code:

```text
psm-model/src/psm_model/model/tiny_transformer.py  → generate() loop
psm-model/src/psm_model/generate.py               → generate_storage_json()
src/psm-core/src/remember-server.ts               → long-lived Python server
src/psm-core/src/psm-model-runtime.ts             → maxNewTokens=384
```

---

## Design

### 1. KV cache in `TinyDecoderModel`

**Prefill phase:** Run model once on prompt tokens; store per-layer `(K, V)` tensors.

**Decode phase:** Append one token at a time; attention reads cached K/V for prior positions; only compute Q/K/V for the new position.

**API sketch:**

```python
@dataclass
class GenerationState:
    past_key_values: tuple[tuple[Tensor, Tensor], ...]  # per layer (k, v)
    input_ids: Tensor  # full sequence for eos/stop checks

def forward_with_cache(self, input_ids, past_key_values=None, use_cache=True) -> ...
def generate(self, input_ids, *, max_new_tokens, eos_id, stop_strings=None) -> ...
```

**Touch points:**

- `_CausalSelfAttention.forward` — accept optional `past_kv`, return `present_kv`
- `_DecoderBlock.forward` — thread cache through
- `TinyDecoderModel.forward` — return `past_key_values` when `use_cache=True`
- `TinyDecoderModel.generate` — prefill once, then single-token steps

**RoPE:** Apply rotary positions using **absolute position index** (`past_len + t`) so cached keys stay valid.

### 2. Tagged early stop (same PR or immediately after)

In `generate()`, after each new token decode:

- Decode running text (or check token pattern for `\nEND\n`)
- Stop when tagged parser sees line `END` (reuse `parse_tagged_decision` prefix check or lightweight line scanner)

Reduces average tokens from ~384 cap to ~50–80.

### 3. Lower production token budget

| Surface | Current | Proposed |
|---------|---------|----------|
| `remember_server` default | 384 | **128** (storage tagged) |
| `PsmModelRuntime` default | 384 | **128** |
| `probe_checkpoint` | 220 | keep |
| recall eval | 256 | keep |

Add env override: `PSM_MAX_NEW_TOKENS`.

### 4. Optional follow-up (not blocking KV)

- **`force_action_head=True`** in prod `generate_storage_json` — one forward for action, shorter generation
- **FP16 on GPU / INT8 dynamic quant on CPU** — separate track after KV lands
- **`torch.compile`** on `forward_with_cache` — benchmark after correctness

---

## Implementation phases

### Phase A — Correctness (required)

1. Add `forward_with_cache` + `GenerationState` to `tiny_transformer.py`
2. Rewrite `generate()` to use cache path; keep non-cache path behind flag for A/B (`use_kv_cache=True` default)
3. Unit tests:
   - Cached vs non-cached outputs **identical** (greedy, `temperature=0`) for same prompt
   - EOS stop still works
   - Context length cap respected
4. Wire through `generate_storage_json` / `remember_server` (no API change)

### Phase B — Tagged stop + token caps

1. Early `END` stop in `generate()` when `output_format=tagged`
2. Drop defaults to 128 in server + TS runtime
3. Log `tokens_generated` in remember_server response (debug)

### Phase C — Benchmark gate

Script: `psm-model/scripts/bench_remember_latency.py`

| Case | Device | Metric |
|------|--------|--------|
| expanded-probe sample (10 rows) | CPU | p50, p95 latency |
| same | CUDA (RunPod or `PSM_ALLOW_LOCAL_GPU=1`) | p50, p95 |
| LoCoMo PSM-format single turn | CPU + GPU | p50 |

**Ship bar (inference):**

| Device | p95 `remember()` |
|--------|------------------|
| CPU | **<5s** |
| GPU | **<1s** |

Record before/after in `psm-model/checkpoints/bench/kv-cache-{date}.json`.

---

## Files to change

| File | Change |
|------|--------|
| `psm-model/src/psm_model/model/tiny_transformer.py` | KV cache, new generate loop |
| `psm-model/tests/test_kv_cache_generate.py` | parity + stop tests (new) |
| `psm-model/src/psm_model/generate.py` | pass `stop_on_tagged_end`, token cap |
| `psm-model/src/psm_model/remember_server.py` | default max_new_tokens=128 |
| `src/psm-core/src/psm-model-runtime.ts` | maxNewTokens default 128 |
| `psm-model/scripts/bench_remember_latency.py` | new benchmark |
| `benchmark/locomo/run-ingest-psm-model.ps1` | allow local GPU optional (not CPU-only) |

---

## Risks

| Risk | Mitigation |
|------|------------|
| RoPE position bugs | Parity test cached vs uncached |
| SDPA + cache interaction | Start with explicit attn matmul path for cache; SDPA for prefill only if needed |
| Memory growth on long prompts | Cap prefill at `context_length`; truncate prompt like today |
| Training/inference divergence | No weight change; inference-only |

---

## Out of scope (this plan)

- ONNX / llama.cpp export
- Quantization
- Recall-path KV (separate `max_new_tokens` for recall_plan)
- Changing training

---

## Success definition

1. Cached generate matches uncached on gate probes (action + tagged fields).
2. CPU p95 `<5s`, GPU p95 `<1s` on standard remember probe.
3. LoCoMo n=25 ingest on 4060 completes in **<10 min** (vs ~74 min CPU forced).
