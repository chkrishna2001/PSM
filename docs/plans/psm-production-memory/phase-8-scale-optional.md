# Phase 8 — Scale (optional, gated)

**Status:** Gated — do not start unless Phase 5 plateaus  
**Goal:** Explore capacity only after prod-extraction mix + chunking + guardrails are exhausted.

---

## Entry criteria

Start Phase 8 **only if**:

- Phase 5 full mix run completed.
- Grounding stuck **< 70%** on plan + workflow suites after guardrails + chunking.
- Team agrees scale is the bottleneck (not curriculum or product path).

---

## What scale does and does not fix

| Approach | Fixes | Does not fix |
|----------|-------|--------------|
| Chunking (Phase 2) | Long `llmResponse` in 2048 ctx | Extraction quality |
| 100M params | Capacity for harder patterns | Context length |
| 4096 context | Slightly larger single chunk | 32k handoffs without chunking |

**Default:** chunk per call; do not chase 32k context first.

---

## Option A — 4096 context (same 50M)

- Extend `context_length` in [configs.py](../../../psm-model/src/psm_model/configs.py).
- RoPE already in [tiny_transformer.py](../../../psm-model/src/psm_model/model/tiny_transformer.py).
- Train short continuation from promoted Phase 6 checkpoint.
- Re-run Phase 1 — expect modest gain; chunking still required for very long text.

---

## Option B — 100M depth expand

**Does not exist yet.**

| Parameter | 50M today | 100M target |
|-----------|-----------|-------------|
| `n_layer` | 16 | 32 |
| `n_embd` | 512 | 512 |
| Init | — | `expand_checkpoint.py` from 058000 / Phase 6 best |

Requires new preset + expansion script before any RunPod job.

---

## Option C — 200M

Only after 100M + `prod-extraction-v1` (or v2) proven on grounding bar.

---

## Tasks (when gated in)

- [ ] Document plateau evidence in Phase 5 results.
- [ ] If 4096: config flag + short train + Phase 1 re-eval.
- [ ] If 100M: implement `expand_checkpoint.py` + preset; smoke 2k steps.
- [ ] Compare ROI vs more curriculum diversity (prefer curriculum first).

---

## Exit criteria

- [ ] Measurable grounding lift vs Phase 6 promoted checkpoint, **or**
- [ ] Explicit decision to stop scale attempts and fix curriculum/product instead.

---

## Results

_(Not applicable until gated.)_
