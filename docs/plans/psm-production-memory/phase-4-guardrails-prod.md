# Phase 4 — Product guardrails

**Status:** Not started  
**Goal:** Stop writing garbage into SQLite even when the model hallucinates.  
**Depends on:** [Phase 1](phase-1-baseline-eval.md) (metrics)  
**Can run in parallel with:** [Phase 2](phase-2-chunking-pipeline.md)

---

## Problem

062000 ingest stored curriculum strings (e.g. D1:5 → `"Today run constoursated fact parser"`) because nothing in the product path rejected ungrounded `store` actions.

---

## Guardrails

### 1. Grounding reject

After model returns `action: store`:

- Compute token overlap between `remember_target` and `memory.content` + `facts[]`.
- If overlap below threshold → treat as `ignore`, log `grounding_reject`, do not persist.

### 2. Bleed blocklist

Reject store if content matches curriculum patterns:

- checkpoint / step-NNNN
- fact parser / malformed parser
- gate datasets / expanded probe templates
- PSM-internal training phrases

Reuse patterns from [ingest-quality-check.ts](../../../benchmark/locomo/src/ingest-quality-check.ts).

### 3. Align decode budget

| Setting | Today | Target |
|---------|-------|--------|
| Prod `max_new_tokens` | 128 ([psm-model-runtime.ts](../../../src/psm-core/src/psm-model-runtime.ts)) | **384** |
| Gate eval `max_new_tokens` | ~1200 | **384** for prod-parity suites |

Multi-fact + indexables need headroom; eval and prod must match.

### 4. Optional Kompress (overflow only)

[kompress-v2-base](https://huggingface.co/chopratejas/kompress-v2-base) for chunks still over budget **after** Phase 2 split:

- Conservative threshold only.
- Never compress workflow step lists without step-preservation verification.

---

## Tasks

- [ ] Implement `groundingReject()` in remember path (TS + optional Python mirror).
- [ ] Implement bleed blocklist shared module (TS + eval harness).
- [ ] Raise prod `max_new_tokens` to 384; align eval scripts.
- [ ] Add structured logs: `grounding_reject`, `bleed_block`.
- [ ] Re-run Phase 1 baseline on 062000 — bleed strings must not persist.

---

## Files to touch

| Path | Role |
|------|------|
| [src/psm-core/src/service.ts](../../../src/psm-core/src/service.ts) | Post-decode guards |
| [src/psm-core/src/psm-model-runtime.ts](../../../src/psm-core/src/psm-model-runtime.ts) | Token budget |
| [psm-model/src/psm_model/remember_server.py](../../../psm-model/src/psm_model/remember_server.py) | Server decode limit |
| `src/psm-core/src/grounding-guards.ts` | New (proposed) |
| [benchmark/locomo/src/ingest-quality-check.ts](../../../benchmark/locomo/src/ingest-quality-check.ts) | Bleed regex source |

---

## Exit criteria

- [ ] Re-running 062000-class ingest on D1:5-style turns does **not** persist bleed strings.
- [ ] `grounding_reject` and `bleed_block` appear in ingest debug logs when appropriate.
- [ ] Prod and eval use same `max_new_tokens` (384).

---

## Results

_(None yet.)_
