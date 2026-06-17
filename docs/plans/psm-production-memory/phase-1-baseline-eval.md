# Phase 1 — Baseline eval (local, cheap)

**Status:** Complete (baseline run 2026-06-17)  
**Goal:** Measure HF checkpoints honestly on prod-shaped input before any training.  
**Depends on:** [Phase 0](phase-0-freeze-governance.md)

---

## Question this phase answers

> Can we ship `062000` (or `058000`) for Cursor `remember()` today?

Expected answer: **no** — document why with numbers.

---

## Eval suites (held-out)

| Suite | Description | Fixtures |
|-------|-------------|----------|
| **plan_chunks** | Markdown handoffs, multi-section plans | Canonical robust-plan chunk + 3 synthetic handoffs |
| **cursor_shaped** | Raw `llmResponse` assistant text | Agent summaries, tool output narratives |
| **workflow** | Procedure text | `review-pr` steps, RunPod train skill excerpt |
| **technical** | Short rules/preferences | Lint rules, API conventions |
| **noise** | Should ignore | Filler, meta-only, empty |

**Rule:** No LoCoMo QA labels in train; LoCoMo may be a separate diagnostic suite, not the primary bar.

---

## Metrics (per suite and aggregate)

| Metric | Definition |
|--------|------------|
| `content_grounding_rate` | Fraction of `store` rows where key tokens from input appear in `memory.content` or `memory_facts` |
| `curriculum_bleed_rate` | Fraction matching bleed regex (see [ingest-quality-check.ts](../../../benchmark/locomo/src/ingest-quality-check.ts)) |
| `fail_safe_ignore_rate` | Fraction of turns ending in fail-safe `"model output unparseable; storing nothing"` |
| `parse_valid_rate` | Valid JSON action on regression subset only |
| `action_accuracy` | store/ignore/update correct vs gold (regression subset) |

Run at **prod** `max_new_tokens` (128 today) and document delta vs raised budget (384, Phase 4).

---

## Implementation

### New harness

Location: [`psm-model/prod-memory/`](../../../psm-model/prod-memory/) — **isolated from legacy `psm_model` gate evals and `benchmark/locomo`.**

```
psm-model/prod-memory/
  fixtures/cases.json
  prod_memory/eval_grounding.py
  prod_memory/grounding.py
  results/
```

**Eval path:** `remember_storage_decision` via `psm_model.generate` + `apply_product_boundary` (same decode/repair as prod). Guard metrics mirror [`grounding-guards.ts`](../../../src/psm-core/src/grounding-guards.ts).

**Full prod path** (SQLite + `PsmService`): optional integration later; ship bar uses model + guard projection first.

### Checkpoints to run

| Checkpoint | HF source | Notes |
|------------|-----------|-------|
| `058000` | `subbu83/psm-50m-mixed-v1-run` | Preferred resume base |
| `062000` | same repo | Current worst-case bleed reference |

### Output artifact

`psm-model/prod-memory/results/prod-grounding-baseline.json`

```json
{
  "checkpoint": "062000",
  "timestamp": "...",
  "suites": {
    "plan_chunks": { "content_grounding_rate": 0.0, "..." : "..." }
  },
  "aggregate": { }
}
```

---

## Tasks

- [x] Create fixture files for each suite.
- [x] Implement grounding overlap scorer (token / keyword overlap).
- [x] Port bleed regex from LoCoMo ingest quality check.
- [x] Run locally on `058000` and `062000`.
- [x] Paste summary into **Results** below.
- [ ] Update [README.md](README.md) phase status.

---

## Files to touch

| Path | Role |
|------|------|
| [`psm-model/prod-memory/`](../../../psm-model/prod-memory/) | Eval harness + fixtures (isolated tree) |
| [`psm-model/prod-memory/prod_memory/eval_grounding.py`](../../../psm-model/prod-memory/prod_memory/eval_grounding.py) | CLI entry |
| [src/psm-core/src/grounding-guards.ts](../../../src/psm-core/src/grounding-guards.ts) | Prod guards (TS source of truth) |

---

## Exit criteria

- [x] `prod-grounding-baseline.json` exists for both checkpoints.
- [x] Report answers “Can we ship 062000?” with evidence.
- [x] Metrics defined here are reused by Phases 4–6.

---

## Results

**Verdict: ship neither 058000 nor 062000** for prod `remember()` (effective store rate 10%, ship bar is 85%).

Artifacts:
- [`psm-model/prod-memory/results/prod-grounding-baseline.json`](../../../psm-model/prod-memory/results/prod-grounding-baseline.json)
- [`prod-grounding-058000.json`](../../../psm-model/prod-memory/results/prod-grounding-058000.json)
- [`prod-grounding-062000.json`](../../../psm-model/prod-memory/results/prod-grounding-062000.json)

Run: CPU, `max_new_tokens=384`, guards projected in eval (same logic as psm-core).

### Aggregate (effective_stored = model store that passes guards)

| Checkpoint | Effective stored | Grounding (of stored) | Bleed | Fail-safe | Parse valid | Guard reject | Action match |
|------------|------------------|----------------------|-------|-----------|-------------|--------------|--------------|
| 058000 | **1 / 10** | 100% (1/1) | 0% | 10% | 90% | 50% | 30% |
| 062000 | **1 / 10** | 100% (1/1) | 0% | 0% | 100% | 40% | 30% |

### Per suite (effective_stored)

| Suite | 058000 | 062000 |
|-------|--------|--------|
| plan_chunks | 0 / 2 | 0 / 2 |
| cursor_shaped | 1 / 2 | 1 / 2 |
| workflow | 0 / 2 | 0 / 2 |
| technical | 0 / 2 | 0 / 2 |
| noise | 0 / 2 (correct) | 0 / 2 (correct) |

### What worked

- **cursor-01-summary** (`cursor_shaped`): only case that effectively stored on both checkpoints with grounded content.

### What failed

- **plan_chunks**: model stores curriculum garbage (e.g. `"User prefers prefers about memory chourcing..."`); guards correctly reject.
- **workflow**: model ignores both `review-pr` and RunPod procedure (0 stores).
- **technical**: 058000 fail-safe on one case; 062000 ignores both.
- **noise**: model *wants* to store (2/2) but guards block all — action_match still passes because nothing persisted.

### LoCoMo diagnostic (historical, not ship bar)

step-062000 n=25: answer accuracy 0%, Hit@1 33%, Hit@k 89%, 11/13 ignores fail-safe.
