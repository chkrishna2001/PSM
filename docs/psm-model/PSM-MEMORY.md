# PSM Project Memory (facts, state, and current best)

Last updated: 2026-07-03

## Goal
Fine-tune a **Qwen 0.5B / 0.6B** LoRA adapter so the model can act as the **PSM storage model** and emit **PSM-compatible StorageDecision memory/facts** for real assistant conversations via **PSM CLI JSON** (`operation=remember_llm_response`, `conversation=[assistant]`).

**Deploy bar:** fixtures **≥7/10** `effective_stored` with a healthy full case table **AND** holdout retrieval ≥ baseline (`v5n-dpo`). `effective_stored` alone is never sufficient. LoCoMo is eval/holdout only — not a training target.

**Eval gate:** every train run must follow `.cursor/rules/psm-train-eval-gate.mdc` — full storage case table + holdout ingestion + holdout retrieval on non-training data, before promoting a checkpoint. No blind training.

## Canonical I/O (locked for v5o+)
- **Train / prod eval / CLI probe:** flatten assistant text via `remember_target_from_input()` → `PROD_STORAGE_USER_PREFIX` + text (`hf_prompts.row_messages` / `storage_inference_messages`).
- **Do not** use `to_model_input()` for probes/eval (rewrites assistant-only as `User:` — train skew).
- Helpers: `storage_llm_response_from_input()`, `storage_inference_messages_from_input()`.

## Current best state
| Checkpoint | Prod fixtures (10) | Holdout conv-30/41 | Notes |
|------------|-------------------|---------------------|-------|
| **v5q @ 768 tok** | **3/10** effective_stored (parse_valid 0.4) | not eval'd | Emits indexables[] but hallucinates enum kinds — **not promotable** |
| **v5n-dpo @ 768 tok** | **6/10** effective_stored | hit@1 **39.5%**, answer **13.3%** (n=30) | **Current prod default** |
| v5n-dpo @ 384 (HF eval on disk) | 5/10 effective_stored | — | Stale pod eval; use 768 for prod |
| v5n | 5/10 | hit@1 42.9%, answer 10.0% | Low store rate (199/1032); not better end-to-end |
| v5h | 0/10 fixtures | hit@1 41.5%, answer 13.3% | LoCoMo-contaminated — unfair holdout |

**Decision:** stay on **`hf-prod-v5n-dpo-qwen0.5b`**. Holdout matrix complete (2026-07-03); no checkpoint beats baseline on answer accuracy while meeting deploy bar.

## Holdout gate (2026-07-03, conv-30 + conv-41)
- Script path: `psm-model/scripts/runpod_holdout_gate.sh` + `_run_hf_holdout_gate_resume.py` (sequential, `GATE_SKIP_INGEST=1` reuse local DBs).
- Artifacts: `benchmark/locomo/results/holdout-gate-*-conv-30-conv-41.{db,json}` + `holdout-gate-matrix.json`.
- v5n-dpo ingest: **797/1032** stored turns → 787 episodic, 1614 facts, 1374 indexables (all auto-built; **0 model-emitted indexables[]**).
- Pods `w4cvqv33efjsks`, `7rumhwb3hu7bqz` — **EXITED**, safe to delete.

## Phase 1 diagnosis (2026-07-03)
Report: `docs/psm-model/2026-07-03-v5n-dpo-phase1-diagnosis.json`  
Script: `psm-model/scripts/_phase1_v5n_dpo_diagnose.py`

### Prod fixture case table (v5n-dpo, HF eval @384 tok)
| Case | effective_stored | Issue |
|------|------------------|-------|
| plan-01-handoff | ❌ | guard `grounding_reject_bleed` |
| plan-02-chunking | ✅ | |
| cursor-01/02 | ✅ | |
| workflow-review-pr | ✅ | |
| workflow-runpod | ❌ | JSON truncate → fail_safe |
| technical-eslint | ✅ | |
| technical-api | ❌ | malformed JSON → fail_safe (768 tok fixes per prior sweep) |
| noise-* | ✅ ignore | |

### Holdout decision audit (50 stored samples)
- **100%** emit `facts[]`; **0%** emit `indexables[]` or temporal in raw StorageDecision JSON.
- Temporal in SQLite (`memory_facts`: 98/1614) is **downstream extraction**, not model output.

### Retrieval miss taxonomy (258 Qs)
- **102** hit@1
- **112** evidence missing from top-k (mostly category 4 multi-hop)
- **44** evidence in top-k but not rank-1

### Phase 2 curriculum targets (from taxonomy)
1. SFT micro-curriculum: teach `indexables[]` + `temporal_expression` on dated assistant turns.
2. DPO/fixture boost: `plan-01-handoff` (guard bleed), `workflow-runpod` (long JSON / truncation at 768).
3. Retrieval: ranking salience for multi-evidence category-1/4 questions (infra, not adapter-only).

## What we did (high signal)
1. **v5n-dpo** DPO from v5n → **6/10 @768 tok**, parse_valid 1.0. HF: `hf-prod-v5n-dpo-qwen0.5b`.
2. **Holdout smoke wired:** single-adapter ingest/retrieval/answer on conv-30/41; matrix vs v5n/v5h.
3. **Phase 1 diagnose** bundled fixture table + 50-turn audit + retrieval slice.

## RunPod / artifact rule
Pods must not be deleted until adapter + metrics + prod eval are on HF and verified locally.

**v5q train + fixture eval complete (2026-07-03):** pod `897tbvxqu23xig` **STOPPED** (adapter + eval report on HF, pulled locally). No pods running.

## Phase 2 — v5q micro-SFT (2026-07-03, trained + evaluated — REGRESSION)
- Curriculum: `hf-prod-v5q-sft.jsonl` (145 rows, 139 w/ indexables). Train: 150 steps, loss 0.356, pod `897tbvxqu23xig` (**stopped**, artifacts on HF + local).
- **Fixture eval @768: 3/10** (report: `psm-model/prod-memory/results/hf-prod-v5q-qwen0.5b-prod-grounding.json`, also on HF `eval/`).
- **Wins:** `plan-01-handoff` now stores WITH model-emitted `indexables[]` (was guard-rejected on v5n-dpo); `plan-02`, `cursor-01` healthy with indexables; noise still ignored.
- **New failure mode — enum hallucination:** invalid `memory.type: "grounded"`, invalid indexable kinds `"semantic"/"fact"/"explicit"` (schema allows only mnemonic/fact_anchor/workflow) → `failed_safe` on cursor-02, workflow-review-pr, technical-eslint, noise rows.
- `workflow-runpod` parses now but hits `grounding_reject_bleed`; `technical-api` still truncates at 768.
- **Verdict: do NOT promote v5q. v5n-dpo stays prod default.**
- **Phase 3 fix:** DPO pairs penalizing invalid enum values (chosen = valid kind/type, rejected = hallucinated), or constrain curriculum to show more enum-diverse examples; plus longer max_new_tokens or shorter labels for technical-api.

## Key scripts
- `psm-model/prod-memory/scripts/build_v5q_indexables_curriculum.py` — Phase 2 curriculum
- `psm-model/prod-memory/prod_memory/build_v5q_indexables_rows.py` — anchor row builders
- `psm-model/scripts/_phase1_v5n_dpo_diagnose.py` — Phase 1 bundle
- `psm-model/scripts/_run_hf_holdout_gate.py` / `_run_hf_holdout_gate_resume.py` — holdout gate
- `psm-model/scripts/runpod_holdout_gate.sh` — pod-side gate (sequential mode)
- `psm-model/prod-memory/prod_memory/eval_hf_grounding.py` — prod fixture eval
- `benchmark/locomo/scripts/_inspect_indexables.py` — indexables layer audit

## Next action
1. **Phase 3 (v5q-dpo or v5q2 SFT):** fix enum hallucination — DPO pairs with invalid `memory.type`/indexable `kind` as rejected; keep indexables emission win.
2. Holdout gate for any new checkpoint only after fixtures ≥6/10.
3. Promote only if fixtures ≥7/10 **and** holdout answer ≥ 13.3% baseline.
