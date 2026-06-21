# Phase 5 — Failure mining (2026-06-21)

**Plan:** [phase-5-curriculum-training.md](phase-5-curriculum-training.md)  
**Baseline:** `058000` gate stem (Phase 1 pinned donor)  
**Compare:** v4 @ `060000` (regression run — do not repeat recipe)  
**Eval artifacts:** `psm-model/prod-memory/results/prod-grounding-058000.json`, `prod-grounding-060000.json`

---

## Per-fixture failure modes (058000 vs v4 @ 060000)

| Fixture | Suite | 058000 effective | v4 effective | Primary failure mode @ 058000 | v4 delta |
|---------|-------|------------------|--------------|-------------------------------|----------|
| plan-01-handoff | plan_chunks | ✗ | ✗ | **Guard reject** — model stores garbage (`"User prefers prefers about memory choureaptabin..."`), overlap 0 | Same guard reject; parse cleaner |
| plan-02-chunking | plan_chunks | ✗ | ✗ | **Wrong action** — `ignore` when gold is `store` | Same |
| cursor-01-summary | cursor_shaped | ✓ | ✗ | **Pass** (only effective store on baseline) | **Regression** — fail-safe ignore |
| cursor-02-debug | cursor_shaped | ✗ | ✗ | **Guard reject** — garbage content, overlap 0 | **Wrong action** — `ignore` |
| workflow-review-pr | workflow | ✗ | ✗ | **Wrong action** — `ignore` (should store procedure) | Same |
| workflow-runpod | workflow | ✗ | ✗ | **Wrong action** — `ignore` (partial garbage in repair path) | Same |
| technical-eslint | technical | ✗ | ✗ | **Guard reject + bleed** — content mentions `checkpoint` | **Wrong action** — `ignore` |
| technical-api | technical | ✗ | ✗ | **Fail-safe** — unparseable output | Same fail-safe |
| noise-filler | noise | ✓ (correct) | ✓ (correct) | Model stores; **guard blocks** (correct product behavior) | Same |
| noise-meta | noise | ✓ (correct) | ✓ (correct) | Model stores; **guard blocks** | Same |

**Aggregate:** 058000 **1/10** effective_stored · v4 **0/10** (lost cursor-01).

---

## Suite rollup (Phase 1 metrics)

| Suite | 058000 effective | v4 effective | Dominant blocker @ 058000 |
|-------|------------------|--------------|---------------------------|
| plan_chunks | 0/2 | 0/2 | Ungrounded store (1) + ignore-on-store (1) |
| cursor_shaped | 1/2 | 0/2 | Guard reject on debug case; v4 broke summary |
| workflow | 0/2 | 0/2 | **Action selection** — both ignore |
| technical | 0/2 | 0/2 | Fail-safe + guard reject (bleed pattern) |
| noise | 0/2 stored (correct) | 0/2 stored (correct) | Guards working; action_match passes |

Phase 5 exit bar remains: **≥15pp** on **plan + workflow** suites vs Phase 1 baseline (currently 0% effective each).

---

## Top 3 fix levers (plan-aligned)

| # | Lever | Evidence | Phase 5 action |
|---|-------|----------|----------------|
| 1 | **Curriculum — grounded extraction, not fail-copy** | Model emits gate-template garbage on plan/workflow; v4 ×40 made cursor_shaped worse | **v5 suite micro-run** using [v1 mix philosophy](phase-5-curriculum-training.md) (plan handoffs ×15, recall ×50 anchor), **≤5 copies** on eval fixtures only. First focus: **plan_chunks**. |
| 2 | **Action selection on procedures** | workflow-review-pr + workflow-runpod both `ignore` @ 058000 and v4 | Include workflow bucket in **second** micro-run (after plan_chunks); v1 profile uses workflow ×10. |
| 3 | **Output format / parse** | Many cases: malformed facts, missing `reasoning`, fail-safe on technical-api | Keep expanded ×2 + recall ×50 regression mass; do **not** relax guards (Phase 4 — guards are saving us). |

**Not the bottleneck:** Guard reject on noise (working as designed). LoCoMo / Gate 6 mass (non-goals per [00-north-star.md](00-north-star.md)).

---

## v5 micro-run sequence (Phase 5 iteration)

Per [phase-5-curriculum-training.md](phase-5-curriculum-training.md) + handoff constraints:

| Run | Focus suite | Builder | Target steps | Success gate |
|-----|-------------|---------|--------------|--------------|
| **v5a** | plan_chunks | `build_prod_extraction_v5_suite_micro.py --focus-suite plan_chunks` | 058000 → ~59200 (1200 steps) | plan_chunks effective ≥1/2; **no cursor_shaped regression** |
| v5b | workflow | same builder, `--focus-suite workflow` | after v5a review | workflow effective ≥1/2 |
| v5c | technical | `--focus-suite technical` | after v5b | technical parse_valid ↑ |

Resume always from **`real-v3-50m-full-v2-step-058000`** (north-star checkpoint policy).

Train infra: Phase 5 doc lists Colab; prod-memory RunPod path is also wired (`runpod_ctl.py train-prod-memory`). Either works — **eval bar is Phase 1 harness**, not gate probes.

---

## References

- [2026-06-20 end-of-day handoff](../../psm-model/2026-06-20-end-of-day-handoff.md)
- v4 rejected recipe: `build_prod_extraction_v4_fixture_repair.py`
- v5 builder: `psm-model/prod-memory/prod_memory/build_prod_extraction_v5_suite_micro.py`
