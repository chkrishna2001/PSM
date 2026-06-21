# PSM Production Memory — Plan Index

**Status:** Active (2026-06-17)  
**Owner:** PSM team  
**Supersedes:** Scattered gate/LoCoMo handoffs for *ship decisions* — legacy docs remain as background only.

This folder is the **single source of truth** for making PSM production-ready: grounded extraction from `llmResponse`, chunking, indexables/workflows, new promotion bar, then optional training.

---

## Phase status

| Phase | Doc | Status | Depends on |
|-------|-----|--------|------------|
| — | [00-north-star.md](00-north-star.md) | Reference | — |
| 0 | [phase-0-freeze-governance.md](phase-0-freeze-governance.md) | **Not started** | — |
| 1 | [phase-1-baseline-eval.md](phase-1-baseline-eval.md) | **Complete** | Phase 0 |
| 2 | [phase-2-chunking-pipeline.md](phase-2-chunking-pipeline.md) | **Complete** | Phase 1 |
| 3 | [phase-3-indexables-workflows.md](phase-3-indexables-workflows.md) | **Complete** | Phase 2 |
| 4 | [phase-4-guardrails-prod.md](phase-4-guardrails-prod.md) | **Complete** | Phase 1 |
| 5 | [phase-5-curriculum-training.md](phase-5-curriculum-training.md) | **v5 micro-run ready** ([failure mining](phase-5-failure-mining-2026-06-21.md)) | Phase 2, 3, 4 |
| 6 | [phase-6-promotion-ship.md](phase-6-promotion-ship.md) | **Not started** | Phase 5 |
| 7 | [phase-7-cursor-integration.md](phase-7-cursor-integration.md) | **Not started** | Phase 6 |
| 8 | [phase-8-scale-optional.md](phase-8-scale-optional.md) | **Gated** | Phase 5 plateau |

Update **only this table** when a phase completes.

---

## Execution order

```
Phase 0 → Phase 1 → Phase 4 (parallel with Phase 2) → Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 7
                                                                              └→ Phase 8 (only if Phase 5 plateaus)
```

**No RunPod for Phase 5** — use HF dataset + Colab notebook (`prod-memory/notebooks/prod-extraction-v1-colab.ipynb`).

---

## Glossary

| Term | Meaning |
|------|---------|
| `remember_target` | Text passed to `remember()` as `llmResponse` (assistant reply, plan, summary) — **not** a human user utterance |
| `content_grounding` | Stored `memory.content` / facts overlap key tokens from `remember_target` |
| `curriculum_bleed` | Stored content contains PSM-internal tokens (checkpoint, fact parser, gate datasets, …) |
| `fail_safe_ignore` | Product boundary forced `ignore` because model output was unparseable |
| `indexable` | Compact recall key (e.g. `review-pr`) pointing at a memory row |
| `workflow` | Indexable kind with ordered `steps[]` for procedures |

---

## Related docs (background only)

| Doc | Use |
|-----|-----|
| [docs/psm-model/training-playbook.md](../../psm-model/training-playbook.md) | RunPod, gates, checkpoints |
| [docs/psm-memory-product-plan.md](../../psm-memory-product-plan.md) | Product architecture vision |
| [docs/psm-model/2026-06-15-conversation-memory-handoff.md](../../psm-model/2026-06-15-conversation-memory-handoff.md) | Gate 6 context (historical) |
| [.cursor/rules/runpod-auto-delete.mdc](../../../.cursor/rules/runpod-auto-delete.mdc) | Pod lifecycle |

---

## Artifacts (created by phases)

| Path | Phase |
|------|-------|
| `psm-model/prod-memory/results/prod-grounding-baseline.json` | 1 |
| `psm-model/prod-memory/data/prod-extraction-v1.jsonl` | 5 |
| `src/psm-core/src/segment-remember.ts` | 2 |
| `src/psm-core` indexables schema | 3 |
