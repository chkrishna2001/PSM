# Phase 3 — Indexables and workflows

**Status:** Complete (2026-06-17)  
**Goal:** Recall by compact keys (`review-pr`, `grounding-bar`) and ordered procedures.  
**Depends on:** [Phase 2](phase-2-chunking-pipeline.md)

---

## Gap today

Indexables exist in nano-psm ([psm-example.mjs](../../../nano-psm/data-pipeline/src/lib/psm-example.mjs) `buildIndexables`) but **not** in prod:

- [MemoryStore](../../../src/psm-core/src/store.ts) — no indexables table
- [schema.py](../../../psm-model/src/psm_model/schema.py) — no indexable output kind
- step-062000 had **0** `memory_facts` rows in LoCoMo ingest

---

## Schema

New table `indexables` (or JSON column on memories):

```json
{
  "kind": "mnemonic | fact_anchor | workflow",
  "key": "review-pr",
  "target_memory_id": "uuid",
  "steps": ["get_pr_info", "check_target_branch", "list_changes", "review_changes"],
  "salience": 0.95,
  "reconstructive_hint": "PR review procedure from agent handoff",
  "tags": ["workflow", "review-pr"]
}
```

### Kinds

| Kind | Recall behavior |
|------|-----------------|
| `mnemonic` | Short key → memory snippet |
| `fact_anchor` | Stable fact lookup |
| `workflow` | Key → ordered `steps[]` + linked memory |

---

## Recall

Extend hybrid retrieval in [service.ts](../../../src/psm-core/src/service.ts):

1. Match query against `indexable.key` and `tags`.
2. `workflow` kind returns procedure + ordered steps.
3. Boost salience for exact key hit.

---

## Training labels

- Port nano `buildIndexables` into curriculum generator (Phase 5).
- Workflow curriculum from `.cursor/skills` + synthetic procedures.

---

## Tasks

- [x] Add `indexables` table + migrations in psm-core store.
- [x] Extend model output schema for optional `indexables[]` in storage JSON.
- [x] Synthesize and persist indexables in `remember()` path.
- [x] Extend recall for key + tag match with workflow steps.
- [x] Tests: ingest `review-pr` workflow → `recall("review-pr")` returns steps.
- [x] Facts persisted when model emits valid `facts[]`.

---

## Exit criteria

- [x] `recall("review-pr")` returns procedure + steps from ingested workflow fixture.
- [ ] Indexable recall suite passes for `review-pr` + 5 synthetic keys (Phase 6 bar).
- [x] Facts persisted when model emits `facts[]`.

---

## Results

| Check | Result |
|-------|--------|
| `indexables` SQLite table | `user_id + key` unique |
| Workflow synthesis | `review-pr` from `# Review a pull request` + 5 steps |
| `recall("review-pr")` | Returns `workflows[0].steps` length 5 |
| Facts on remember | Unit test passes with explicit fact payload |
| Python schema | Optional `indexables[]` in [`schema.py`](../../../psm-model/src/psm_model/schema.py) |

**Code:** [`src/psm-core/src/indexables.ts`](../../../src/psm-core/src/indexables.ts), store upsert/select, [`PsmService.recall`](../../../src/psm-core/src/service.ts) `workflows` field.
