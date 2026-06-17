# Phase 0 — Freeze and governance

**Status:** Not started  
**Goal:** Stop RunPod spend and checkpoint promotion until the new bar exists.

---

## Why

Weeks of Gate 4–6 training improved **gate-shaped** metrics while **product grounding** regressed. Further gate-heavy runs waste GPU without fixing extraction.

---

## Tasks

- [ ] **Freeze training** — no new RunPod jobs until [phase-1-baseline-eval.md](phase-1-baseline-eval.md) report exists.
- [ ] **Freeze promotion** — do not ship checkpoints on dual gate alone; require Phase 6 bar.
- [ ] **Document denylist** in this file (see Ship rules below).
- [ ] Point operators to RunPod lifecycle: [.cursor/rules/runpod-auto-delete.mdc](../../../.cursor/rules/runpod-auto-delete.mdc).

---

## Ship rules (effective immediately)

### Promote only when ALL pass

1. Phase 1 prod suites: grounding ≥ 85%, bleed ≤ 2%, fail-safe ≤ 10%.
2. Indexable recall on `review-pr` + 5 synthetic keys.
3. Regression gate on **×2** expanded subset: parse ≥ 95%, action ≥ 85%.

### Denylist (do not promote)

| Condition | Reason |
|-----------|--------|
| Dual gate only | Does not measure `llmResponse` grounding |
| LoCoMo Hit@k alone | Tags can match while content is garbage |
| Curriculum bleed in ingest sample | Model memorized training templates |
| HF manifest incomplete | Missing `.pt` / `.tokenizer.json` / `.meta.json` or eval JSON |

### Training denylist (until Phase 5 approved)

| Action | Reason |
|--------|--------|
| Gate 6 ×25 expanded anchor mix | Reinforces wrong objective |
| Resume from `062000` for new curriculum | More gate-heavy direction |
| LoCoMo labels in train | Eval contamination |

---

## Files to touch

| File | Change |
|------|--------|
| [README.md](README.md) | Mark Phase 0 complete when agreed |
| [docs/psm-model/training-playbook.md](../../psm-model/training-playbook.md) | Link to this plan (Phase 6 wires full bar) |

---

## Exit criteria

- [ ] Written ship rules in this doc (above).
- [ ] Team agrees: **no gate-only promotes**.
- [ ] Phase 0 marked complete in [README.md](README.md).

---

## Results

_(None yet.)_
