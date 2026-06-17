# Phase 7 — Cursor integration

**Status:** Not started  
**Goal:** Real agent loop — remember assistant output, recall workflows in session.  
**Depends on:** [Phase 6](phase-6-promotion-ship.md)

---

## Scope

LoCoMo proved retrieval tags can work; Cursor is the product surface:

1. Hook `remember()` on assistant / agent output after each turn.
2. Ingest `.cursor/skills` as workflow indexables.
3. Session smoke: user asks “how do I review a PR?” → stored workflow recalled.

---

## Integration points

| Component | Path |
|-----------|------|
| PI plugin hooks | [src/psm-pi-plugin/src/index.ts](../../../src/psm-pi-plugin/src/index.ts) `createPsmHooks` |
| Core service | [src/psm-core/src/service.ts](../../../src/psm-core/src/service.ts) |
| CLI remember | [psm-model/src/psm_model/remember_cli.py](../../../psm-model/src/psm_model/remember_cli.py) |

---

## Workflow indexables to ingest

| Key | Source skill |
|-----|--------------|
| `review-pr` | create-pull-requests / review workflow |
| `runpod-gpu-train` | `.cursor/skills/runpod-gpu-train/SKILL.md` |
| `grounding-bar` | This plan Phase 6 thresholds (meta) |

Use `rememberChunked()` for long skill bodies (Phase 2).

---

## Tasks

- [ ] Wire hook: on assistant message complete → `remember({ llmResponse })`.
- [ ] Batch-ingest skills directory as workflow fixtures + live indexables.
- [ ] 20-turn session smoke with grounding eval after session.
- [ ] Document hook config for local dev.

---

## Eval

| Check | Target |
|-------|--------|
| Session grounding rate | ≥ 85% on stored turns |
| `recall("review-pr")` after ingest | Returns ordered steps |
| No bleed in session DB | ≤ 2% |

---

## Exit criteria

- [ ] User can ask “how do I review a PR?” and get stored workflow from prior agent output.
- [ ] 20-turn smoke passes grounding bar.

---

## Results

_(None yet.)_
