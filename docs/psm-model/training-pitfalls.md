# PSM training pitfalls — read before any curriculum or RunPod launch

**Purpose:** Stop repeating the same mistakes across nano → 50M gate → prod-memory.  
**When to read:** Before building curriculum, before `train-prod-memory`, before answering "why did we train like this?"  
**Companion:** [docs/plans/psm-production-memory/00-north-star.md](../plans/psm-production-memory/00-north-star.md) (product task), [training-playbook.md](training-playbook.md) (commands)

---

## The product task (non-negotiable)

```text
remember({ llmResponse })   ← assistant text: plans, handoffs, agent summaries, tool narratives
  → store | ignore
  → memory.content + facts[] grounded in llmResponse
  → guards reject ungrounded store
```

**Not:** short `User:` preference sentences, gate probe templates, LoCoMo QA labels, or dialogue-turn user chat.

If curriculum rows don't look like **Cursor agent output / markdown handoffs**, they are wrong for prod-memory — even if gate eval passes.

---

## Pitfall 1 — Trained on user utterances, evaluated on llmResponse

**What happened (50M gates):** `direct_probes.jsonl` and most gate mass use `"User: I prefer SQLite..."` (50–120 chars). Model learned **user preference → memory** autocomplete.

**Symptom:** High gate parse/action; prod `effective_stored` 1/10; outputs like `"User prefers prefers about memory choureaptabin..."`.

**Rule:** Prod-memory curriculum must use `operation: remember_llm_response` and **assistant-role** content. Audit with:

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -c "
import json; from prod_memory.row_validation import remember_target_from_input
from psm_model.data.rows import infer_row_task
rows=[json.loads(l) for l in open('PATH.jsonl') if l.strip()]
storage=[r for r in rows if infer_row_task(r)=='storage']
lens=sorted(len(remember_target_from_input(r['input'])) for r in storage)
print('storage n=', len(storage), 'p50 chars=', lens[len(lens)//2] if lens else 0)
"
```

**Targets:** p50 ≥ **600** chars for storage rows (handoff scale); v3 teacher hit **1126** p50.

---

## Pitfall 2 — Curriculum inputs too short (synthetic/heuristic rows)

**What happened (v1, v5):** Hand-written `MEMORY_SUMMARY` one-liners and tiny fixture copies → storage p50 **149–163 chars**.

**What happened (v5 eval fixtures):** p50 **224 chars** — still short smoke tests, not real handoffs.

**Real llmResponse scale:** Agent plans/handoffs **600–4800+ chars** (v3 teacher p90 **2411**, max **4798**).

**Rule:** Do not ship prod-memory train from curriculum with storage p50 < 500. Use **teacher-labeled** session exports or chunked handoffs, not heuristic summaries alone.

---

## Pitfall 3 — Mixing recall into storage-fix runs

**What happened (v5):** Copied Gate 5 regression anchor (`recall ×50`) into a run meant to fix storage. Result: **71% recall_plan**, **9% storage** → train metrics almost all `recall_plan`.

**What Gate 5 is for:** After storage passes Gate 4, **add** recall while keeping 65–75% storage mass. See [training-playbook.md](training-playbook.md) Gate 5.

**Rule for prod-memory iteration:**

| Run type | Curriculum | After train |
|----------|------------|-------------|
| **Storage fix** | **100% storage** (teacher v3), no recall/context rows | Prod grounding eval |
| **Regression check** | **Eval only** — gate4 expanded ×2 | Do not train on this mix |
| **Recall refresh** | Separate run, only if storage eval + regression pass | Dual gate eval |

**Never** use `recall_copies: 50` in a prod-memory storage micro-run.

---

## Pitfall 4 — Optimized gate metrics, not prod grounding

**What happened:** Weeks Gate 4–6 → parse ≥95%, recall Hit@k high; prod `effective_stored` stuck at **1/10**.

**Rule:** Gate pass is **necessary**, not **sufficient**. Promotion requires [phase-6 bar](../plans/psm-production-memory/phase-6-promotion-ship.md): prod suites ≥85% grounding, not dual gate alone.

---

## Pitfall 5 — action_loss_weight=0 on prod-memory train

**What happened:** Prod train uses LM loss only; workflow fixtures always **ignore** at eval; no gradient on store vs ignore boundary.

**Rule:** Storage-fix runs set `--action-loss-weight 1.0` and `--sampling action_balanced` unless explicitly ablating.

---

## Pitfall 6 — Train metric ≠ ship metric

**Train:** LM loss on perfect teacher JSON.  
**Ship:** `effective_stored` = correct action + **grounded content** + guard accept.

**Symptom:** Loss → 0.0002 (v5) while `effective_stored` unchanged. Model learns JSON shape, not extractive grounding.

**Rule:** Labels need `facts[]` with `evidence_text` from llmResponse spans (v3 teacher: **1357/1494** rows). Heuristic summaries without facts (v1: **8/203**) are insufficient.

---

## Pitfall 7 — v4-style fail-copy drilling

**What happened:** ×40 copies on failing fixtures → **0/10**, cursor_shaped regression.

**Rule:** ≤5 copies per failing fixture; suite-focused micro-runs; abort if cursor_shaped drops vs 058000 baseline.

---

## Pitfall 8 — Confusing context window with input length

**4096 context-length** in train script ≠ training on long llmResponse.

- v5: 4096 window + **149 char** median inputs + recall rows (no assistant handoff at all) = wasted capacity + wrong task mix.
- Long-context product path = **Phase 2 chunking** (600–1200 tokens per chunk), not one 4k monolithic row.

**Rule:** Match `--context-length` to actual input p90; prefer chunked handoffs over padding short rows in a huge window.

---

## Pitfall 9 — Nano / structured-model detour (10M)

**What happened:** Nano structured classifier path explored; PSM product still needs **generative StorageDecision JSON** at 50M for `remember()`.

**Lesson:** Classification-only or tiny encoders don't replace decoder path for prod; gate curriculum still dominated by short user-turn data.

---

## Pitfall 10 — No pre-flight checklist (agents repeat nearest builder)

**What happened:** Each session copies last builder (`build_prod_extraction_v1` → v4 fail-copy → v5 gate5 anchor) without validating task mix or input length.

**Rule — pre-flight before every prod-memory GPU run:**

- [ ] Read this file
- [ ] Curriculum storage p50 ≥ 600 chars (or document why chunk-smoke)
- [ ] Storage rows ≥ 80% of curriculum (storage-fix runs: 100%)
- [ ] Facts populated on ≥ 50% storage rows (teacher v3: ~91%)
- [ ] `action_loss_weight` set for storage runs
- [ ] Resume stem documented; eval fixtures unchanged
- [ ] No recall rows in storage-fix mix
- [ ] Success metric = prod `effective_stored`, not train loss

---

## Checkpoint policy (prod-memory)

| Donor | When to use |
|-------|-------------|
| **058000** gate stem | Default; best dual-gate balance |
| **048000** | A/B if 058000 recall mass blocks storage fine-tune |
| **228000** | JSON shape debug only — prod grounding also ~1/10 at baseline |

Do not promote prod-memory stems until prod eval ≥ Phase 6 bar.

---

## History (why we keep circling)

| Era | Mistake | Result |
|-----|---------|--------|
| Nano / 10M | Structured heads vs generative prod path; small data | Did not ship prod remember() |
| Gate 2–3 | Short user-turn probes | JSON works on 5 cases |
| Gate 4–6 | Expanded gate mass, not llmResponse | Parse high; prod grounding flat |
| Prod v1 | Heuristic labels, recall ×50, short rows | Colab-ready but wrong shape |
| Prod v3 teacher | Better labels (1126 p50, facts) | Train lateral — still mixed recall + no action loss |
| Prod v4 | ×40 fail-copy | Regression 0/10 |
| Prod v5 | Gate5 anchor in storage-fix run | 9% storage steps; lateral 1/10 |

**Next (v6):** Storage-only teacher v3, action loss, regression eval after — not in mix.

---

## References

- Prod eval: `psm-model/prod-memory/fixtures/cases.json`
- Teacher labels: `prod-extraction-v2.jsonl` (v3 content)
- Failure mining: [phase-5-failure-mining-2026-06-21.md](../plans/psm-production-memory/phase-5-failure-mining-2026-06-21.md)
- Handoff: [2026-06-20-end-of-day-handoff.md](2026-06-20-end-of-day-handoff.md)
