# Plan 2: LoCoMo Tests That Return Answers

**Goal:** End-to-end LoCoMo pipeline that reports **LLM-judge answer accuracy**, not just evidence `dia_id` hit rate.

**Current gap:** `evaluate.js` is retrieval-only (`answer_judgment: "not_evaluated"`). `answer-evaluate.ts` exists but uses **GGUF/Qwen** for PSM recall, not **058000 `PsmModelRuntime`**. Our n=25 PSM-format ingest stored rows but **memory content was garbage** (training bleed), so even tag-based “hits” are meaningless.

---

## Target pipeline

```text
LoCoMo turns
  → ingest (PSM probe format, PsmModelRuntime @ 058000)
  → SQLite memory DB
  → PsmService.context() / recall per question
  → exact DB-backed context bullets
  → answer model (OpenRouter)
  → judge model (OpenRouter)
  → answer_accuracy + failure buckets
```

Headline metric: **`answer_accuracy`** (LLM-as-judge vs gold).

Diagnostics retained: `evidence_hit_at_k`, `answer_context_hit_at_k`, `failure_bucket`.

---

## What broke in n=25 (must fix first)

| Layer | n=25 result |
|-------|-------------|
| Ingest format | PSM probe text ✅ (16/25 wrote) |
| Memory content | ❌ garbled (checkpoints, PowerShell, CPU training bleed) |
| `evaluate.js` | Tag hit on `locomo_dia_id` — not content quality |
| Answers | **Never run** |

**Blocker:** Storage quality on LoCoMo chat before answer eval is meaningful.

---

## Phase 0 — Storage quality gate (ingest must pass)

**Before any answer eval**, add `ingest-quality-check.ts` (benchmark-only):

Per ingested turn with gold evidence in QA set, assert:

| Check | Example |
|-------|---------|
| No curriculum bleed | content must not match `/checkpoint\|PowerShell\|gate datasets\|nvidia-smi/i` |
| No raw JSON / wrapper | no `{`, no `Current utterance:` |
| Speaker names | `Caroline` / `Melanie`, not generic `User prefers` |
| Gold fact probe (manual set) | D1:3 → content mentions LGBTQ support group + resolved date |
| `memory_facts` rows | >0 when explicit facts in utterance |
| Parse failures | `failed=0` on smoke |

**Smoke commands:**

```powershell
npm run build
node dist/benchmark/locomo/src/ingest-psm-model.js `
  --input-format psm --device cuda --limit 20 `
  --checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-058000.pt `
  --db benchmark/locomo/results/locomo-ingest-smoke.db

node dist/benchmark/locomo/src/ingest-quality-check.js `
  --db benchmark/locomo/results/locomo-ingest-smoke.db --limit 20
```

**Exit non-zero** until quality gate passes. Do not run answer eval until this is green.

### Phase 0b — Debug why 058000 emits garbage on LoCoMo

Investigate (in order):

1. **Raw tagged decode** — log `remember_server` `raw` + `parsed` for D1:3, D1:12, D1:5
2. **Prompt token dump** — `render_storage_prompt` + `to_model_input` for PSM-format turn
3. **Training leakage** — grep curriculum for phrases appearing in DB rows
4. **`flag_conflict` spam** — 11/25 conflicts on casual chat; may need LoCoMo-specific threshold or empty `memory_store` on ingest (`includeExistingMemories: false` already set — verify conflict path)
5. **Tagged parser repair** — `apply_product_boundary` rewriting valid output to junk?

**Fix options (pick after root cause):**

- Tighten prod prompt for chat (shorter inference prompt — ties to KV plan)
- LoCoMo ingest: skip `flag_conflict` route for first pass (benchmark adapter only)
- Add post-decode validation (reject content not grounded in input utterance — extend `ingest-quality-check`)
- Fine-tune slice on LoCoMo-shaped `User: Speaker said "..."` rows (only if decode is correct but action wrong)

---

## Phase 1 — Wire PSM 50M into answer evaluation

**New file:** `benchmark/locomo/src/answer-evaluate-psm-model.ts`

Clone `answer-evaluate.ts` but replace:

```typescript
// OLD
const runtime = new NodeLlamaRuntime({ modelPath: options.psmModel, ... });

// NEW
const runtime = new PsmModelRuntime({
  checkpoint: options.checkpoint,
  python: options.python,
  repoRoot: options.repoRoot,
  device: options.device,
  outputFormat: "tagged"
});
```

Requirements:

- `service.context()` uses **058000** for `recall_plan` + ranking (already in `PsmService`)
- Context rendering: **exact DB-backed items** from `context_items` (benchmark v1 — no free-form PSM context rewrite)
- Embeddings: `TransformersEmbeddingRuntime` (same as today) for hybrid rank

**CLI flags:** `--checkpoint`, `--python`, `--device`, `--repo-root` (mirror `ingest-psm-model.ts`)

**Env:** `OPENROUTER_API_KEY` for answer + judge models (existing).

---

## Phase 2 — Unified LoCoMo runner

**New script:** `benchmark/locomo/run-locomo-psm-model.ps1`

```powershell
# 1. ingest (PSM format)
# 2. ingest-quality-check (fail fast)
# 3. evaluate.js (retrieval diagnostic)
# 4. answer-evaluate-psm-model.js (answers + judge)
```

Outputs:

| File | Content |
|------|---------|
| `locomo-psm-model-step-{S}-n{N}.db` | memories |
| `...-ingest-summary.json` | stored/ignored/failed |
| `...-retrieval-results.json` | hit@k diagnostic |
| `...-answer-results.json` | **answer_accuracy**, per-Q records |
| `...-answer-debug.md` | failure buckets |

---

## Phase 3 — Answer eval scope for early tests

Start small, same as ingest:

| Stage | Ingest turns | Answer Q limit | Pass bar |
|-------|--------------|----------------|----------|
| Smoke | 20 | 20 (answerable-only) | `failed=0`, quality gate, **accuracy > 0** |
| Dev | 100 | 50 | accuracy > 10% (baseline) |
| Reportable | full conv-26 session 1 | all answerable Qs | compare vs Mem0/Zep docs |

Add `--answerable-only` to answer eval: only questions whose gold `evidence` ⊆ ingested `locomo_dia_id` set (fair partial-ingest eval).

---

## Phase 4 — Retrieval improvements (if answers still fail)

After storage is good, if `failure_bucket=retrieval_miss` dominates:

1. Use `PsmService.recall()` not just lexical `rankMemories` in `evaluate.js`
2. Ensure `locomo_dia_id` tags flow into `context_items.source_ids`
3. Hybrid rank: vector + BM25 + entity (product plan)
4. Windowed ingest (`window-size 2`) — already in adapter

---

## Phase 5 — CI / gate integration

| Gate | When |
|------|------|
| Dual eval @ checkpoint | ship weights (existing) |
| LoCoMo ingest quality smoke n=20 | before LoCoMo answer report |
| LoCoMo answer smoke n=20 | nightly / pre-release |
| Full LoCoMo | manual until accuracy stable |

Do **not** block weight ship on full LoCoMo until Phase 0–3 pass.

---

## Files to create / change

| File | Action |
|------|--------|
| `benchmark/locomo/src/ingest-quality-check.ts` | **new** — storage content gates |
| `benchmark/locomo/src/answer-evaluate-psm-model.ts` | **new** — 50M recall + answer + judge |
| `benchmark/locomo/run-locomo-psm-model.ps1` | **new** — full pipeline |
| `benchmark/locomo/src/ingest-psm-model.ts` | log raw model output optional `--debug-raw` |
| `benchmark/locomo/src/evaluate.ts` | optional: use `PsmService.recall` path |
| `docs/locomo-benchmark-plan.md` | link to this plan, update status |

---

## Dependencies

| Dependency | Plan |
|------------|------|
| KV cache | speeds ingest/remember; not required for answer correctness |
| PSM input format | ✅ done (`--input-format psm`) |
| 058000 on disk / HF | required |
| OpenRouter API key | required for answer + judge |
| Storage quality | **blocking** |

---

## Success definition

1. **Ingest quality smoke (n=20):** `failed=0`, no bleed, D1:3 memory contains support group + date.
2. **Answer smoke (n=20, answerable-only):** `answer-evaluate-psm-model` writes `answer_accuracy` > 0 with non-empty `generated_answer` per row.
3. **Example row passes end-to-end:**
   - Q: *When did Caroline go to the LGBTQ support group?*
   - Gold: *7 May 2023*
   - Generated: semantically correct
   - Judge: `correct: true`
4. Summary JSON includes `metric: "LoCoMo LLM-as-judge answer accuracy"` (not `not_evaluated`).

---

## Immediate next actions (ordered)

1. Add `--debug-raw` to ingest; re-run **3 turns** (D1:3, D1:5, D1:12); inspect raw tagged output.
2. Implement `ingest-quality-check.ts`; run on existing psmfmt n=25 DB (expect **fail** — documents baseline).
3. Fix storage root cause from Phase 0b.
4. Implement `answer-evaluate-psm-model.ts`; run `--limit 10 --answerable-only` on fixed ingest DB.
5. Wire `run-locomo-psm-model.ps1` for repeatable smokes.
