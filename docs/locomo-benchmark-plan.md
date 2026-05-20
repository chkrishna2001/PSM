# LOCOMO Benchmark Recovery Plan

## Goal

Produce a defensible LOCOMO score for PSM Memory that can be compared against memory tools such as Mem0, Zep, Letta, LangMem, and Memori.

The benchmark must show PSM as a memory system:

```text
conversation -> PSM memory ingestion -> PSM recall/context -> answer model -> judge -> LOCOMO answer accuracy
```

The headline score should be LLM-judge answer accuracy. Retrieval hit rates remain diagnostics.

## Current Diagnosis

The first full Colab run proved the pipeline can ingest the full dataset and checkpoint reliably:

- `5882` LOCOMO turns ingested.
- `1982` LOCOMO questions evaluated for retrieval.
- Retrieval-only result from the initial DB:
  - Evidence Hit@1: `14.98%`
  - Evidence Hit@3: `24.62%`

That result is not a comparable market benchmark because public LOCOMO results are usually answer-generation scores judged by an LLM.

The later answer-evaluation work exposed these issues:

1. Raw single-turn ingestion loses answerable context.
   - Example: storing `yesterday` without enough date/session grounding.
   - Example: related answer is sometimes in a nearby turn rather than the gold evidence turn.

2. Windowed ingestion improves memory quality.
   - Example: PSM generated `Caroline attended a LGBTQ+ support group on 7 May 2023`.
   - Example: PSM generated `sunrise from 2022`.

3. PSM context rendering is not stable enough as a free-form generation task.
   - It sometimes returns memory-maintenance schemas like `merge_candidates`.
   - It sometimes echoes payload-like data.
   - It sometimes produces malformed JSON or non-grounded context.

4. Retrieval/context selection still misses key memories for some questions.
   - Example: counseling/certification memory exists in DB but was not surfaced for the education-fields question.

## Current Status On 2026-05-20

Do not run a full LOCOMO ingestion yet.

The latest local smoke was run against the repo-built entrypoint, not a globally installed CLI:

```powershell
npm run build

node .\dist\benchmark\locomo\src\ingest-node.js `
  --db .\benchmark\locomo\results\locomo-readiness-smoke-2.db `
  --model "$env:LOCALAPPDATA\psm-memory\models\psm-memory-qwen-1.5b-q4_k_m.gguf" `
  --limit 5 `
  --batch-size 5 `
  --window-size 2 `
  --context-size 4096
```

Corrected smoke result:

```json
{
  "seen": 5,
  "stored": 2,
  "ignored": 0,
  "failed": 3
}
```

The three failures were malformed JSON from the local 1.5B model. The two actual writes showed mixed quality:

- One row had source and temporal metadata, but it was a broad summary rather than a benchmark-critical factual memory.
- One row resolved `yesterday` from source timestamp `1:56 pm on 8 May, 2023` to `7 May 2023`, but stored the raw LOCOMO JSON payload as memory content.
- `memory_facts` had `0` rows.

The ingestion stats were also fixed after this finding. `benchmark/locomo/src/ingest-node.ts` and `benchmark/locomo/src/ingest.ts` now count parse errors as `failed`, count ignored/recall-only routes as `ignored`, and count `stored` only when `PsmService.remember()` reports written rows. This makes future smoke summaries meaningful.

Conclusion: the temporal/factual database plumbing exists, but the current LOCOMO ingestion path is not reliable enough because the benchmark was feeding structured JSON to a model trained for natural-language memory extraction. Full ingestion should wait until the benchmark-only natural-language adapter below is implemented and smoke-tested. This adapter must not leak into product code.

Follow-up implementation moved the adapter to product-shaped natural language:

- LOCOMO JSON rows are rendered as natural conversation text plus source metadata.
- QA/gold answers/evidence labels are not rendered during ingestion.
- Future turns are not rendered; only prior context is included.
- Benchmark-only runtime validation rejects raw JSON payloads, wrapper text, and generic `User` memories.
- Product SDK/service code was not changed.

Post-adapter 5-turn smoke still failed quality gates:

```json
{
  "seen": 5,
  "stored": 2,
  "ignored": 0,
  "failed": 3
}
```

The stored rows no longer copied raw JSON, but the local 1.5B model still produced malformed storage JSON on most turns and did not extract `memory_facts`. Full ingestion remains no-go until a 20-turn smoke passes the Phase 1 criteria below.

## Product Architecture For Benchmark V1

For a marketable benchmark, do not rely on unconstrained PSM prose generation until that behavior is fine-tuned.

Use this stable contract:

```text
PSM stores normalized memories.
PSM creates recall plan / ranking hints.
Semantic retrieval finds candidates.
Code renders exact DB-backed memory content.
Strong answer model answers from that context.
Strong judge model scores against gold answer.
```

This is still a PSM Memory benchmark because PSM owns storage decisions and recall planning. Exact rendering prevents hallucinated context.

The experimental context-rendering path should remain a research target:

```text
candidate memories + prompt -> clean private context bullets
```

but it should not block the benchmark v1.

## Implementation Plan

### Phase 1: Stabilize Ingestion

Status: blocked on ingestion quality.

Use product-shaped LOCOMO ingestion:

- current turn
- previous `N` turns as context only
- `sample_id`
- `session`
- `dia_id`
- speaker
- image query/caption
- no QA hints, gold answers, or benchmark labels during ingestion

Expected memory behavior:

- Preserve speaker names, not `User`.
- Preserve source `dia_id` tags.
- Normalize answerable facts when local context permits.
- Preserve relative time phrases if absolute date is unavailable.
- Do not merge speakers incorrectly.
- Never store raw LOCOMO JSON payloads as memory content.
- Write extracted `memory_facts` for benchmark-critical facts.
- Treat malformed model JSON as a failed ingest item, not as a successful store.

Required adapter changes before full ingest:

1. Parse LOCOMO input deterministically.
   - Read `sample_id`, `session`, `dia_id`, `speaker`, `text`, `query`, `blip_caption`, `img_url`, session timestamp, and nearby turns from the dataset.
   - Do not read or render `qa`, gold answers, evidence labels, categories, or benchmark questions during ingestion.
   - Keep the current turn as the only turn being remembered.
   - Render previous turns only as context for pronoun and continuity resolution.
   - Do not render future turns during ingestion; that is not the normal online product flow and can cause speaker leakage.
   - Preserve the original source id as `<sample_id>:<dia_id>`, for example `conv-26:D1:3`.

2. Build clean memory candidates before calling the store path.
   - Render the benchmark row as natural conversation text plus source metadata, not as raw JSON.
   - Use concise natural-language memory content, not the raw JSON payload.
   - Include source metadata on every candidate:
     - `source_kind = locomo_turn`
     - `source_id`
     - `source_timestamp`
     - `source_label`
     - `locomo_sample_id`
     - `locomo_dia_id`
     - `locomo_speaker`
     - `locomo_session`

3. Add deterministic temporal normalization for obvious relative dates.
   - Resolve `yesterday`, `today`, `tomorrow`, `last week`, `last month`, and `last year` when `source_timestamp` is available.
   - Preserve the original phrase in `temporal_expression`.
   - Store the resolved value in `resolved_time`.
   - Example: `session_timestamp = 1:56 pm on 8 May, 2023` and text contains `yesterday` -> `resolved_time = 7 May 2023`.

4. Add deterministic benchmark-only rendering, not benchmark-answer extraction.
   - Convert structured LOCOMO rows into natural-language turns that resemble product input.
   - Do not create facts from gold answers.
   - Do not use QA evidence links to decide what to store.
   - Explicit event/activity facts:
     - Let PSM extract these from natural conversation text when the speaker explicitly states an activity/event.
     - Speaker location, preference, career interest, relationship/status, family detail, project, and workflow facts when explicitly stated.
   - Image facts:
     - Render image query and caption as natural context fields so PSM can extract visual memories when directly supported.
   - All facts must carry evidence text and source metadata.

5. Harden model-output validation.
   - Reject memory content that starts with `{`, contains `"operation":"locomo_remember_turn"`, or otherwise looks like the raw payload.
   - Reject memory content that copies the benchmark natural-language wrapper, such as `LOCOMO benchmark conversation turn`, `Current turn to remember:`, or `Extraction guidance:`.
   - Reject generic LOCOMO memories using `User` instead of the real speaker name.
   - Reject missing or empty `memory.content`.
   - Reject rows without `source_id`, `source_timestamp`, and `locomo_dia_id` in LOCOMO ingestion.
   - If repair fails, increment `failed` and record the parse error.
   - Keep this validation inside `benchmark/locomo`; do not add LOCOMO-specific checks to product code.

6. Keep PSM ownership where it is useful.
   - PSM still owns normal storage tables, source metadata persistence, recall planning, ranking, and grounded context rendering.
   - The benchmark adapter only translates LOCOMO dataset structure into natural-language product-like input.
   - The final benchmark should be described as `PSM Memory + LOCOMO natural-language ingestion adapter + answer model + judge model`.

Smoke command:

```powershell
npm run build

node .\dist\benchmark\locomo\src\ingest-node.js `
  --db .\benchmark\locomo\results\locomo-ingest-quality-smoke.db `
  --model "$env:LOCALAPPDATA\psm-memory\models\psm-memory-qwen-1.5b-q4_k_m.gguf" `
  --limit 20 `
  --batch-size 5 `
  --window-size 2 `
  --context-size 4096
```

Manual memory checks:

```powershell
node -e "import('@psm-memory/sdk').then(({MemoryStore})=>{ const s=new MemoryStore('benchmark/locomo/results/locomo-window-smoke.db'); const mem=s.selectMemories('locomo-conv-26',['episodic','semantic'],10000); for(const q of ['7 May 2023','lake sunrise','2022','certification','counseling or mental health','transgender woman']) { console.log('\nQUERY',q); for(const m of mem.filter(x=>(x.content+' '+x.tags).toLowerCase().includes(q.toLowerCase())).slice(0,12)) console.log(m.id, m.content, m.tags); } s.close(); })"
```

Success criteria:

- `failed=0`, or every failure has a clear non-data reason worth accepting.
- No raw JSON payloads stored as memory content.
- `memory_facts` has rows.
- Q1 support-group memory contains `7 May 2023`.
- D1:3 resolves `yesterday` to `7 May 2023`.
- Sunrise memory contains `2022` or enough date-resolved context.
- Counseling/certification memory exists and is tagged with `D1:11`.
- No generic `User is...` memories for LOCOMO people.

After the 20-turn smoke passes, run a 100-turn smoke with the same checks:

```powershell
node .\dist\benchmark\locomo\src\ingest-node.js `
  --db .\benchmark\locomo\results\locomo-window-smoke-v2.db `
  --model "$env:LOCALAPPDATA\psm-memory\models\psm-memory-qwen-1.5b-q4_k_m.gguf" `
  --limit 100 `
  --batch-size 20 `
  --window-size 2 `
  --context-size 4096
```

### Phase 2: Stabilize Recall And Context

Status: needs cleanup.

Benchmark v1 should use exact DB-backed context. PSM should contribute recall planning and ranking hints, but code should render exact memory content.

Required changes:

- Keep PSM recall plan.
- Avoid free-form PSM context rewriting in benchmark v1.
- Ensure table fallback does not miss episodic memories when PSM chooses only semantic.
- Add debug output:
  - recall plan
  - candidate memory ids
  - selected/rendered context
  - evidence hit flags

Smoke answer command:

```powershell
Remove-Item .\benchmark\locomo\results\locomo-answer-results.json -ErrorAction SilentlyContinue

node .\dist\benchmark\locomo\src\answer-evaluate.js `
  --db .\benchmark\locomo\results\locomo-window-smoke.db `
  --out .\benchmark\locomo\results\locomo-answer-results.json `
  --top-k 10 `
  --limit 20 `
  --checkpoint-every 1
```

Inspect:

```powershell
node -e "const d=require('./benchmark/locomo/results/locomo-answer-results.json'); console.log(d.summary); for (const r of d.records.slice(0,10)) console.log('\nQ:',r.question,'\nGold:',r.gold_answer,'\nEvidence:',r.evidence,'\nCtx:',r.psm_context_items,'\nA:',r.generated_answer,'\nJ:',r.judgment)"
```

Success criteria:

- No hallucinated context.
- No malformed context payloads reaching the answer model.
- Context contains useful exact memories.
- Initial 20-question smoke accuracy is meaningfully above zero.

### Phase 3: Add Failure Analysis

Add a dedicated debug report for a 50-question dev set.

Each row should include:

- sample id
- category
- question
- gold answer
- gold evidence ids
- retrieved/selected memory ids
- whether gold evidence appeared in top-k
- PSM context text
- generated answer
- judge result
- failure bucket

Failure buckets:

- `missing_memory`
- `retrieval_miss`
- `bad_context_selection`
- `answer_model_error`
- `judge_error`
- `ambiguous_relative_date`
- `speaker_confusion`
- `image_context_missing`

This avoids blind full-run attempts.

### Phase 4: Full Re-Ingest

Only after smoke tests pass.

Use a fresh DB:

```powershell
Remove-Item .\benchmark\locomo\results\locomo-window-full.db -ErrorAction SilentlyContinue

node .\dist\benchmark\locomo\src\ingest-node.js `
  --db .\benchmark\locomo\results\locomo-window-full.db `
  --model "$env:LOCALAPPDATA\psm-memory\models\psm-memory-qwen-1.5b-q4_k_m.gguf" `
  --batch-size 20 `
  --window-size 2 `
  --context-size 4096
```

For Colab, use the same windowed ingest path and checkpoint to Drive.

### Phase 5: Full Answer Evaluation

Use a strong answer model and judge for the market number.

Free Nemotron is acceptable for smoke tests only. For market-facing LOCOMO, prefer:

- Claude Sonnet through OpenRouter, if available.
- GPT-class model through OpenAI or OpenRouter, if available.

Run:

```powershell
Remove-Item .\benchmark\locomo\results\locomo-answer-results.json -ErrorAction SilentlyContinue

node .\dist\benchmark\locomo\src\answer-evaluate.js `
  --db .\benchmark\locomo\results\locomo-window-full.db `
  --out .\benchmark\locomo\results\locomo-answer-results.json `
  --top-k 10 `
  --limit 0 `
  --checkpoint-every 10 `
  --answer-model "<chosen-answer-model>" `
  --judge-model "<chosen-judge-model>"
```

Generate comparison:

```powershell
node .\dist\benchmark\locomo\src\report.js `
  --psm .\benchmark\locomo\results\locomo-answer-results.json `
  --baselines .\benchmark\locomo\baselines\memory-tools.json `
  --out .\benchmark\locomo\results\locomo-comparison.md
```

## Fine-Tuning Plan

Fine-tuning is likely needed for high-quality PSM context rendering, but it should be targeted.

Do not fine-tune on broad chat answering. PSM should not become the final answer model.

Fine-tune data should target:

1. LOCOMO memory extraction
   ```text
   turn window + metadata -> normalized durable memory JSON
   ```

2. Relative date normalization
   ```text
   session date + "yesterday" -> resolved date memory
   ```

3. Multi-turn memory compression
   ```text
   image/query turn + answer turn -> one answerable memory
   ```

4. Context selection
   ```text
   question + candidate memories -> selected memory ids
   ```

5. Context rendering, later
   ```text
   question + selected memories -> clean private context bullets
   ```

The immediate benchmark should not depend on item 5.

## Tomorrow Runbook

1. Pull latest code.
2. Run `npm run build`.
3. Delete old smoke outputs.
4. Run 100-turn windowed ingest smoke.
5. Manually inspect memories for known questions.
6. Run 20-question answer smoke.
7. Create/inspect failure analysis.
8. Decide whether to run full ingest or patch retrieval first.

## Go / No-Go Criteria For Market Claim

Go only if:

- Full answer-evaluation completes.
- Result file has model names, top-k, DB version, and timestamp.
- Context is grounded in DB memory.
- No generated fake memory IDs appear.
- A failure sample audit shows no systematic benchmark leakage or cheating.
- Comparison report clearly labels:
  ```text
  PSM Memory + answer model + judge model
  ```

No-go if:

- PSM context is hallucinated.
- Correct memories exist but retrieval misses them often.
- Answer model produces noisy reasoning instead of answers.
- Score is based on retrieval-only metrics while comparing to answer-accuracy baselines.
