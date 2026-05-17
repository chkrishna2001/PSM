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

Status: partially implemented.

Use windowed LOCOMO ingestion:

- current turn
- previous/next `2` turns
- `sample_id`
- `session`
- `dia_id`
- speaker
- image query/caption
- QA hints when the current turn is gold evidence

Expected memory behavior:

- Preserve speaker names, not `User`.
- Preserve source `dia_id` tags.
- Normalize answerable facts when local context permits.
- Preserve relative time phrases if absolute date is unavailable.
- Do not merge speakers incorrectly.

Smoke command:

```powershell
npm run build

node .\dist\benchmark\locomo\src\ingest-node.js `
  --db .\benchmark\locomo\results\locomo-window-smoke.db `
  --model "$env:LOCALAPPDATA\psm-memory\models\psm-memory-qwen-1.5b-q4_k_m.gguf" `
  --limit 100 `
  --batch-size 20 `
  --window-size 2 `
  --context-size 4096
```

Manual memory checks:

```powershell
node -e "import('@psm-memory/sdk').then(({MemoryStore})=>{ const s=new MemoryStore('benchmark/locomo/results/locomo-window-smoke.db'); const mem=s.selectMemories('locomo-conv-26',['episodic','semantic'],10000); for(const q of ['7 May 2023','lake sunrise','2022','certification','counseling or mental health','transgender woman']) { console.log('\nQUERY',q); for(const m of mem.filter(x=>(x.content+' '+x.tags).toLowerCase().includes(q.toLowerCase())).slice(0,12)) console.log(m.id, m.content, m.tags); } s.close(); })"
```

Success criteria:

- Q1 support-group memory contains `7 May 2023`.
- Sunrise memory contains `2022` or enough date-resolved context.
- Counseling/certification memory exists and is tagged with `D1:11`.
- No generic `User is...` memories for LOCOMO people.

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
