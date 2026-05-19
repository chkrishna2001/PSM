# Product-Aligned PSM Ingestion and Retrieval Fix

## Summary

We will remove benchmark-only memory-writing behavior and make LOCOMO use the same `PsmService.remember()` and `PsmService.context()` / `PsmService.recall()` paths as the real product. The goal is to stop proving a custom harness and instead harden PSM itself: temporal grounding, factual preservation, retrieval ranking, context dedupe, and reviewable training data.

Current facts from this machine:

- LOCOMO ingest currently bypasses product memory flow with direct `store.applyDecision(...)`.
- The downloaded LOCOMO DB has 5,882 episodic rows but `0` populated `source_timestamp`, `temporal_expression`, or `resolved_time` fields.
- Local real PSM data exists but is small: `52` episodic, `2` semantic, `54` decisions, `208` hook audit events.
- This is useful for debugging and review labels, but not enough for fine-tuning yet.
- Do not fine-tune now. Fix schema flow, product behavior, and evaluation first.

## Key Changes

- Remove temporary benchmark write paths:
  - LOCOMO ingest must not call `MemoryStore.applyDecision()` directly.
  - LOCOMO should adapt each turn into a normal `remember` request with `llmResponse`, `userId`, and `source`.
  - Keep benchmark-only code only for dataset loading, checkpointing, result reporting, and answer evaluation.

- Make `remember` product-grade for factual memory:
  - Always preserve source metadata when provided: `source_kind`, `source_id`, `source_timestamp`, `source_label`.
  - Add deterministic temporal normalization before storage, not only prompt instructions.
  - Add a generic `memory_facts` table for extracted facts instead of adding one column per fact type.
  - Ask PSM to return `facts[]` alongside `memory`; TypeScript validates, normalizes safe mechanics, stores, indexes, and retrieves those facts.
  - Store original relative phrase separately from resolved date/time when facts are temporal.
  - Render facts before raw prose in recall/context output.

- Add a generic facts table to avoid schema explosion:

  ```text
  memory_facts
  - id
  - user_id
  - subject
  - predicate
  - object
  - value_text
  - value_json
  - fact_type
  - confidence
  - inference_kind
  - evidence_text
  - source_memory_table
  - source_memory_id
  - source_id
  - source_timestamp
  - temporal_expression
  - resolved_time
  - created_at
  - updated_at
  ```

- Store facts as generic subject-predicate-value records:

  ```text
  subject: Caroline
  predicate: relationship_status
  value_text: single
  fact_type: profile_fact
  confidence: 0.78
  inference_kind: inferred
  evidence_text: single parent
  source_memory_id: ...
  ```

  ```text
  subject: Melanie
  predicate: activity
  value_text: painting
  fact_type: preference_or_activity
  confidence: 0.92
  inference_kind: explicit
  evidence_text: painted a sunrise
  ```

  ```text
  subject: Caroline
  predicate: career_interest
  value_text: counseling
  fact_type: profile_fact
  confidence: 0.88
  inference_kind: explicit
  evidence_text: interested in counseling
  ```

  ```text
  subject: Melanie
  predicate: event_date
  value_text: 2022
  fact_type: temporal_fact
  confidence: 0.95
  inference_kind: explicit
  evidence_text: painting of a sunrise from 2022
  ```

- Keep responsibilities explicit:
  - Memory tables store durable narrative memory.
  - Fact table stores extracted searchable facts.
  - Tags are lightweight routing hints.
  - Embeddings help semantic retrieval.
  - Ranking combines all of them.

- Layer storage after ingest:

  ```text
  episodic memory:
    Caroline is a single parent creating a family.

  memory_facts:
    Caroline | relationship_status | single | inferred | evidence="single parent"
    Caroline | family_goal | creating a family | explicit/inferred
    Caroline | parental_status | parent | explicit
  ```

- Make relationship-style queries fact-addressable:
  - Query: `What is Caroline's relationship status?`
  - Retrieval should not rely only on vector similarity between `relationship status` and `single parent`.
  - Retrieval should search facts where `subject ~= Caroline` and `predicate ~= relationship_status`, then include the source memory as supporting context.
  - PSM decides which facts exist and whether they are explicit or inferred.
  - TypeScript provides the generic structure and retrieval mechanics.

- Revise the write path around `facts[]`:

  ```text
  Input message / agent response
    -> PSM storage prompt
    -> JSON decision:
         action
         memory
         facts[]
         reasoning
    -> TypeScript validation/post-processing
    -> store memory row
    -> store fact rows linked to memory
    -> embed memory/facts
  ```

- Expected PSM storage output shape:

  ```json
  {
    "action": "store_episodic",
    "memory": {
      "content": "Caroline is a single parent creating a family.",
      "type": "episodic",
      "confidence": 0.86,
      "tags": ["family", "parenting", "relationship_status"]
    },
    "facts": [
      {
        "subject": "Caroline",
        "predicate": "parental_status",
        "value": "parent",
        "confidence": 0.9,
        "inference_kind": "explicit",
        "evidence_text": "single parent"
      },
      {
        "subject": "Caroline",
        "predicate": "relationship_status",
        "value": "single",
        "confidence": 0.75,
        "inference_kind": "inferred",
        "evidence_text": "single parent"
      },
      {
        "subject": "Caroline",
        "predicate": "family_goal",
        "value": "creating a family",
        "confidence": 0.85,
        "inference_kind": "explicit",
        "evidence_text": "creating a family"
      }
    ]
  }
  ```

- TypeScript responsibilities for facts:
  - Validate each fact shape.
  - Reject empty or low-confidence junk facts.
  - Normalize predicate names lightly.
  - Resolve temporal facts when `source_timestamp` is available.
  - Link facts to source memory ID.
  - Index facts for retrieval.

- LOCOMO ingestion adapter:
  - Extract session timestamp from `conversation.session_N_date_time` first.
  - Fall back to `event_summary.events_session_N.date`.
  - Pass `source_timestamp` as the session anchor for every turn.
  - Use `source_id = "<sample_id>:<dia_id>"`.
  - Add tags such as `locomo_sample_id`, `locomo_dia_id`, `locomo_session`, and speaker through normal product metadata.

- Retrieval and context injection:
  - Keep hybrid vector + lexical retrieval, but make exact factual matches harder to bury.
  - Add anti-repeat context selection so the same stale memories are not injected every time.
  - Penalize duplicate content and near-duplicate memories.
  - Prefer memories with matching entities, exact terms, dates, and source IDs.
  - Expose ranking diagnostics in debug output: lexical score, vector score, exact coverage, temporal score, duplicate penalty.

- Context rendering:
  - Render compact, grounded rows:

    ```text
    [episodic] Fact: Caroline attended LGBTQ+ support group on 7 May 2023.
    Source phrase: yesterday.
    Source time: 8 May 2023.
    Source id: conv-26:D1:3.
    ```

  - Do not ask the answer model to infer dates from `yesterday`, `last week`, or `last year`.
  - If `resolved_time` exists, it must be visible in injected context.

- Real-data training pipeline:
  - Treat current local data as seed/debug data only.
  - Add an export command that produces reviewable JSONL from decisions, retrieval attempts, injected context, and optional user labels.
  - Do not train on raw private transcripts by default.
  - Fine-tune only after we collect labeled failures for:
    - bad storage decision
    - missed memory
    - wrong context
    - duplicate/stale context
    - temporal grounding failure
    - should ignore

## Implementation Order

1. Clean temporary benchmark ingestion.
   - Replace LOCOMO direct `store.applyDecision()` calls with `PsmService.remember()`.
   - Remove duplicate LOCOMO storage prompt logic where it conflicts with product storage prompts.
   - Keep the Colab script generated from the same source path, not manually divergent code.

2. Add temporal grounding v1.
   - Add a deterministic temporal normalizer in core.
   - Resolve common relative phrases using `source_timestamp`: `yesterday`, `today`, `tomorrow`, `last week`, `next week`, `last month`, `next month`, `last year`, `next year`.
   - Store `temporal_expression`, `resolved_time`, and confidence on the memory row when applicable.
   - Store temporal facts in `memory_facts`, for example:

     ```text
     subject: Caroline LGBTQ support group attendance
     predicate: event_date
     value_text: 7 May 2023
     temporal_expression: yesterday
     source_timestamp: 8 May 2023
     ```

   - Preserve original content and normalized searchable facts.

3. Harden retrieval and context.
   - Add duplicate and stale-context suppression.
   - Add exact factual ranking boosts for entity + activity + date queries.
   - Search both memory rows and fact rows.
   - Render facts first and source memory second.
   - Ensure recall searches across episodic, semantic, and archival unless explicitly constrained.
   - Make context selection stable and auditable.

4. Re-ingest and benchmark.
   - Rebuild local packages.
   - Re-ingest LOCOMO from scratch through product `remember`.
   - Verify DB temporal coverage is non-zero and spot-check known evidence IDs.
   - Run retrieval-only benchmark first.
   - Run answer smoke only after retrieval/context looks sane.

5. Prepare training data.
   - Add `psm-memory review-log --jsonl` or equivalent export.
   - Add optional labels without modifying raw memories.
   - Use this machine's current `54` decisions and `208` hook events only as a pilot dataset.
   - Fine-tune only after the product produces enough labeled examples and the remaining failures are model-decision failures, not code-path failures.

## Fact Extraction Strategy

PSM extracts facts. TypeScript validates, normalizes safe mechanics, stores, and retrieves.

Short term: use prompting plus a strict schema first.

- The current storage prompt mostly asks for one memory.
- Update it to ask for `facts[]` explicitly with examples and strict JSON repair.
- This may substantially improve extraction without fine-tuning.

Fine-tuning becomes necessary only if Qwen repeatedly fails after the schema/prompt change, for example:

- Misses obvious facts like `single parent -> relationship_status: single`.
- Emits vague predicates like `personal_info`.
- Invents facts not supported by evidence.
- Fails to distinguish explicit vs inferred.
- Produces unstable JSON even after repair.
- Misses temporal/factual extraction across many real examples.

## Model Decision

Keep Qwen2.5 for now.

- Qwen2.5 remains a reasonable base because PSM is a local memory-management model, not the final answer model.
- Qwen models are generally decent at structured JSON and extraction.
- `1.5B q4` is small, and this task is harder than plain summarization: extraction, inference, schema discipline, temporal grounding, and routing.
- Do not switch models before fixing the product contract; the model was not yet asked to produce normalized fact rows, and LOCOMO was not using the product write path.

Recommended sequence:

1. Keep Qwen2.5 now.
2. Implement `facts[]` schema and deterministic post-processing.
3. Evaluate with LOCOMO and real hook traces.
4. If still weak, fine-tune.
5. If fine-tune still weak, compare base models.

Models worth comparing later:

- `Qwen2.5-3B-Instruct`: likely better extraction, still practical.
- `Llama-3.2-3B-Instruct`: good instruction following, worth A/B.
- `Phi-3.5-mini` / Phi small models: sometimes strong reasoning, but licensing/runtime fit must be checked.
- `Gemma 2 2B`: possible, but JSON reliability needs testing.

## Test Plan

- Unit tests:
  - `PsmService.remember()` stores provided `source_timestamp`.
  - `PsmService.remember()` stores `facts[]` into `memory_facts` linked to the written source memory.
  - Relative dates resolve correctly from source time.
  - Existing DBs migrate without dropping or rewriting user tables.
  - Duplicate context rows are suppressed.
  - Fact retrieval can answer relationship-style queries without relying only on vector similarity.
  - Exact factual query ranks the right memory above generic memories.

- LOCOMO regression tests:
  - `D1:3`: `yesterday` resolves to `7 May 2023`.
  - `D1:12`: sunrise painting retrieves `2022`.
  - `D2:1`: charity race preserves `Sunday before 25 May 2023`.
  - `D3:11`: `last week` resolves against `9 June 2023`.
  - `D5:13`: conference retrieves `July 2023`.
  - `D6:4`: museum retrieves `5 July 2023`.

- Product hook tests:
  - Codex/Gemini hooks call the same remember/context APIs.
  - Repeated prompts do not inject the same irrelevant memory every time.
  - Hook audit logs enough metadata for review without storing raw private prompts.

- Acceptance criteria:
  - LOCOMO DB after re-ingest has populated `source_timestamp` for benchmark memories.
  - Known temporal examples have either normalized content or `resolved_time`.
  - Retrieval-only Hit@K improves or remains stable while context quality improves.
  - Answer smoke no longer commonly answers `yesterday`, `last week`, or `last year` when source time is known.

## Fine-Tune Decision

Do not fine-tune yet.

Fine-tuning is justified only if, after these fixes:

- Product `remember` receives source timestamps.
- Deterministic temporal normalization is working.
- PSM has been prompted with and evaluated against the `facts[]` schema.
- Retrieval ranks the correct memory into answer context.
- Context rendering exposes normalized facts.
- PSM still repeatedly chooses wrong memory actions or bad recall plans.

If those conditions hold, fine-tune PSM Qwen on memory-management tasks only: storage decisions, fact extraction, temporal grounding, consolidation, and retrieval planning. Do not fine-tune it as a general answer model.

## Assumptions

- Existing user DBs are treated as user data. All schema changes are additive and idempotent.
- LOCOMO is a regression harness, not a separate memory product.
- Public package behavior should improve without requiring users to delete or re-create databases.
- Current local real data is useful for review and diagnostics, but too small and insufficiently labeled for a serious fine-tune.
