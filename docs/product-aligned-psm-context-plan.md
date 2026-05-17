# Product-Aligned PSM-Led Context Plan

## Summary

Build PSM around the real product contract: PSM does not answer questions; it manages memory and produces grounded context for a downstream LLM. The full flow is:

```text
prompt/question
-> PSM recall plan
-> semantic/vector search scoped by that plan
-> PSM context rendering over bounded candidates
-> plain-text memory bullets
-> host LLM answers
```

The benchmark should use the same product path and report both PSM context quality and downstream answer accuracy.

## Key Changes

- Add first-class nullable memory metadata:
  - `source_kind`, `source_id`, `source_timestamp`, `source_label`
  - `temporal_expression`, `resolved_time`, `resolved_time_confidence`
  - Keep `created_at` as "when PSM wrote this row," not necessarily when the remembered event happened.
  - Keep tags/source episodes for compatibility, but stop relying on tags as the only source/date carrier.

- Improve storage quality:
  - Storage prompts should ask PSM to preserve relative phrases and resolve them when source timestamp exists.
  - Example: store both `temporal_expression="yesterday"` and `resolved_time=2023-05-07` when `source_timestamp=2023-05-08`.
  - Product hooks should pass transcript/session provenance when available; otherwise metadata remains nullable.

- Make recall PSM-led:
  - First PSM call: create a recall plan from the prompt.
  - Plan includes `target_tables`, `ranking_hints`, `filters`, `temporal_intent`, `top_k`.
  - Semantic/vector search runs after the plan and only over PSM-selected tiers.
  - Search query uses original prompt plus PSM ranking hints.
  - If the plan is invalid or empty, fall back to `semantic + episodic` and record `plan_fallback=true`.

- Render compact grounded context:
  - Second PSM call receives only bounded candidate memories.
  - Candidate payload includes memory id, table, content, score, `created_at`, source metadata, temporal metadata, tags/source episodes.
  - PSM returns plain text bullets only, not JSON and not an answer.
  - Bullet format:

    ```text
    - [episodic | saved_at=... | source_time=... | source=...] memory content...
    ```

- Add context-length and safety controls:
  - Use separate budgets for candidate count and rendered context chars/tokens.
  - Retrieve a wider internal candidate pool, then render only top useful bullets.
  - If PSM returns too much text, trim by bullet boundaries.
  - If PSM returns maintenance/action/schema-like text, discard it and fall back to exact code-rendered candidate bullets.

## Product Interfaces

- Extend `MemoryPayload` and `MemoryRecord` with optional source/temporal fields.
- Extend insert/apply decision paths to persist metadata for episodic and semantic memories.
- Extend internal `RecallPlan` with:
  - `target_tables`
  - `ranking_hints`
  - `filters`
  - `temporal_intent`
  - `top_k`
  - `plan_fallback`
  - `raw_model_output`
- Extend `ContextItem` with optional provenance:
  - `memory_id`, `source_kind`, `source_id`, `source_timestamp`, `saved_at`
  - `temporal_expression`, `resolved_time`, `resolved_time_confidence`, `score`
- Keep `context_items[].content` compatible for existing CLI/plugin callers.

## Test Plan

- Storage tests:
  - Stores relative phrase plus resolved date when source timestamp exists.
  - Leaves `resolved_time` null when source timestamp is unavailable.
  - Preserves `created_at` separately from `source_timestamp`.

- Planning tests:
  - Specific event question targets episodic.
  - Stable profile/preference question targets semantic.
  - Historical question can target archival.
  - Invalid plan falls back to `semantic + episodic` and records fallback.

- Search tests:
  - Semantic search runs after PSM planning.
  - Search uses PSM-selected tables, not all tables by default.
  - Search query includes PSM ranking hints.
  - Candidate results preserve table, score, saved/source timestamps, and temporal metadata.

- Context rendering tests:
  - PSM receives only bounded candidate memories.
  - Plain-text bullets include source/date metadata where available.
  - PSM does not answer the user or return maintenance actions.
  - Long output is trimmed by bullet boundaries.
  - Bad output falls back to exact code-rendered candidate bullets.

- LOCOMO smoke:
  - Re-run 100-turn windowed ingest.
  - Inspect date-sensitive cases: `yesterday`, `last year`, `2022`, `7 May 2023`.
  - Run 20-question answer evaluation.
  - Report PSM plan quality, selected context quality, source/resolved-date usage, and final answer accuracy separately.

## Assumptions

- PSM owns recall planning; code chooses tables only for invalid-plan fallback.
- Semantic search means vector/embedding retrieval scoped by PSM-selected memory tiers.
- "Source id" is product provenance, not benchmark-only. It can be a transcript path, session id, message id, import id, or null.
- Store-time temporal resolution is preferred, but context still passes `created_at` and `source_timestamp`.
- PSM returns plain bullets because this is cheaper and more robust than JSON.
- The downstream LLM answers; PSM's job is complete, compact, grounded memory context.
