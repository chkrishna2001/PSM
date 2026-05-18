# PSM Memory Product Plan

## Purpose

Build PSM as a trustworthy local-first memory layer, not only as a LOCOMO benchmark runner.

LOCOMO remains useful because it exposes real product failures:

- exact factual recall
- temporal grounding
- relationship and preference recall
- context noise
- provenance
- answerability from stored memories

But the product should not overfit to LOCOMO. PSM should support both human-like contextual memory and reliable structured facts.

## Current Diagnosis

The core PSM idea is still valid:

```text
local model decides memory behavior
+ deterministic storage/retrieval/context infrastructure
+ downstream answer model uses exact retrieved context
```

The current implementation is incomplete in three areas.

### 1. Memory Shape Is Too Summary-Heavy

PSM currently stores many useful memories, but the stored content is often prose:

```text
Melanie mentioned being a great counselor with empathy and understanding,
and shared a painting of a sunrise from 2022...
```

That is useful, but product recall also needs explicit facts:

```json
{
  "kind": "fact",
  "subject": "Melanie",
  "relation": "shared_painting",
  "object": "sunrise painting",
  "value": "2022",
  "source_id": "D1:12",
  "confidence": 0.95
}
```

Summary memory should remain, but it cannot be the only representation.

### 2. Retrieval Must Be Hybrid

Vector search is not enough for personal memory. It is weak for:

- names
- dates and numbers
- rare exact terms
- relationship status
- source ids
- short factual questions

Retrieval should use a union candidate set:

```text
vector top N
+ lexical/BM25 top N
+ entity/name matches
+ date/number matches
+ tag/source matches
+ recent/high-strength memories when relevant
```

Then rerank the union.

Vector similarity should be a signal, not the gatekeeper.

### 3. Context Rendering Must Be Exact First

For benchmark and product reliability, PSM should not free-form rewrite memory context until the model is trained and validated for that task.

Preferred v1 path:

```text
question
-> recall plan / ranking hints
-> hybrid retrieval
-> deterministic rerank
-> exact DB-backed context items
-> answer model
```

Generative context compression can be added later as an optional optimization, but exact memory rows should be the fallback and benchmark default.

## Target Architecture

### Memory Types

PSM should write more than one representation when useful.

#### Episodic Memory

Specific event or turn-grounded memory.

```json
{
  "table": "episodic",
  "content": "Melanie shared a painting of a sunrise from 2022 that holds special meaning to her.",
  "source_kind": "locomo_turn",
  "source_id": "conv-26:D1:12",
  "source_label": "Melanie, session_1",
  "tags": ["painting", "sunrise", "2022", "locomo_speaker:Melanie"]
}
```

#### Structured Fact Memory

Queryable facts extracted from episodic memories.

```json
{
  "table": "facts",
  "subject": "Melanie",
  "relation": "painted_or_shared",
  "object": "sunrise painting",
  "value": "2022",
  "source_memory_id": "...",
  "source_id": "conv-26:D1:12",
  "confidence": 0.95
}
```

This can start as a JSON payload in semantic memory if adding a new table is too large for the first pass. A first-class facts table is better long term.

#### Semantic Memory

Stable consolidated profile/preference knowledge.

```json
{
  "table": "semantic",
  "content": "Melanie's recurring activities include pottery, camping, painting, and swimming.",
  "source_episodes": ["...", "..."],
  "confidence": 0.88
}
```

#### Archival Memory

Low-frequency or old memories preserved for long-tail recall.

Archival memory should not be deleted only because it is old. It should be lower priority unless directly matched by entity/date/source/keyword.

## Structured Fact Extraction

Do not fine-tune first. Implement the schema and harness first.

Phase 1 should prompt the current PSM model to emit structured facts alongside episodic memory:

```json
{
  "action": "store_episodic",
  "memory": {...},
  "facts": [
    {
      "subject": "...",
      "relation": "...",
      "object": "...",
      "value": "...",
      "time": "...",
      "confidence": 0.0,
      "source_span": "..."
    }
  ]
}
```

Fine-tune only after collecting failures from this schema.

Fine-tune targets:

- extract exact dates, people, activities, relationships
- preserve raw relative temporal phrases
- resolve relative time when source timestamp exists
- emit facts with provenance
- distinguish explicit facts from inferred facts
- promote repeated episodic facts into semantic memory

## Hybrid Retrieval Plan

Add a retriever that returns a candidate union before reranking.

Inputs:

- user id
- query
- requested tables
- optional filters from recall plan
- top K final
- candidate limits per strategy

Candidate strategies:

1. Vector search over memory embeddings.
2. Lexical token overlap over content and tags.
3. Exact entity/name match.
4. Number/date match.
5. Structured fact match.
6. Source/tag match.
7. Recency/strength/confidence fallback.

Rerank signals:

```text
exact entity match
+ exact keyword match
+ exact date/number match
+ fact subject/relation/object match
+ vector score
+ lexical score
+ confidence
+ strength
+ source/provenance match
- unrelated entity penalty
- excessive context/noise penalty
```

The final context should usually be 3-5 items for answer generation. Wider top K can remain available for diagnostics.

## Daily Decay And Dream Consolidation

The paper describes decay and dream consolidation. The implementation should add this as opportunistic background maintenance, not as a required service.

### Design Goal

Run consolidation at most once per day when PSM is already active.

No daemon or scheduler is required for v1.

### Trigger

Whenever PSM is initialized by CLI, SDK, plugin, or daemon:

```text
read last_consolidation_at
if now - last_consolidation_at > 24h:
  acquire consolidation lock
  start background consolidation task
  continue user request without blocking
```

If the runtime is short-lived, consolidation may not finish. That is acceptable. The next PSM startup can retry.

### Metadata

Add a small metadata table:

```sql
CREATE TABLE IF NOT EXISTS maintenance_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Keys:

- `last_consolidation_started_at`
- `last_consolidation_completed_at`
- `consolidation_lock_until`
- `last_decay_completed_at`

Use `consolidation_lock_until` to avoid concurrent consolidations from multiple PSM processes.

### Consolidation Steps

1. Select candidate episodic memories:
   - high access count
   - high strength/confidence
   - repeated entities/topics
   - old but still accessed
   - related facts scattered across sessions

2. Apply decay:
   - reduce strength for old, low-access memories
   - preserve high emotional weight or user-pinned memories
   - never delete in v1; only lower retrieval priority

3. Promote facts:
   - repeated episodic facts become semantic memories
   - exact facts become structured fact records
   - preserve source episode ids

4. Merge duplicates:
   - identify near-duplicate episodic memories
   - create consolidated semantic memory
   - mark duplicates as promoted or lower strength

5. Archive low-value memories:
   - move or copy stale low-strength memories to archival
   - preserve source/provenance

6. Write audit decision:
   - what changed
   - why
   - source memory ids
   - model output when model-assisted

### Background Execution

Node implementation options:

- SDK: fire and forget with `setTimeout(() => runConsolidation(), 0)` after initialization.
- CLI: start background promise but do not block command completion unless command is `psm-memory consolidate`.
- Daemon/plugin: run opportunistically after first request if due.

The task must:

- catch and log errors
- never crash the user command
- release/expire lock
- limit runtime and batch size

### Manual Command

Also add an explicit command:

```powershell
psm-memory consolidate --pretty
psm-memory consolidate --dry-run --pretty
```

Manual command is useful for tests, demos, and debugging.

## Implementation Phases

### Phase 1: Stabilize Benchmark Path Without Overfitting

- Keep 50-question smoke runs.
- Use exact DB-backed context.
- Add hybrid retrieval to benchmark and product code.
- Report answer accuracy plus retrieval diagnostics.
- Do not run full LOCOMO until smoke crosses an agreed threshold.

Success target:

```text
50-question smoke answer accuracy >= 50%
Evidence Hit@K improves over current baseline
Failure review shows no systematic context pollution
```

### Phase 2: Structured Fact Memory

- Add fact schema or semantic JSON fact records.
- Extend storage prompt/parser to accept facts.
- Store facts with source memory ids and confidence.
- Retrieve facts as first-class context candidates.

Success target:

```text
Known failures like sunrise/2022, relationship status, and activities are answerable from fact records.
```

### Phase 3: Opportunistic Consolidation

- Add `maintenance_state`.
- Add due-check and lock.
- Add manual `consolidate` command.
- Implement no-delete v1 decay and semantic/fact promotion.
- Add tests for lock behavior and idempotency.

Success target:

```text
Consolidation runs at most once per 24h, does not block normal PSM commands, and produces traceable semantic/fact promotions.
```

### Phase 4: Fine-Tune

Fine-tune after phases 1-3 produce training data.

Datasets:

- storage decision failures
- structured fact extraction failures
- temporal grounding failures
- consolidation/promotion examples

Do not fine-tune broad chat answering. PSM should remain the memory system, not the final answer model.

## Product Guardrails

- Do not delete memories automatically in v1 consolidation.
- Always preserve provenance.
- Separate explicit facts from inferred facts.
- Keep exact context fallback.
- Avoid hiding product behavior behind benchmark-only code.
- LOCOMO should be a regression harness, not the product definition.
- Treat the SQLite DB as user data. No destructive migrations, resets, or silent table rewrites.

## Migration Safety

PSM has existing users and downloaded installs, so schema changes must be conservative.

### Rules

- Prefer additive migrations.
- Never drop or rewrite existing memory tables during install/setup.
- Never require users to delete `user_memory.db`.
- Make migrations idempotent so setup can run repeatedly.
- Keep old tables readable until a later major migration with an explicit compatibility plan.
- Back up the DB before any migration that changes schema.
- If migration fails, leave the original DB usable.
- Explain schema changes in release notes.

### Safe Next Step

Do not migrate immediately to one physical `memories` table.

Instead, add a unified retrieval abstraction over existing tables:

```text
episodic
+ semantic
+ archival
+ future facts
-> memory index candidates
-> hybrid ranking
-> exact context items
```

This gives most of the retrieval benefit of a single table without breaking user data.

### Additive Schema Changes

Allowed v1 additions:

```sql
CREATE TABLE IF NOT EXISTS facts (...);
CREATE TABLE IF NOT EXISTS maintenance_state (...);
CREATE INDEX IF NOT EXISTS ...;
ALTER TABLE episodic ADD COLUMN ...;
ALTER TABLE semantic ADD COLUMN ...;
```

Only use `ALTER TABLE ... ADD COLUMN` after checking the column does not already exist.

### Backup Strategy

Before schema migration:

```text
user_memory.db
user_memory.backup-YYYYMMDD-HHMMSS.db
```

Backup behavior:

- create the backup in the same memory directory
- do not overwrite an existing backup
- skip backup only when DB does not exist yet
- record backup path in setup output when `--pretty` or JSON output is used

### Migration History

Keep `schema_version`, but use it as real migration state:

```text
1 = current core memory tables
2 = facts table added
3 = maintenance_state and consolidation metadata added
4 = hybrid retrieval indexes added
```

Each migration should be a named, testable function:

```text
migrateTo2FactsTable()
migrateTo3MaintenanceState()
migrateTo4HybridRetrievalIndexes()
```

### Compatibility Plan

Short term:

- existing `episodic`, `semantic`, `archival` tables remain canonical
- hybrid retrieval reads across them
- new fact records are additive

Long term:

- if a single physical `memories` table becomes necessary, introduce it as a shadow table first
- dual-write new memories to old and new storage for at least one minor release
- provide a migration command that copies old rows into the new table
- only remove old-table dependency in a major version

### Install/Setup Expectations

During `psm-memory setup` or first SDK initialization:

```text
open DB
read schema version
backup if migration needed
apply additive migrations
verify required tables/indexes
continue normal setup
```

Setup must not block on consolidation. Consolidation is maintenance, not schema migration.

## Immediate Next Steps

1. Commit the current Colab benchmark fixes.
2. Run `smoke-50-exact`.
3. Implement hybrid retrieval in SDK.
4. Re-run the same 50-question smoke.
5. Add structured fact extraction schema.
6. Add opportunistic consolidation metadata and manual command.
