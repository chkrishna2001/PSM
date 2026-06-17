# Phase 2 — Chunking pipeline

**Status:** Complete (2026-06-17)  
**Goal:** Ingest large `llmResponse` without 32k context — segment and `remember()` per chunk.  
**Depends on:** [Phase 1](phase-1-baseline-eval.md)

---

## Problem

All presets use `context_length: 2048` ([configs.py](../../../psm-model/src/psm_model/configs.py)). Long agent handoffs overflow when passed as a single `remember()` call.

**Chunking is not a substitute for training** — it is a product requirement so each decode fits budget.

---

## Design

### Structure-aware segmenter

Split on (in priority order):

1. Markdown headers (`##`, `###`)
2. Numbered step lists (`1.`, `2.`)
3. Paragraph boundaries (double newline)
4. Hard max token fallback (~1200 tokens text)

**Target:** ~600–1200 tokens text per chunk so chunk + [`buildStoragePrompt`](../../../src/psm-core/src/prompts.ts) JSON overhead fits **2048**.

### Per-chunk remember

- Call `remember()` once per chunk.
- Shared `source_id` with `:chunk-N` suffix for provenance.
- Log chunk count + per-chunk grounding (Phase 1 metrics).

### Dedupe

At store layer: content hash or fact key collision → skip duplicate insert.

### Workflow integrity

- Do **not** split mid-step list when avoidable.
- Prefer one workflow procedure → one chunk when under budget.

---

## API sketch

```typescript
// src/psm-core/src/segment-remember.ts (or PsmService extension)
rememberChunked(input: {
  llmResponse: string;
  sourceId: string;
  metadata?: Record<string, unknown>;
}): Promise<ChunkRememberResult[]>
```

---

## Tasks

- [x] Implement `StructureAwareSegmenter` with tests on canonical plan fixture.
- [x] Implement `rememberChunked()` wrapping `remember()`.
- [x] Add dedupe in store layer for identical content from same `source_id` family.
- [x] Per-chunk grounding fields on `rememberChunked()` response.
- [x] Unit tests: header split, workflow integrity, dedupe, chunked remember.

---

## Exit criteria

- [x] Canonical plan text segments without requiring full handoff in one decode (via `maxChunkTokens`).
- [x] Chunk count + per-chunk grounding on `rememberChunked()` result.
- [x] Workflow fixture stays one chunk when under budget.

---

## Results

**Implemented in** [`src/psm-core/src/segment-remember.ts`](../../../src/psm-core/src/segment-remember.ts) + [`PsmService.rememberChunked()`](../../../src/psm-core/src/service.ts).

| Check | Result |
|-------|--------|
| `review-pr` workflow @ 1200 token budget | 1 chunk, all 5 steps preserved |
| 3-section plan @ 40 token budget | ≥3 chunks, one per header section |
| Dedupe same content across `:chunk-N` sources | `route: dedupe_skip` |
| Tests | 6 new tests in [`tests/segment-remember.test.ts`](../../../tests/segment-remember.test.ts) |

**Usage:**

```typescript
await service.rememberChunked({
  userId: "u1",
  llmResponse: longHandoff,
  source: { source_id: "plan-abc", source_kind: "agent_plan" },
  maxChunkTokens: 1200
});
```
