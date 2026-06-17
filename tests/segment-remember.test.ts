import test from "node:test";
import assert from "node:assert/strict";
import { MemoryStore, PsmService, estimateTextTokens, segmentLlmResponse } from "@psm-memory/sdk";

const REVIEW_PR_WORKFLOW = `# Review a pull request

1. Get PR info with \`gh pr view\`.
2. Check the target branch tracks the intended base.
3. List changed files with \`gh pr diff --name-only\`.
4. Review each changed file for correctness and scope.
5. Summarize findings and request changes or approve.`;

const MULTI_SECTION_PLAN = `## Phase 1 — Baseline eval

Goal: measure HF checkpoints on prod-shaped remember_target before training.

Metrics: content_grounding_rate, curriculum_bleed_rate, fail_safe_ignore_rate.

## Phase 2 — Chunking pipeline

Split long assistant handoffs on markdown headers and numbered steps.
Target 600–1200 tokens per chunk so prompt overhead fits 2048 context.
Call remember() per chunk with shared source_id and :chunk-N suffix.

## Phase 3 — Indexables

Add workflow keys like review-pr and recall by compact indexable keys.`;

test("segmentLlmResponse keeps short text as a single chunk", () => {
  const text = "Short assistant summary about SQLite preferences.";
  const segments = segmentLlmResponse(text);
  assert.equal(segments.length, 1);
  assert.equal(segments[0].splitReason, "single");
});

test("segmentLlmResponse splits markdown handoffs by headers", () => {
  const segments = segmentLlmResponse(MULTI_SECTION_PLAN, { maxChunkTokens: 40, minChunkTokens: 10 });
  assert.ok(segments.length >= 3);
  assert.ok(segments.every((segment) => segment.estimatedTokens <= 40));
  assert.ok(segments.some((segment) => segment.text.includes("Phase 1")));
  assert.ok(segments.some((segment) => segment.text.includes("Phase 3")));
});

test("segmentLlmResponse keeps numbered workflow in one chunk", () => {
  const segments = segmentLlmResponse(REVIEW_PR_WORKFLOW, { maxChunkTokens: 1200 });
  assert.equal(segments.length, 1);
  assert.equal(segments[0].splitReason, "single");
  assert.ok(/1\.\s+Get PR info/.test(segments[0].text));
  assert.ok(/5\.\s+Summarize findings/.test(segments[0].text));
});

test("estimateTextTokens uses a conservative chars/4 heuristic", () => {
  assert.equal(estimateTextTokens("abcd"), 1);
  assert.ok(estimateTextTokens("x".repeat(400)) >= 100);
});

test("rememberChunked calls remember per chunk with chunk source ids", async () => {
  let callCount = 0;
  const runtime = {
    async generateJson(prompt: string): Promise<string> {
      if (prompt.includes("repair_remember_json")) {
        return JSON.stringify({ action: "ignore", memory: null, facts: [], reasoning: "repair" });
      }
      callCount++;
      return JSON.stringify({
        action: "store_episodic",
        memory: { content: `Stored chunk ${callCount}`, confidence: 0.9 },
        facts: [],
        reasoning: "chunk store"
      });
    }
  };

  const store = new MemoryStore(":memory:");
  store.initializeSchema();
  const service = new PsmService(store, runtime);
  const result = await service.rememberChunked({
    userId: "u1",
    llmResponse: MULTI_SECTION_PLAN,
    source: { source_id: "handoff-abc", source_kind: "agent_plan" },
    maxChunkTokens: 40,
    minChunkTokens: 10
  });

  assert.equal(result.chunked, true);
  assert.ok(Number(result.chunk_count) >= 3);
  assert.equal(callCount, Number(result.chunk_count));
  const chunks = result.chunks as Array<{ source_id: string; chunk_index: number }>;
  assert.ok(chunks.every((chunk) => chunk.source_id.startsWith("handoff-abc:chunk-")));
  assert.equal(chunks[0].source_id, "handoff-abc:chunk-0");
  store.close();
});

test("store dedupes identical content within the same chunked source family", () => {
  const store = new MemoryStore(":memory:");
  store.initializeSchema();
  const first = store.applyDecision(
    "u1",
    "handoff:chunk-0",
    {
      action: "store_episodic",
      memory: { content: "Review pull request changes before approve." },
      reasoning: "first",
      raw_json: "{}"
    }
  );
  assert.deepEqual(first.written, ["episodic"]);

  const second = store.applyDecision(
    "u1",
    "handoff:chunk-1",
    {
      action: "store_episodic",
      memory: { content: "Review pull request changes before approve." },
      reasoning: "duplicate",
      raw_json: "{}"
    }
  );
  assert.equal(second.route, "dedupe_skip");
  assert.deepEqual(second.written, []);
  store.close();
});
