import test from "node:test";
import assert from "node:assert/strict";
import { MemoryStore, parseRecallPlan, parseStorageDecision, rankMemories, routeForAction, type ModelRuntime } from "@psm-memory/sdk";
import { createPsmHooks } from "@psm-memory/pi-plugin";

const testRuntime: ModelRuntime = {
  async generateJson(prompt: string): Promise<string> {
    if (prompt.includes("context_plan") || prompt.includes("recall_plan")) {
      return JSON.stringify({
        intent: "recall",
        target_tables: ["semantic", "episodic"],
        filters: {},
        ranking_hints: ["sqlite", "database"],
        top_k: 3
      });
    }
    if (prompt.includes("render_context")) {
      return JSON.stringify({
        context_items: [
          {
            id: "memory-1",
            table: "semantic",
            content: "User prefers SQLite for local memory tools.",
            reason: "Relevant database preference."
          }
        ],
        reasoning: "Selected relevant memory context."
      });
    }
    return JSON.stringify({
      action: "store_episodic",
      memory: { content: "Use SQLite for this local memory workflow.", confidence: 0.9 },
      reasoning: "Relevant durable preference."
    });
  }
};

test("storage decision parser falls back on invalid JSON", () => {
  const decision = parseStorageDecision("not json", "User likes SQLite.");
  assert.equal(decision.action, "store_episodic");
  assert.equal(decision.memory?.content, "User likes SQLite.");
  assert.ok(decision.parse_error);
});

test("storage decision parser accepts string memory payloads", () => {
  const decision = parseStorageDecision(JSON.stringify({
    action: "store_episodic",
    memory: "User prefers hook integrations to stay model-backed.",
    reasoning: "Durable project preference."
  }), "fallback response");
  assert.equal(decision.memory?.content, "User prefers hook integrations to stay model-backed.");
});

test("recall plan parser normalizes bad model output", () => {
  const plan = parseRecallPlan("{bad", "database preference", 3);
  assert.deepEqual(plan.target_tables, ["semantic", "episodic"]);
  assert.equal(plan.top_k, 3);
  assert.ok(plan.parse_error);
});

test("action routing maps conflicts and semantic promotion", () => {
  assert.equal(routeForAction("flag_conflict"), "conflict_log_and_hold");
  assert.equal(routeForAction("promote_semantic"), "semantic_upsert");
  assert.equal(routeForAction("store_episodic"), "episodic_insert");
});

test("ranking returns relevant memories first", () => {
  const ranked = rankMemories("What database does the user prefer?", [
    {
      id: "1",
      user_id: "u",
      table: "episodic",
      content: "User prefers SQLite for local apps.",
      confidence: 0.8,
      strength: 0.8
    },
    {
      id: "2",
      user_id: "u",
      table: "episodic",
      content: "User went hiking yesterday.",
      confidence: 0.8,
      strength: 0.8
    }
  ], 1);
  assert.equal(ranked[0].id, "1");
});

test("SQLite store applies episodic decisions", () => {
  const store = new MemoryStore(`dist/test-core-${Date.now()}.db`);
  store.initializeSchema();
  const result = store.applyDecision("u1", "test", parseStorageDecision(JSON.stringify({
    action: "store_episodic",
    memory: { content: "User likes TypeScript.", confidence: 0.9 },
    reasoning: "Useful preference."
  }), "fallback"));
  const rows = store.selectMemories("u1", ["episodic"], 10);
  store.close();
  assert.equal(result.route, "episodic_insert");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].content, "User likes TypeScript.");
});

test("PI hooks inject memory before prompt and store response asynchronously", async () => {
  const dbPath = `dist/test-pi-hooks-${Date.now()}.db`;
  const seedStore = new MemoryStore(dbPath);
  seedStore.initializeSchema();
  seedStore.insertSemantic("demo", "User prefers SQLite for local memory tools.");
  seedStore.close();

  const hooks = createPsmHooks({ dbPath, userId: "demo", runtime: testRuntime });
  const prepared = await hooks.enrichPrompt({ prompt: "Which database should I use?", topK: 3 });

  assert.equal(prepared.messages[0]?.role, "system");
  assert.ok(prepared.memoryContext.includes("SQLite"));
  assert.equal(prepared.messages.at(-1)?.content, "Which database should I use?");

  hooks.rememberResponse({ response: "Use SQLite for this local memory workflow." });
  await hooks.flush();
  await hooks.close();

  const checkStore = new MemoryStore(dbPath);
  const rows = checkStore.selectTable("decisions", 10);
  checkStore.close();
  assert.equal(rows.length, 1);
});

test("service remember repairs invalid model JSON before writing", async () => {
  const dbPath = `dist/test-repair-remember-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  let calls = 0;
  const runtime: ModelRuntime = {
    async generateJson(): Promise<string> {
      calls++;
      if (calls === 1) return "not valid json";
      return JSON.stringify({
        action: "store_episodic",
        memory: { content: "User prefers repair retry before dropping invalid PSM JSON.", confidence: 0.9 },
        reasoning: "Durable implementation preference."
      });
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  const result = await service.remember({ userId: "u1", llmResponse: "User prefers retry logic." });
  assert.equal(result.route, "episodic_insert");
  assert.equal(result.repair_attempted, true);
  assert.equal(store.selectTable("decisions", 10).length, 1);
  assert.equal(store.selectMemories("u1", ["episodic"], 10).length, 1);
  store.close();
});

test("service remember does not write if repair also returns invalid JSON", async () => {
  const dbPath = `dist/test-invalid-remember-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const runtime: ModelRuntime = {
    async generateJson(): Promise<string> {
      return "not valid json";
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  const result = await service.remember({ userId: "u1", llmResponse: "User prefers short logs." });
  assert.equal(result.route, "parse_error_noop");
  assert.equal(result.repair_attempted, true);
  assert.equal(store.selectTable("decisions", 10).length, 0);
  assert.equal(store.selectMemories("u1", ["episodic"], 10).length, 0);
  store.close();
});
