import test from "node:test";
import assert from "node:assert/strict";
import { MemoryStore, parseRecallPlan, parseStorageDecision, rankMemories, routeForAction } from "psm-sdk";

test("storage decision parser falls back on invalid JSON", () => {
  const decision = parseStorageDecision("not json", "User likes SQLite.");
  assert.equal(decision.action, "store_episodic");
  assert.equal(decision.memory?.content, "User likes SQLite.");
  assert.ok(decision.parse_error);
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
