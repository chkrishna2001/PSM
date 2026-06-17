import test from "node:test";
import assert from "node:assert/strict";
import {
  MemoryStore,
  PsmService,
  buildIndexablesForRemember,
  extractWorkflowSteps,
  inferWorkflowKey,
  rankIndexables
} from "@psm-memory/sdk";

const REVIEW_PR_WORKFLOW = `# Review a pull request

1. Get PR info with \`gh pr view\`.
2. Check the target branch tracks the intended base.
3. List changed files with \`gh pr diff --name-only\`.
4. Review each changed file for correctness and scope.
5. Summarize findings and request changes or approve.`;

test("buildIndexablesForRemember creates review-pr workflow indexable", () => {
  const rows = buildIndexablesForRemember({
    llmResponse: REVIEW_PR_WORKFLOW,
    memoryContent: "Review pull request procedure with five ordered steps.",
    memoryTable: "episodic",
    memoryId: "mem-1"
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].kind, "workflow");
  assert.equal(rows[0].key, "review-pr");
  assert.equal(rows[0].steps?.length, 5);
});

test("extractWorkflowSteps slugifies numbered steps", () => {
  const steps = extractWorkflowSteps(REVIEW_PR_WORKFLOW);
  assert.equal(steps.length, 5);
  assert.equal(steps[0], "get_pr_info_with_gh_pr_view");
});

test("inferWorkflowKey detects review-pr from header", () => {
  assert.equal(inferWorkflowKey(REVIEW_PR_WORKFLOW), "review-pr");
});

test("remember persists workflow indexable and recall returns steps", async () => {
  const runtime = {
    async generateJson(prompt: string): Promise<string> {
      if (prompt.includes("recall_plan")) {
        return JSON.stringify({
          intent: "recall",
          target_tables: ["semantic", "episodic"],
          filters: {},
          ranking_hints: ["review-pr"],
          top_k: 3
        });
      }
      return JSON.stringify({
        action: "store_episodic",
        memory: {
          content: "Review pull request: get PR info, check target branch, list changed files, review changes, approve.",
          confidence: 0.9
        },
        facts: [],
        reasoning: "Stored workflow procedure."
      });
    }
  };

  const store = new MemoryStore(":memory:");
  store.initializeSchema();
  const service = new PsmService(store, runtime);
  await service.remember({
    userId: "u1",
    llmResponse: REVIEW_PR_WORKFLOW,
    source: { source_id: "workflow-review-pr", source_kind: "workflow_fixture" },
    extraTags: ["workflow:review-pr"]
  });

  const stored = store.getIndexable("u1", "review-pr");
  assert.ok(stored);
  assert.equal(stored?.kind, "workflow");
  assert.equal(stored?.steps.length, 5);

  const recalled = await service.recall({ userId: "u1", question: "review-pr" });
  const workflows = recalled.workflows as Array<{ key: string; steps: string[] }>;
  assert.equal(workflows.length, 1);
  assert.equal(workflows[0].key, "review-pr");
  assert.equal(workflows[0].steps.length, 5);
  store.close();
});

test("rankIndexables prefers exact key matches", () => {
  const ranked = rankIndexables("review-pr", [
    {
      id: "1",
      user_id: "u1",
      kind: "workflow",
      key: "review-pr",
      steps: ["get_pr_info"],
      salience: 0.9,
      tags: ["workflow"]
    },
    {
      id: "2",
      user_id: "u1",
      kind: "mnemonic",
      key: "sqlite-local",
      steps: [],
      salience: 0.95,
      tags: []
    }
  ]);
  assert.equal(ranked[0]?.key, "review-pr");
});

test("remember persists explicit facts from model output", async () => {
  const runtime = {
    async generateJson(): Promise<string> {
      return JSON.stringify({
        action: "store_episodic",
        memory: { content: "Caroline attended an LGBTQ support group yesterday.", confidence: 0.9 },
        facts: [{
          subject: "Caroline",
          predicate: "attended_event",
          value: "LGBTQ support group",
          value_text: "LGBTQ support group",
          confidence: 0.9,
          inference_kind: "explicit",
          evidence_text: "Caroline attended an LGBTQ support group yesterday."
        }],
        reasoning: "Stored explicit fact."
      });
    }
  };
  const store = new MemoryStore(":memory:");
  store.initializeSchema();
  const service = new PsmService(store, runtime);
  await service.remember({
    userId: "u1",
    llmResponse: "Caroline attended an LGBTQ support group yesterday and found it powerful.",
    source: { source_id: "fact-turn-1" }
  });
  const facts = store.selectMemoryFacts("u1", 10);
  assert.equal(facts.length, 1);
  assert.equal(facts[0].subject, "Caroline");
  store.close();
});
