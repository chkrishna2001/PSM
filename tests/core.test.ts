import test from "node:test";
import assert from "node:assert/strict";
import { hybridRankMemories, MemoryStore, parseRecallPlan, parseStorageDecision, rankMemories, routeForAction, type ModelRuntime } from "@psm-memory/sdk";
import { run as runCli } from "@psm-memory/cli";
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

async function captureCli(argv: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  const originalStdout = process.stdout.write;
  const originalStderr = process.stderr.write;
  let stdout = "";
  let stderr = "";
  process.stdout.write = ((data: string) => {
    stdout += data;
    return true;
  }) as typeof process.stdout.write;
  process.stderr.write = ((data: string) => {
    stderr += data;
    return true;
  }) as typeof process.stderr.write;
  try {
    const code = await runCli(argv);
    return { code, stdout, stderr };
  } finally {
    process.stdout.write = originalStdout;
    process.stderr.write = originalStderr;
  }
}

test("storage decision parser falls back on invalid JSON", () => {
  const decision = parseStorageDecision("not json", "User likes SQLite.");
  assert.equal(decision.action, "store_episodic");
  assert.equal(decision.memory?.content, "User likes SQLite.");
  assert.ok(decision.parse_error);
});

test("CLI help exposes the single memory flow", async () => {
  const result = await captureCli(["help"]);
  assert.equal(result.code, 0);
  assert.ok(result.stdout.includes('remember "<text>"'));
  assert.ok(result.stdout.includes('recall "<question>"'));
  assert.ok(result.stdout.includes("install-agent codex|claude|gemini|all"));
  assert.ok(!result.stdout.includes("context --prompt"));
  assert.ok(!result.stdout.includes("--top-k"));
  assert.ok(!result.stdout.includes("--embedding-model"));
});

test("CLI installs Gemini hooks using Gemini JSON protocol", async () => {
  const originalHome = process.env.HOME;
  const originalUserProfile = process.env.USERPROFILE;
  const originalLocalAppData = process.env.LOCALAPPDATA;
  const originalPsmDir = process.env.PSM_MEMORY_DIR;
  const root = `dist/test-gemini-hooks-${Date.now()}`;
  process.env.HOME = root;
  process.env.USERPROFILE = root;
  process.env.LOCALAPPDATA = root;
  process.env.PSM_MEMORY_DIR = `${root}/memory`;
  try {
    const result = await captureCli(["install-agent", "gemini", "--pretty"]);
    assert.equal(result.code, 0, result.stderr);
    const parsed = JSON.parse(result.stdout) as { agents: Array<{ agent: string; settings: string }> };
    assert.equal(parsed.agents[0].agent, "gemini");
    assert.ok(parsed.agents[0].settings.endsWith(".gemini\\settings.json") || parsed.agents[0].settings.endsWith(".gemini/settings.json"));
    const settings = JSON.parse((await import("node:fs")).readFileSync(parsed.agents[0].settings, "utf8")) as {
      hooksConfig?: { enabled?: boolean };
      hooks?: Record<string, Array<{ hooks: Array<{ command: string }> }>>;
    };
    assert.equal(settings.hooksConfig?.enabled, true);
    assert.equal(settings.hooks?.BeforeAgent?.[0]?.hooks?.[0]?.command, "psm-memory hook recall --agent gemini");
    assert.equal(settings.hooks?.AfterAgent?.[0]?.hooks?.[0]?.command, "psm-memory hook remember --agent gemini");
  } finally {
    if (originalHome === undefined) delete process.env.HOME; else process.env.HOME = originalHome;
    if (originalUserProfile === undefined) delete process.env.USERPROFILE; else process.env.USERPROFILE = originalUserProfile;
    if (originalLocalAppData === undefined) delete process.env.LOCALAPPDATA; else process.env.LOCALAPPDATA = originalLocalAppData;
    if (originalPsmDir === undefined) delete process.env.PSM_MEMORY_DIR; else process.env.PSM_MEMORY_DIR = originalPsmDir;
  }
});

test("CLI installs Codex session hooks with recall and remember", async () => {
  const originalHome = process.env.HOME;
  const originalUserProfile = process.env.USERPROFILE;
  const originalLocalAppData = process.env.LOCALAPPDATA;
  const originalPsmDir = process.env.PSM_MEMORY_DIR;
  const root = `dist/test-codex-session-hooks-${Date.now()}`;
  process.env.HOME = root;
  process.env.USERPROFILE = root;
  process.env.LOCALAPPDATA = root;
  process.env.PSM_MEMORY_DIR = `${root}/memory`;
  try {
    const result = await captureCli(["install-agent", "codex", "--pretty"]);
    assert.equal(result.code, 0, result.stderr);
    const parsed = JSON.parse(result.stdout) as { agents: Array<{ agent: string; hooks: string; commands: string[] }> };
    assert.equal(parsed.agents[0].agent, "codex");
    assert.deepEqual(parsed.agents[0].commands, [
      "psm-memory hook session-start",
      "psm-memory hook recall",
      "psm-memory hook remember",
      "psm-memory hook session-end"
    ]);
    const hooksJson = JSON.parse((await import("node:fs")).readFileSync(parsed.agents[0].hooks, "utf8")) as {
      hooks?: Record<string, Array<{ hooks: Array<{ command: string }> }>>;
    };
    assert.equal(hooksJson.hooks?.SessionStart?.[0]?.hooks?.[0]?.command, "psm-memory hook session-start");
    assert.equal(hooksJson.hooks?.UserPromptSubmit?.[0]?.hooks?.[0]?.command, "psm-memory hook recall");
    assert.equal(hooksJson.hooks?.Stop?.[0]?.hooks?.[0]?.command, "psm-memory hook remember");
    assert.equal(hooksJson.hooks?.SessionEnd?.[0]?.hooks?.[0]?.command, "psm-memory hook session-end");
  } finally {
    if (originalHome === undefined) delete process.env.HOME; else process.env.HOME = originalHome;
    if (originalUserProfile === undefined) delete process.env.USERPROFILE; else process.env.USERPROFILE = originalUserProfile;
    if (originalLocalAppData === undefined) delete process.env.LOCALAPPDATA; else process.env.LOCALAPPDATA = originalLocalAppData;
    if (originalPsmDir === undefined) delete process.env.PSM_MEMORY_DIR; else process.env.PSM_MEMORY_DIR = originalPsmDir;
  }
});

test("CLI exposes version output", async () => {
  const direct = await captureCli(["version"]);
  const flag = await captureCli(["--version"]);
  const shortFlag = await captureCli(["-v"]);
  assert.equal(direct.code, 0);
  assert.equal(flag.code, 0);
  assert.equal(shortFlag.code, 0);
  assert.equal(direct.stdout.trim(), "0.1.1");
  assert.equal(flag.stdout.trim(), "0.1.1");
  assert.equal(shortFlag.stdout.trim(), "0.1.1");
});

test("CLI init uses PSM-owned app data by default", async () => {
  const originalLocalAppData = process.env.LOCALAPPDATA;
  const originalPsmDb = process.env.PSM_MEMORY_DB;
  const originalPsmDir = process.env.PSM_MEMORY_DIR;
  const localAppData = `dist/test-cli-appdata-${Date.now()}`;
  delete process.env.PSM_MEMORY_DB;
  delete process.env.PSM_MEMORY_DIR;
  process.env.LOCALAPPDATA = localAppData;
  try {
    const result = await captureCli(["init", "--pretty"]);
    assert.equal(result.code, 0, result.stderr);
    const parsed = JSON.parse(result.stdout) as Record<string, unknown>;
    assert.ok(String(parsed.db).includes("psm-memory"));
    assert.ok(String(parsed.db).endsWith("psm-memory.db"));
    assert.ok(!String(parsed.db).includes(".codex"));
  } finally {
    if (originalLocalAppData === undefined) {
      delete process.env.LOCALAPPDATA;
    } else {
      process.env.LOCALAPPDATA = originalLocalAppData;
    }
    if (originalPsmDb === undefined) {
      delete process.env.PSM_MEMORY_DB;
    } else {
      process.env.PSM_MEMORY_DB = originalPsmDb;
    }
    if (originalPsmDir === undefined) {
      delete process.env.PSM_MEMORY_DIR;
    } else {
      process.env.PSM_MEMORY_DIR = originalPsmDir;
    }
  }
});

test("CLI setup writes editable config with daemon settings", async () => {
  const originalLocalAppData = process.env.LOCALAPPDATA;
  const originalPsmDb = process.env.PSM_MEMORY_DB;
  const originalPsmDir = process.env.PSM_MEMORY_DIR;
  const localAppData = `dist/test-cli-setup-${Date.now()}`;
  delete process.env.PSM_MEMORY_DB;
  delete process.env.PSM_MEMORY_DIR;
  process.env.LOCALAPPDATA = localAppData;
  try {
    const memoryDir = `${localAppData}/custom-memory`;
    const result = await captureCli(["setup", "--memory-dir", memoryDir, "--user", "test-user", "--daemon", "--daemon-idle-ms", "900000", "--daemon-startup-ms", "60000", "--skip-model", "--skip-embeddings", "--yes", "--pretty"]);
    assert.equal(result.code, 0, result.stderr);
    const parsed = JSON.parse(result.stdout) as Record<string, unknown>;
    assert.ok(String(parsed.memory_dir).endsWith("custom-memory"));
    assert.ok(String(parsed.config).endsWith("config.json"));
    const configResult = await captureCli(["config"]);
    assert.equal(configResult.code, 0, configResult.stderr);
    const configOutput = JSON.parse(configResult.stdout) as { config: { userId: string; daemon: { enabled: boolean; autostart: boolean; idleTimeoutMs: number; startupTimeoutMs: number } } };
    assert.equal(configOutput.config.userId, "test-user");
    assert.equal(configOutput.config.daemon.enabled, true);
    assert.equal(configOutput.config.daemon.autostart, true);
    assert.equal(configOutput.config.daemon.idleTimeoutMs, 900000);
    assert.equal(configOutput.config.daemon.startupTimeoutMs, 60000);
  } finally {
    if (originalLocalAppData === undefined) {
      delete process.env.LOCALAPPDATA;
    } else {
      process.env.LOCALAPPDATA = originalLocalAppData;
    }
    if (originalPsmDb === undefined) {
      delete process.env.PSM_MEMORY_DB;
    } else {
      process.env.PSM_MEMORY_DB = originalPsmDb;
    }
    if (originalPsmDir === undefined) {
      delete process.env.PSM_MEMORY_DIR;
    } else {
      process.env.PSM_MEMORY_DIR = originalPsmDir;
    }
  }
});

test("storage decision parser accepts string memory payloads", () => {
  const decision = parseStorageDecision(JSON.stringify({
    action: "store_episodic",
    memory: "User prefers hook integrations to stay model-backed.",
    reasoning: "Durable project preference."
  }), "fallback response");
  assert.equal(decision.memory?.content, "User prefers hook integrations to stay model-backed.");
});

test("storage decision parser accepts extracted facts", () => {
  const decision = parseStorageDecision(JSON.stringify({
    action: "store_episodic",
    memory: {
      content: "Caroline is a single parent creating a family.",
      confidence: 0.86,
      tags: ["family", "relationship_status"]
    },
    facts: [
      {
        subject: "Caroline",
        predicate: "Relationship Status",
        value: "single",
        confidence: 0.75,
        inference_kind: "inferred",
        evidence_text: "single parent"
      }
    ],
    reasoning: "Durable personal fact."
  }), "fallback response");
  assert.equal(decision.facts?.length, 1);
  assert.equal(decision.facts?.[0]?.predicate, "relationship_status");
  assert.equal(decision.facts?.[0]?.value_text, "single");
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

test("hybrid ranking boosts exact factual terms and numbers", () => {
  const ranked = hybridRankMemories("When did Melanie paint a sunrise?", [
    {
      id: "wrong",
      user_id: "u",
      table: "episodic",
      content: "Melanie experienced a meteor shower during a camping trip last year.",
      tags: JSON.stringify(["camping", "memory"]),
      confidence: 0.95,
      strength: 0.9
    },
    {
      id: "right",
      user_id: "u",
      table: "episodic",
      content: "Melanie shared a painting of a sunrise from 2022 that holds special meaning to her.",
      tags: JSON.stringify(["painting", "locomo_speaker:Melanie", "locomo_dia_id:D1:12"]),
      confidence: 0.9,
      strength: 0.85
    }
  ], { topK: 1 });
  assert.equal(ranked[0].id, "right");
});

test("hybrid ranking suppresses duplicate memory content", () => {
  const ranked = hybridRankMemories("What relationship status does Caroline have?", [
    {
      id: "one",
      user_id: "u",
      table: "episodic",
      content: "Caroline is a single parent creating a family.",
      confidence: 0.9,
      strength: 0.9
    },
    {
      id: "two",
      user_id: "u",
      table: "episodic",
      content: "Caroline is a single parent creating a family.",
      confidence: 0.8,
      strength: 0.8
    }
  ], { topK: 5 });
  assert.equal(ranked.length, 1);
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

test("SQLite store persists source and temporal metadata", () => {
  const store = new MemoryStore(`dist/test-metadata-${Date.now()}.db`);
  store.initializeSchema();
  store.applyDecision("u1", "session-1", parseStorageDecision(JSON.stringify({
    action: "store_episodic",
    memory: {
      content: "User shipped the CLI yesterday.",
      confidence: 0.9,
      source_kind: "transcript",
      source_id: "session-1",
      source_timestamp: "2026-05-16T10:00:00.000Z",
      source_label: "Codex session",
      temporal_expression: "yesterday",
      resolved_time: "2026-05-15",
      resolved_time_confidence: 0.85
    },
    reasoning: "Durable shipped-work memory."
  }), "fallback"));
  const rows = store.selectMemories("u1", ["episodic"], 10);
  store.close();
  assert.equal(rows[0].source_kind, "transcript");
  assert.equal(rows[0].source_id, "session-1");
  assert.equal(rows[0].source_timestamp, "2026-05-16T10:00:00.000Z");
  assert.equal(rows[0].temporal_expression, "yesterday");
  assert.equal(rows[0].resolved_time, "2026-05-15");
  assert.equal(rows[0].resolved_time_confidence, 0.85);
  assert.ok(rows[0].created_at);
});

test("SQLite store persists extracted memory facts linked to source memory", () => {
  const store = new MemoryStore(`dist/test-memory-facts-${Date.now()}.db`);
  store.initializeSchema();
  const result = store.applyDecision("u1", "session-relationship", parseStorageDecision(JSON.stringify({
    action: "store_episodic",
    memory: {
      content: "Caroline is a single parent creating a family.",
      confidence: 0.86,
      source_id: "session-relationship",
      source_timestamp: "2026-05-16T12:00:00.000Z"
    },
    facts: [
      {
        subject: "Caroline",
        predicate: "relationship_status",
        value: "single",
        fact_type: "profile_fact",
        confidence: 0.75,
        inference_kind: "inferred",
        evidence_text: "single parent"
      }
    ],
    reasoning: "Durable relationship context."
  }), "fallback"));
  const facts = store.selectMemoryFacts("u1", 10);
  store.close();
  assert.equal(result.route, "episodic_insert");
  assert.equal(facts.length, 1);
  assert.equal(facts[0].subject, "Caroline");
  assert.equal(facts[0].predicate, "relationship_status");
  assert.equal(facts[0].value_text, "single");
  assert.equal(facts[0].source_memory_table, "episodic");
  assert.equal(facts[0].source_memory_id, result.memory_refs[0].id);
  assert.equal(facts[0].source_id, "session-relationship");
  assert.equal(facts[0].source_timestamp, "2026-05-16T12:00:00.000Z");
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

test("PI hooks render lean memory context", async () => {
  const dbPath = `dist/test-pi-lean-context-${Date.now()}.db`;
  const seedStore = new MemoryStore(dbPath);
  seedStore.initializeSchema();
  const longText = `User prefers SQLite for local memory tools. ${"Detailed implementation note. ".repeat(40)}`;
  const memoryId = seedStore.insertSemantic("demo", longText);
  seedStore.insertMemoryFact("demo", {
    subject: "User",
    predicate: "database_preference",
    value: "SQLite",
    confidence: 0.9,
    inference_kind: "explicit",
    evidence_text: "prefers SQLite"
  }, { table: "semantic", id: memoryId, content: longText });
  seedStore.insertSemantic("demo", "User also discussed a less relevant logging detail.");
  seedStore.insertSemantic("demo", "User also discussed a less relevant deployment detail.");
  seedStore.insertSemantic("demo", "User also discussed a less relevant notebook detail.");
  seedStore.close();

  const runtime: ModelRuntime = {
    async generateJson(): Promise<string> {
      return JSON.stringify({
        intent: "recall",
        target_tables: ["semantic", "episodic"],
        filters: {},
        ranking_hints: ["SQLite database preference"],
        top_k: 5
      });
    }
  };
  const hooks = createPsmHooks({ dbPath, userId: "demo", runtime, topK: 5 });
  const prepared = await hooks.enrichPrompt({ prompt: "Which database should I use?", topK: 5 });
  await hooks.close();

  const memoryLines = prepared.memoryContext.split(/\r?\n/).filter((line) => /^\d+\./.test(line));
  assert.ok(memoryLines.length <= 3);
  assert.ok(prepared.memoryContext.length <= 1200);
  assert.ok(prepared.memoryContext.includes("[memory_fact]"));
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

test("service remember applies source metadata when model omits it", async () => {
  const dbPath = `dist/test-remember-source-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const runtime: ModelRuntime = {
    async generateJson(): Promise<string> {
      return JSON.stringify({
        action: "store_episodic",
        memory: { content: "User prefers source provenance in memories.", confidence: 0.9 },
        reasoning: "Durable product preference."
      });
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  await service.remember({
    userId: "u1",
    llmResponse: "User prefers provenance.",
    source: {
      source_kind: "transcript",
      source_id: "session-2",
      source_timestamp: "2026-05-16T12:00:00.000Z",
      source_label: "Test session"
    }
  });
  const rows = store.selectMemories("u1", ["episodic"], 10);
  store.close();
  assert.equal(rows[0].source_kind, "transcript");
  assert.equal(rows[0].source_id, "session-2");
  assert.equal(rows[0].source_timestamp, "2026-05-16T12:00:00.000Z");
  assert.equal(rows[0].source_label, "Test session");
});

test("service remember stores model-extracted facts with product source metadata", async () => {
  const dbPath = `dist/test-remember-facts-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const runtime: ModelRuntime = {
    async generateJson(): Promise<string> {
      return JSON.stringify({
        action: "store_episodic",
        memory: { content: "Caroline is a single parent creating a family.", confidence: 0.86 },
        facts: [
          {
            subject: "Caroline",
            predicate: "relationship_status",
            value: "single",
            confidence: 0.75,
            inference_kind: "inferred",
            evidence_text: "single parent"
          }
        ],
        reasoning: "Durable relationship status fact."
      });
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  const result = await service.remember({
    userId: "u1",
    llmResponse: "Caroline is a single parent creating a family.",
    source: {
      source_kind: "transcript",
      source_id: "session-facts",
      source_timestamp: "2026-05-16T12:00:00.000Z",
      source_label: "Test session"
    }
  });
  const facts = store.selectMemoryFacts("u1", 10);
  store.close();
  assert.equal(result.route, "episodic_insert");
  assert.equal(facts.length, 1);
  assert.equal(facts[0].predicate, "relationship_status");
  assert.equal(facts[0].source_memory_table, "episodic");
  assert.equal(facts[0].source_id, "session-facts");
  assert.equal(facts[0].source_timestamp, "2026-05-16T12:00:00.000Z");
});

test("service remember resolves relative temporal memories and facts from source timestamp", async () => {
  const dbPath = `dist/test-remember-temporal-facts-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const runtime: ModelRuntime = {
    async generateJson(): Promise<string> {
      return JSON.stringify({
        action: "store_episodic",
        memory: { content: "Caroline attended a LGBTQ support group yesterday.", confidence: 0.9 },
        facts: [
          {
            subject: "Caroline LGBTQ support group attendance",
            predicate: "event_date",
            value: "yesterday",
            fact_type: "temporal_fact",
            confidence: 0.9,
            inference_kind: "explicit",
            evidence_text: "yesterday"
          }
        ],
        reasoning: "Durable dated event."
      });
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  await service.remember({
    userId: "u1",
    llmResponse: "Caroline attended a LGBTQ support group yesterday.",
    source: {
      source_kind: "transcript",
      source_id: "conv-26:D1:3",
      source_timestamp: "1:56 pm on 8 May, 2023"
    }
  });
  const rows = store.selectMemories("u1", ["episodic"], 10);
  const facts = store.selectMemoryFacts("u1", 10);
  store.close();
  assert.equal(rows[0].temporal_expression, "yesterday");
  assert.equal(rows[0].resolved_time, "7 May 2023");
  assert.equal(facts[0].temporal_expression, "yesterday");
  assert.equal(facts[0].resolved_time, "7 May 2023");
});

test("service context uses exact DB rows instead of generated memory text", async () => {
  const dbPath = `dist/test-context-plan-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  store.insertSemantic("u1", "User prefers SQLite for local databases.");
  store.insertEpisodic("u1", "User fixed a TypeScript bug yesterday.", {
    source_kind: "transcript",
    source_id: "session-3",
    source_timestamp: "2026-05-16T12:00:00.000Z",
    temporal_expression: "yesterday",
    resolved_time: "2026-05-15"
  });
  const runtime: ModelRuntime = {
    async generateJson(prompt: string): Promise<string> {
      if (prompt.includes("context_plan")) {
        return JSON.stringify({
          intent: "specific_event_recall",
          target_tables: ["episodic"],
          filters: {},
          ranking_hints: ["TypeScript bug yesterday"],
          temporal_intent: "recent relative date",
          top_k: 3
        });
      }
      return "- [episodic | source_time=2024-01-15T14:30:00.000Z] User struggled with Terraform state management in production.";
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  const result = await service.context({ userId: "u1", prompt: "What did I fix yesterday?", topK: 3 });
  store.close();
  const rows = result.memory_context as Array<Record<string, unknown>>;
  const items = result.context_items as Array<Record<string, unknown>>;
  assert.ok(rows.length >= 1);
  assert.equal(rows[0].table, "episodic");
  assert.equal(rows[0].source_id, "session-3");
  assert.equal(items[0].memory_id, rows[0].id);
  assert.equal(items[0].resolved_time, "2026-05-15");
  assert.ok(/resolved_time=2026-05-15/.test(String(items[0].content)));
  assert.ok(!/Terraform/.test(String(items[0].content)));
  assert.equal((result.grounding as Record<string, unknown>).generated_text_allowed, false);
});

test("service context treats recall plan tables as boosts instead of hard filters", async () => {
  const dbPath = `dist/test-context-hybrid-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  store.insertEpisodic("u1", "Melanie shared a painting of a sunrise from 2022 that holds special meaning to her.", {
    tags: ["painting", "locomo_speaker:Melanie", "locomo_dia_id:D1:12"]
  });
  store.insertSemantic("u1", "Melanie enjoys community events and meaningful outdoor memories.");
  const runtime: ModelRuntime = {
    async generateJson(prompt: string): Promise<string> {
      if (prompt.includes("context_plan")) {
        return JSON.stringify({
          intent: "semantic_profile_recall",
          target_tables: ["semantic"],
          filters: {},
          ranking_hints: ["Melanie sunrise painting"],
          top_k: 3
        });
      }
      return "{}";
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  const result = await service.context({ userId: "u1", prompt: "When did Melanie paint a sunrise?", topK: 3 });
  store.close();
  const rows = result.memory_context as Array<Record<string, unknown>>;
  assert.equal(rows[0].table, "episodic");
  assert.ok(String(rows[0].content).includes("2022"));
});

test("service context renders extracted facts before source memory prose", async () => {
  const dbPath = `dist/test-context-facts-${Date.now()}.db`;
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const memoryId = store.insertEpisodic("u1", "Caroline is a single parent creating a family.", {
    source_id: "session-facts",
    source_timestamp: "2026-05-16T12:00:00.000Z"
  });
  store.insertMemoryFact("u1", {
    subject: "Caroline",
    predicate: "relationship_status",
    value: "single",
    confidence: 0.8,
    inference_kind: "inferred",
    evidence_text: "single parent"
  }, { table: "episodic", id: memoryId, content: "Caroline is a single parent creating a family." }, {
    source_id: "session-facts",
    source_timestamp: "2026-05-16T12:00:00.000Z"
  });
  const runtime: ModelRuntime = {
    async generateJson(prompt: string): Promise<string> {
      if (prompt.includes("context_plan")) {
        return JSON.stringify({
          intent: "profile_fact_recall",
          target_tables: ["semantic", "episodic"],
          filters: {},
          ranking_hints: ["Caroline relationship status"],
          top_k: 3
        });
      }
      return "{}";
    }
  };
  const { PsmService } = await import("@psm-memory/sdk");
  const service = new PsmService(store, runtime);
  const result = await service.context({ userId: "u1", prompt: "What is Caroline's relationship status?", topK: 3 });
  store.close();
  const items = result.context_items as Array<Record<string, unknown>>;
  const facts = result.fact_context as Array<Record<string, unknown>>;
  assert.equal(items[0].table, "memory_fact");
  assert.ok(String(items[0].content).includes("Caroline relationship_status single"));
  assert.ok(String(items[0].content).includes("Evidence: single parent"));
  assert.equal(facts[0].predicate, "relationship_status");
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

test("raw row import preserves memories across databases", () => {
  const sourceDb = `dist/test-export-source-${Date.now()}.db`;
  const targetDb = `dist/test-export-target-${Date.now()}.db`;
  const source = new MemoryStore(sourceDb);
  source.initializeSchema();
  source.insertEpisodic("u1", "User migrated portable memories yesterday.", {
    source_kind: "transcript",
    source_id: "session-export",
    source_timestamp: "2026-05-16T15:00:00.000Z",
    temporal_expression: "yesterday",
    resolved_time: "2026-05-15"
  });
  const exportedRows = source.selectTable("episodic", 10);
  source.close();

  const target = new MemoryStore(targetDb);
  target.initializeSchema();
  target.insertRawRow("episodic", exportedRows[0]);
  const rows = target.selectMemories("u1", ["episodic"], 10);
  target.close();
  assert.equal(rows.length, 1);
  assert.equal(rows[0].source_id, "session-export");
  assert.equal(rows[0].resolved_time, "2026-05-15");
});
