import { appendFileSync, copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { createInterface } from "node:readline";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { defaultEmbeddingModel, defaultPsmConfig, defaultPsmConfigPath, MemoryStore, NodeLlamaRuntime, PsmService, readPsmConfig, resolvePsmDbPath, resolvePsmMemoryDir, TransformersEmbeddingRuntime, writePsmConfig, memoryTables, type EmbeddingRuntime, type MemoryTable, type ModelRuntime, type PsmConfig } from "@psm-memory/sdk";
import { boolOption, intOption, parseArgs, stringOption } from "./args.js";
import { callDaemon, startDaemon } from "./daemon.js";
import { defaultModelPath, hasDefaultModel, resolveModelPath, setupModel } from "./model.js";

const hookContextMaxItems = 3;
const hookContextMaxItemChars = 300;
const hookContextMaxTotalChars = 1200;

export async function run(argv: string[]): Promise<number> {
  const { command, options, positionals } = parseArgs(argv);
  const json = boolOption(options, "json") || boolOption(options, "pretty");
  const pretty = boolOption(options, "pretty");

  try {
    if (command === "version" || command === "--version" || command === "-v") {
      write(`${cliVersion()}\n`);
      return 0;
    }

    if (command === "help" || command === "--help" || command === "-h") {
      write(helpText());
      return 0;
    }

    if (command === "advanced" || command === "--advanced") {
      write(advancedHelpText());
      return 0;
    }

    if (command === "daemon-run") {
      await startDaemon();
      return 0;
    }

    if (command === "config") {
      if (boolOption(options, "path")) {
        write(`${defaultPsmConfigPath()}\n`);
      } else {
        output({ path: defaultPsmConfigPath(), config: readPsmConfig() }, true);
      }
      return 0;
    }

    if (command === "setup") {
      const force = boolOption(options, "force");
      const config = await resolveSetupConfig(options, { interactive: canPrompt() && !boolOption(options, "yes") });
      writePsmConfig(config);
      const modelPath = boolOption(options, "skip-model")
        ? undefined
        : await setupModel({
          force,
          log: (message) => write(`${message}\n`)
        });
      const dbPath = configuredDbPath(options);
      mkdirSync(dirname(dbPath), { recursive: true });
      const store = new MemoryStore(dbPath);
      try {
        store.initializeSchema();
      } finally {
        store.close();
      }
      const embeddingModel = config.embeddings.model;
      if (config.embeddings.enabled && !boolOption(options, "skip-embeddings")) {
        write(`Preparing embedding model: ${embeddingModel}\n`);
        await createEmbeddingRuntime({ ...options, "embedding-model": embeddingModel }).runtime.embed("PSM Memory embedding setup check.");
      }
      output({ model: modelPath, config: defaultPsmConfigPath(), memory_dir: config.memoryDir, db: dbPath, schema_migrated: true, embedding_model: config.embeddings.enabled && !boolOption(options, "skip-embeddings") ? embeddingModel : undefined, daemon: config.daemon, installed: true }, pretty || json);
      return 0;
    }

    if (command === "migrate") {
      const dbPath = adminDbPath(options);
      mkdirSync(dirname(dbPath), { recursive: true });
      const store = new MemoryStore(dbPath);
      try {
        store.initializeSchema();
      } finally {
        store.close();
      }
      output({ db: dbPath, schema_migrated: true }, pretty);
      return 0;
    }

    if (command === "backup") {
      const dbPath = adminDbPath(options);
      if (!existsSync(dbPath)) throw new Error(`Memory DB does not exist: ${dbPath}`);
      const out = stringOption(options, "out", defaultBackupPath(dbPath));
      mkdirSync(dirname(out), { recursive: true });
      copyFileSync(dbPath, out);
      output({ db: dbPath, backup: out }, pretty);
      return 0;
    }

    if (command === "export") {
      const dbPath = adminDbPath(options);
      const out = stringOption(options, "out", positionals[0] ?? "");
      const archive = exportMemories(dbPath);
      if (out) {
        mkdirSync(dirname(out), { recursive: true });
        writeFileSync(out, `${JSON.stringify(archive, null, 2)}\n`, "utf8");
        output({ db: dbPath, out, counts: archive.counts }, pretty);
      } else {
        output(archive, true);
      }
      return 0;
    }

    if (command === "import") {
      const dbPath = adminDbPath(options);
      const inputPath = stringOption(options, "in", positionals[0] ?? "");
      if (!inputPath) throw new Error("Usage: psm-memory import <path>");
      importMemories(dbPath, inputPath);
      output({ db: dbPath, imported: inputPath }, pretty);
      return 0;
    }

    if (command === "install-agent") {
      const agents = parseAgentList(stringOption(options, "agent", positionals[0] ?? ""));
      const installed = installAgents(agents);
      output({
        installed: true,
        agents: installed
      }, pretty);
      return 0;
    }

    if (command === "hook") {
      const mode = positionals[0] ?? stringOption(options, "mode", "");
      if (mode === "recall" || mode === "context") {
        await runHookRecall(options, pretty || json);
        return 0;
      }
      if (mode === "remember") {
        await runHookRemember(options);
        return 0;
      }
      if (mode === "session-start" || mode === "session-end") {
        await runHookSession(options, mode);
        return 0;
      }
      throw new Error("Usage: psm-memory hook recall|remember|session-start|session-end");
    }

    if (command === "review" || command === "review-log") {
      const review = buildReviewLog({
        dbPath: configuredDbPath(options),
        logPath: stringOption(options, "log", defaultHookLogPath()),
        date: stringOption(options, "date", new Date().toISOString().slice(0, 10)),
        limit: intOption(options, "limit", 50)
      });
      if (pretty) {
        output(review, true);
      } else {
        write(renderReviewLog(review));
      }
      return 0;
    }

    if (command === "remember") {
      const llmResponse = stringOption(options, "llm-response", positionals.join(" "));
      if (!llmResponse.trim()) throw new Error('Usage: psm-memory remember "memory text"');
      const daemonResult = await callDaemon({
        operation: "remember",
        payload: {
          llmResponse,
          userId: defaultUserId(options),
          source: sourceOptions(options, "manual", "psm-memory remember")
        }
      });
      if (daemonResult) {
        if (json) {
          output(daemonResult, pretty);
        } else {
          write(renderRememberText(daemonResult));
        }
        return 0;
      }
    }

    if (command === "recall") {
      const question = stringOption(options, "question", positionals.join(" "));
      if (!question.trim()) throw new Error('Usage: psm-memory recall "question"');
      const daemonResult = await callDaemon({
        operation: "recall",
        payload: {
          question,
          userId: defaultUserId(options),
          topK: intOption(options, "top-k", readPsmConfig().recallTopK)
        }
      });
      if (daemonResult) {
        if (json) {
          output(daemonResult, pretty);
        } else {
          write(renderRecallText(daemonResult));
        }
        return 0;
      }
    }

    const dbPath = configuredDbPath(options);
    mkdirSync(dirname(dbPath), { recursive: true });
    const store = new MemoryStore(dbPath);
    try {
      if (command === "init") {
        store.initializeSchema();
        output({
          db: dbPath,
          initialized: true,
          model_installed: hasDefaultModel(),
          next_step: hasDefaultModel() ? undefined : "Run `psm-memory setup` to download the PSM Memory model before using remember or recall."
        }, pretty);
        return 0;
      }

      store.initializeSchema();
      const runtime = createRuntime(options);
      const service = createService(store, runtime, options);

      if (command === "remember") {
        const llmResponse = stringOption(options, "llm-response", positionals.join(" "));
        if (!llmResponse.trim()) throw new Error('Usage: psm-memory remember "memory text"');
        const result = await service.remember({
          llmResponse,
          userId: defaultUserId(options),
          source: sourceOptions(options, "manual", "psm-memory remember")
        });
        if (json) {
          output(result, pretty);
        } else {
          write(renderRememberText(result));
        }
        return 0;
      }

      if (command === "recall") {
        const question = stringOption(options, "question", positionals.join(" "));
        if (!question.trim()) throw new Error('Usage: psm-memory recall "question"');
        const result = await service.recall({
          question,
          userId: defaultUserId(options),
          topK: intOption(options, "top-k", readPsmConfig().recallTopK)
        });
        if (json) {
          output(result, pretty);
        } else {
          write(renderRecallText(result));
        }
        return 0;
      }

      if (command === "show") {
        const table = stringOption(options, "table", "episodic") as MemoryTable;
        if (!memoryTables.includes(table)) throw new Error(`Unsupported table: ${table}`);
        output(store.selectTable(table, intOption(options, "limit", 20)), pretty);
        return 0;
      }

      if (command === "conflicts") {
        output(store.selectConflicts(stringOption(options, "status", "unresolved"), intOption(options, "limit", 20)), pretty);
        return 0;
      }

      throw new Error(`Unknown command: ${command}`);
    } finally {
      store.close();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    return 1;
  }
}

function createRuntime(options: Record<string, string | boolean>): ModelRuntime {
  const modelPath = resolveModelPath();
  const config = readPsmConfig();
  return new NodeLlamaRuntime({
    modelPath,
    contextSize: intOption(options, "context-size", config.runtime.contextSize),
    gpu: stringOption(options, "gpu", config.runtime.gpu) as "auto",
    gpuLayers: stringOption(options, "gpu-layers", config.runtime.gpuLayers) as "auto"
  });
}

function createService(store: MemoryStore, runtime: ModelRuntime, options: Record<string, string | boolean>): PsmService {
  return new PsmService(store, runtime, boolOption(options, "no-embeddings") || !readPsmConfig().embeddings.enabled ? undefined : createEmbeddingRuntime(options));
}

function createEmbeddingRuntime(options: Record<string, string | boolean>): { model: string; runtime: EmbeddingRuntime } {
  const model = stringOption(options, "embedding-model", process.env.PSM_MEMORY_EMBEDDING_MODEL ?? readPsmConfig().embeddings.model);
  return {
    model,
    runtime: new TransformersEmbeddingRuntime({
      model,
      cacheDir: join(modelCacheBaseDir(), "hf")
    })
  };
}

function modelCacheBaseDir(): string {
  return process.env.PSM_MEMORY_HOME ?? join(dirname(defaultModelPath()));
}

async function runHookRecall(options: Record<string, string | boolean>, pretty: boolean): Promise<void> {
  const started = Date.now();
  const timings: Record<string, number> = {};
  const dbPath = configuredDbPath(options);
  const userId = defaultUserId(options);
  const topK = intOption(options, "top-k", readPsmConfig().recallTopK);
  const hookAgent = stringOption(options, "agent", "");
  let status = "ok";
  let reason: string | undefined;
  let promptChars = 0;
  let memoryCount = 0;
  let hookOutputWritten = false;

  try {
    const readStarted = Date.now();
    const hookInput = readHookInput();
    timings.read_input = Date.now() - readStarted;

    const prompt = firstText(hookInput, ["prompt", "user_prompt", "message", "input"]);
    if (!prompt) {
      status = "skipped";
      reason = "missing_prompt";
      return;
    }
    promptChars = prompt.length;

    const daemonStarted = Date.now();
    const daemonResult = await callDaemon({
      operation: "context",
      payload: { prompt, userId, topK }
    });
    if (daemonResult) {
      timings.model_context = Date.now() - daemonStarted;
      memoryCount = renderHookRecallResult(daemonResult, pretty, hookAgent);
      hookOutputWritten = true;
      return;
    }

    const dbStarted = Date.now();
    mkdirSync(dirname(dbPath), { recursive: true });
    const store = new MemoryStore(dbPath);
    timings.open_db = Date.now() - dbStarted;
    try {
      store.initializeSchema();
      const modelStarted = Date.now();
      const service = createService(store, createRuntime(options), options);
      const result = await service.context({ prompt, userId, topK });
      timings.model_context = Date.now() - modelStarted;
      const memories = Array.isArray(result.context_items) ? result.context_items : [];
      memoryCount = memories.length;

      const renderStarted = Date.now();
      renderHookRecallResult(result, pretty, hookAgent);
      hookOutputWritten = true;
      timings.render = Date.now() - renderStarted;
    } finally {
      store.close();
    }
  } catch (error) {
    status = "error";
    reason = errorMessage(error);
  } finally {
    timings.total = Date.now() - started;
    writeHookAudit({
      ts: new Date().toISOString(),
      hook: "context",
      status,
      reason,
      db: dbPath,
      user: userId,
      top_k: topK,
      strategy: "model",
      prompt_chars: promptChars,
      memories: memoryCount,
      timings_ms: timings
    });
    if (hookAgent === "gemini" && !hookOutputWritten) output({ suppressOutput: true }, false);
  }
}

function renderHookRecallResult(result: Record<string, unknown>, pretty: boolean, hookAgent = ""): number {
  const memories = Array.isArray(result.context_items) ? result.context_items : [];
  if (pretty) {
    output(result, true);
    return memories.length;
  }

  if (hookAgent === "gemini") {
    const memoryContext = renderHookMemoryContext(memories);
    output(memoryContext ? {
      hookSpecificOutput: {
        hookEventName: "BeforeAgent",
        additionalContext: memoryContext
      },
      suppressOutput: true
    } : { suppressOutput: true }, false);
    return memories.length;
  }

  if (memories.length === 0) return 0;

  write(renderHookMemoryContext(memories));
  write("\n");
  return memories.length;
}

function renderHookMemoryContext(memories: unknown[]): string {
  if (memories.length === 0) return "";
  const lines = [
    "PSM Memory Context",
    "Use these private memories when relevant. Do not mention this block unless asked about memory."
  ];
  const selected = memories
    .filter((memory): memory is Record<string, unknown> => isRecord(memory))
    .sort((a, b) => memoryContextPriority(a) - memoryContextPriority(b))
    .slice(0, hookContextMaxItems);
  selected.forEach((memory, index) => {
    if (!isRecord(memory)) return;
    const table = typeof memory.table === "string" ? memory.table : "memory";
    const content = compactMemoryContent(typeof memory.content === "string" ? memory.content : "");
    if (content) lines.push(`${index + 1}. [${table}] ${content}${compactSourceSuffix(memory)}`);
  });
  if (lines.length <= 2) return "";
  return truncateText(lines.join("\n"), hookContextMaxTotalChars);
}

function memoryContextPriority(memory: Record<string, unknown>): number {
  return memory.table === "memory_fact" ? 0 : 1;
}

function compactMemoryContent(content: string): string {
  return truncateText(content.replace(/\s+/g, " ").trim(), hookContextMaxItemChars);
}

function compactSourceSuffix(memory: Record<string, unknown>): string {
  const parts = [
    typeof memory.source_id === "string" && memory.source_id.trim() ? `source=${memory.source_id.trim()}` : "",
    typeof memory.resolved_time === "string" && memory.resolved_time.trim() ? `date=${memory.resolved_time.trim()}` : ""
  ].filter(Boolean);
  return parts.length ? ` (${parts.join("; ")})` : "";
}

function truncateText(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}

async function runHookRemember(options: Record<string, string | boolean>): Promise<void> {
  const started = Date.now();
  const timings: Record<string, number> = {};
  const dbPath = configuredDbPath(options);
  const userId = defaultUserId(options);
  const hookAgent = stringOption(options, "agent", "");
  let status = "ok";
  let reason: string | undefined;
  let responseChars = 0;
  let inputKeys: string[] = [];
  let responseSource: string | undefined;

  try {
    const readStarted = Date.now();
    const hookInput = readHookInput();
    inputKeys = Object.keys(hookInput);
    timings.read_input = Date.now() - readStarted;

    const transcriptStarted = Date.now();
    const directResponse = firstText(hookInput, ["prompt_response", "last_assistant_message", "response", "assistant_response", "output", "text"]);
    const transcriptPath = firstText(hookInput, ["transcript_path", "transcriptPath"]);
    const transcriptResponse = transcriptAssistantText(transcriptPath);
    const latestSessionResponse = transcriptResponse ? undefined : latestCodexAssistantText();
    const response = directResponse ?? transcriptResponse ?? latestSessionResponse;
    responseSource = directResponse ? "hook_input" : transcriptResponse ? "transcript" : latestSessionResponse ? "latest_codex_session" : undefined;
    timings.resolve_response = Date.now() - transcriptStarted;

    if (!response) {
      status = "skipped";
      reason = "missing_response";
      return;
    }
    responseChars = response.length;

    const daemonStarted = Date.now();
    const daemonResult = await callDaemon({
      operation: "remember",
      payload: {
        llmResponse: response,
        userId,
        source: {
          source_kind: responseSource ?? "hook",
          source_id: transcriptPath ?? latestSessionPath() ?? undefined,
          source_timestamp: new Date().toISOString(),
          source_label: responseSource ? `agent:${responseSource}` : "agent hook"
        }
      }
    });
    if (daemonResult) {
      timings.model_remember = Date.now() - daemonStarted;
      return;
    }

    const dbStarted = Date.now();
    mkdirSync(dirname(dbPath), { recursive: true });
    const store = new MemoryStore(dbPath);
    timings.open_db = Date.now() - dbStarted;
    try {
      store.initializeSchema();
      const modelStarted = Date.now();
      const service = createService(store, createRuntime(options), options);
      await service.remember({
        llmResponse: response,
        userId,
        source: {
          source_kind: responseSource ?? "hook",
          source_id: transcriptPath ?? latestSessionPath() ?? undefined,
          source_timestamp: new Date().toISOString(),
          source_label: responseSource ? `agent:${responseSource}` : "agent hook"
        }
      });
      timings.model_remember = Date.now() - modelStarted;
    } finally {
      store.close();
    }
  } catch (error) {
    status = "error";
    reason = errorMessage(error);
  } finally {
    timings.total = Date.now() - started;
    writeHookAudit({
      ts: new Date().toISOString(),
      hook: "remember",
      status,
      reason,
      db: dbPath,
      user: userId,
      input_keys: inputKeys,
      response_source: responseSource,
      response_chars: responseChars,
      timings_ms: timings
    });
    if (hookAgent === "gemini") output({ suppressOutput: true }, false);
  }
}

async function runHookSession(options: Record<string, string | boolean>, mode: "session-start" | "session-end"): Promise<void> {
  const started = Date.now();
  const timings: Record<string, number> = {};
  const dbPath = configuredDbPath(options);
  const userId = defaultUserId(options);
  const hookAgent = stringOption(options, "agent", "");
  const sourceKind = mode === "session-start" ? "session_start" : "session_end";
  let status = "ok";
  let reason: string | undefined;
  let summaryChars = 0;

  try {
    const readStarted = Date.now();
    const hookInput = readHookInput();
    timings.read_input = Date.now() - readStarted;

    const summary = buildSessionSummary(hookInput, mode, hookAgent);
    summaryChars = summary.length;
    if (!summary.trim()) {
      status = "skipped";
      reason = "empty_session_summary";
      return;
    }

    const source = {
      source_kind: sourceKind,
      source_id: firstText(hookInput, ["transcript_path", "transcriptPath", "session_id", "sessionId"]) ?? latestSessionPath() ?? undefined,
      source_timestamp: new Date().toISOString(),
      source_label: `agent ${mode}`
    };

    const daemonStarted = Date.now();
    const daemonResult = await callDaemon({
      operation: "remember",
      payload: { llmResponse: summary, userId, source }
    });
    if (daemonResult) {
      timings.model_remember = Date.now() - daemonStarted;
      return;
    }

    const dbStarted = Date.now();
    mkdirSync(dirname(dbPath), { recursive: true });
    const store = new MemoryStore(dbPath);
    timings.open_db = Date.now() - dbStarted;
    try {
      store.initializeSchema();
      const modelStarted = Date.now();
      const service = createService(store, createRuntime(options), options);
      await service.remember({ llmResponse: summary, userId, source });
      timings.model_remember = Date.now() - modelStarted;
    } finally {
      store.close();
    }
  } catch (error) {
    status = "error";
    reason = errorMessage(error);
  } finally {
    timings.total = Date.now() - started;
    writeHookAudit({
      ts: new Date().toISOString(),
      hook: mode,
      status,
      reason,
      db: dbPath,
      user: userId,
      summary_chars: summaryChars,
      timings_ms: timings
    });
    if (hookAgent === "gemini") output({ suppressOutput: true }, false);
  }
}

function buildSessionSummary(hookInput: Record<string, unknown>, mode: "session-start" | "session-end", hookAgent: string): string {
  const cwd = firstText(hookInput, ["cwd", "workspace", "workspacePath", "workspace_path"]) ?? process.cwd();
  const repo = gitOutput(cwd, ["rev-parse", "--show-toplevel"]);
  const gitCwd = repo || cwd;
  const branch = gitOutput(gitCwd, ["branch", "--show-current"]);
  const dirty = gitOutput(gitCwd, ["status", "--short"])?.split(/\r?\n/).filter(Boolean).slice(0, 12) ?? [];
  const packageName = packageNameFor(gitCwd);
  const transcript = firstText(hookInput, ["transcript_path", "transcriptPath"]);
  const header = mode === "session-start" ? "Developer session started." : "Developer session ended.";
  const lines = [
    header,
    hookAgent ? `Agent: ${hookAgent}.` : "",
    packageName ? `Project: ${packageName}.` : "",
    repo ? `Repo: ${repo}.` : `CWD: ${cwd}.`,
    branch ? `Branch: ${branch}.` : "",
    dirty.length ? `Changed files: ${dirty.join("; ")}.` : "Changed files: none detected.",
    transcript ? `Transcript: ${transcript}.` : ""
  ].filter(Boolean);
  return lines.join("\n");
}

function gitOutput(cwd: string, args: string[]): string | undefined {
  try {
    const result = spawnSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    return result.status === 0 ? result.stdout.trim() || undefined : undefined;
  } catch {
    return undefined;
  }
}

function packageNameFor(cwd: string): string | undefined {
  const path = join(cwd, "package.json");
  if (!existsSync(path)) return undefined;
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
    return isRecord(parsed) && typeof parsed.name === "string" ? parsed.name : undefined;
  } catch {
    return undefined;
  }
}

function readHookInput(): Record<string, unknown> {
  const raw = readFileSync(0, "utf8");
  if (!raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return { raw };
  }
}

function firstText(value: unknown, names: string[]): string | undefined {
  if (!isRecord(value)) return undefined;
  for (const name of names) {
    const candidate = value[name];
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }
  return undefined;
}

function transcriptAssistantText(path: string | undefined): string | undefined {
  if (!path || !existsSync(path)) return undefined;
  const lines = readFileSync(path, "utf8").trimEnd().split(/\r?\n/).slice(-200).reverse();
  for (const line of lines) {
    try {
      const event = JSON.parse(line);
      if (isRecord(event) && isRecord(event.payload) && typeof event.payload.last_agent_message === "string" && event.payload.last_agent_message.trim()) {
        return event.payload.last_agent_message;
      }
      if (!isAssistantEvent(event)) continue;
      const text = firstText(event, ["text", "content", "message"]);
      if (text) return text;
      if (isRecord(event) && isRecord(event.payload)) {
        const payloadText = firstText(event.payload, ["text", "content", "message"]);
        if (payloadText) return payloadText;
      }
    } catch {
      continue;
    }
  }
  return undefined;
}

function latestCodexAssistantText(): string | undefined {
  const latestPath = latestSessionPath();
  return transcriptAssistantText(latestPath);
}

function latestSessionPath(): string | undefined {
  const sessionsRoot = join(homedir(), ".codex", "sessions");
  if (!existsSync(sessionsRoot)) return undefined;

  let latestPath: string | undefined;
  let latestMtime = -1;
  for (const path of walkJsonlFiles(sessionsRoot, 4)) {
    try {
      const stat = statSync(path);
      if (stat.mtimeMs > latestMtime) {
        latestMtime = stat.mtimeMs;
        latestPath = path;
      }
    } catch {
      continue;
    }
  }

  return latestPath;
}

function walkJsonlFiles(root: string, maxDepth: number): string[] {
  if (maxDepth < 0 || !existsSync(root)) return [];
  const files: string[] = [];
  for (const entry of readdirSync(root)) {
    const path = join(root, entry);
    try {
      const stat = statSync(path);
      if (stat.isDirectory()) {
        files.push(...walkJsonlFiles(path, maxDepth - 1));
      } else if (entry.endsWith(".jsonl")) {
        files.push(path);
      }
    } catch {
      continue;
    }
  }
  return files;
}

function isAssistantEvent(value: unknown): boolean {
  if (!isRecord(value)) return false;
  if (value.role === "assistant") return true;
  return isRecord(value.payload) && value.payload.role === "assistant";
}

type AgentName = "codex" | "claude" | "gemini";

const allAgents: AgentName[] = ["codex", "claude", "gemini"];

function parseAgentList(value: string): AgentName[] {
  if (!value.trim()) throw new Error("Usage: psm-memory install-agent codex|claude|gemini|all");
  const requested = value
    .split(",")
    .map((agent) => agent.trim().toLowerCase())
    .filter(Boolean);
  const expanded = requested.includes("all") ? allAgents : requested;
  const unknown = expanded.filter((agent) => !allAgents.includes(agent as AgentName));
  if (unknown.length > 0) {
    throw new Error(`Unsupported agent: ${unknown.join(", ")}`);
  }
  return [...new Set(expanded as AgentName[])];
}

function installAgents(agents: AgentName[]): Record<string, unknown>[] {
  return agents.map((agent) => {
    if (agent === "codex") return installCodexHooks();
    if (agent === "claude") return installClaudeHooks();
    return installGeminiHooks();
  });
}

function installCodexHooks(): Record<string, unknown> {
  mkdirSync(dirname(codexConfigPath()), { recursive: true });
  ensureCodexHooksFeature(codexConfigPath());

  const hooksPath = codexHooksPath();
  const current = readHooksJson(hooksPath);
  const hooks = isRecord(current.hooks) ? current.hooks : {};
  removeOldPsmHooks(hooks);
  addHook(hooks, "SessionStart", "psm-memory hook session-start");
  addHook(hooks, "UserPromptSubmit", "psm-memory hook recall");
  addHook(hooks, "Stop", "psm-memory hook remember");
  addHook(hooks, "SessionEnd", "psm-memory hook session-end");
  mkdirSync(dirname(configuredDbPath({})), { recursive: true });
  writeFileSync(hooksPath, `${JSON.stringify({ ...current, hooks }, null, 2)}\n`, "utf8");
  return {
    agent: "codex",
    config: codexConfigPath(),
    hooks: hooksPath,
    commands: ["psm-memory hook session-start", "psm-memory hook recall", "psm-memory hook remember", "psm-memory hook session-end"]
  };
}

function installClaudeHooks(): Record<string, unknown> {
  const settingsPath = claudeSettingsPath();
  mkdirSync(dirname(settingsPath), { recursive: true });

  const current = readJsonObject(settingsPath);
  const hooks = isRecord(current.hooks) ? current.hooks : {};
  removeOldPsmHooks(hooks);
  addHook(hooks, "SessionStart", "psm-memory hook session-start", { async: true });
  addHook(hooks, "UserPromptSubmit", "psm-memory hook recall");
  addHook(hooks, "Stop", "psm-memory hook remember", { async: true });
  addHook(hooks, "SessionEnd", "psm-memory hook session-end", { async: true });
  mkdirSync(dirname(configuredDbPath({})), { recursive: true });
  writeFileSync(settingsPath, `${JSON.stringify({ ...current, hooks }, null, 2)}\n`, "utf8");

  return {
    agent: "claude",
    settings: settingsPath,
    commands: ["psm-memory hook session-start", "psm-memory hook recall", "psm-memory hook remember", "psm-memory hook session-end"]
  };
}

function installGeminiHooks(): Record<string, unknown> {
  const settingsPath = geminiSettingsPath();
  mkdirSync(dirname(settingsPath), { recursive: true });

  const current = readJsonObject(settingsPath);
  const hooks = isRecord(current.hooks) ? current.hooks : {};
  removeOldPsmHooks(hooks);
  addHook(hooks, "BeforeAgent", "psm-memory hook recall --agent gemini");
  addHook(hooks, "AfterAgent", "psm-memory hook remember --agent gemini");
  const hooksConfig = isRecord(current.hooksConfig) ? { ...current.hooksConfig, enabled: true } : { enabled: true };
  mkdirSync(dirname(configuredDbPath({})), { recursive: true });
  writeFileSync(settingsPath, `${JSON.stringify({ ...current, hooksConfig, hooks }, null, 2)}\n`, "utf8");

  return {
    agent: "gemini",
    settings: settingsPath,
    commands: ["psm-memory hook recall --agent gemini", "psm-memory hook remember --agent gemini"]
  };
}

function ensureCodexHooksFeature(path: string): void {
  if (!existsSync(path)) {
    writeFileSync(path, "[features]\ncodex_hooks = true\n", "utf8");
    return;
  }

  let content = readFileSync(path, "utf8");
  if (/^\s*codex_hooks\s*=\s*true\s*$/m.test(content)) return;
  if (/^\[features\]\s*$/m.test(content)) {
    content = content.replace(/^(\[features\]\s*)$/m, "$1\ncodex_hooks = true");
  } else {
    content = `${content.trimEnd()}\n\n[features]\ncodex_hooks = true\n`;
  }
  writeFileSync(path, content, "utf8");
}

function readHooksJson(path: string): Record<string, unknown> {
  return readJsonObject(path);
}

function readJsonObject(path: string): Record<string, unknown> {
  if (!existsSync(path)) return {};
  const raw = readFileSync(path, "utf8").replace(/^\uFEFF/, "");
  if (!raw.trim()) return {};
  const parsed = JSON.parse(raw);
  return isRecord(parsed) ? parsed : {};
}

function removeOldPsmHooks(hooks: Record<string, unknown>): void {
  for (const [event, entries] of Object.entries(hooks)) {
    if (!Array.isArray(entries)) continue;
    hooks[event] = entries
      .map((entry) => {
        if (!isRecord(entry)) return entry;
        const childHooks = Array.isArray(entry.hooks) ? entry.hooks : [];
        const filtered = childHooks.filter((hook) => {
          if (!isRecord(hook) || typeof hook.command !== "string") return true;
          return !/psm-codex-hook\.ps1|psm-memory hook (context|recall|remember|session-start|session-end)/i.test(hook.command);
        });
        return { ...entry, hooks: filtered };
      })
      .filter((entry) => !isRecord(entry) || !Array.isArray(entry.hooks) || entry.hooks.length > 0);
  }
}

function addHook(hooks: Record<string, unknown>, event: string, command: string, fields: Record<string, unknown> = {}): void {
  const entries = Array.isArray(hooks[event]) ? hooks[event] : [];
  hooks[event] = [
    ...entries,
    {
      matcher: "*",
      hooks: [{ type: "command", command, ...fields }]
    }
  ];
}

function codexConfigPath(): string {
  return join(homedir(), ".codex", "config.toml");
}

function codexHooksPath(): string {
  return join(homedir(), ".codex", "hooks.json");
}

function claudeSettingsPath(): string {
  return join(homedir(), ".claude", "settings.json");
}

function geminiSettingsPath(): string {
  return join(homedir(), ".gemini", "settings.json");
}

function defaultHookLogPath(): string {
  return process.env.PSM_MEMORY_HOOK_LOG ?? join(dirname(configuredDbPath({})), "psm-memory-hooks.jsonl");
}

function defaultBackupPath(dbPath: string): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return join(dirname(dbPath), `psm-memory-${stamp}.db`);
}

interface MemoryArchive {
  format: "psm-memory-export";
  version: 1;
  exported_at: string;
  counts: Record<string, number>;
  tables: Record<string, Record<string, unknown>[]>;
}

function exportMemories(dbPath: string): MemoryArchive {
  if (!existsSync(dbPath)) throw new Error(`Memory DB does not exist: ${dbPath}`);
  const store = new MemoryStore(dbPath);
  try {
    store.initializeSchema();
    const tables: Record<string, Record<string, unknown>[]> = {
      episodic: store.selectTable("episodic", 1_000_000),
      semantic: store.selectTable("semantic", 1_000_000),
      archival: store.selectTable("archival", 1_000_000),
      conflicts: store.selectTable("conflicts", 1_000_000),
      decisions: store.selectTable("decisions", 1_000_000),
      decay_schedule: store.selectTable("decay_schedule", 1_000_000)
    };
    return {
      format: "psm-memory-export",
      version: 1,
      exported_at: new Date().toISOString(),
      counts: Object.fromEntries(Object.entries(tables).map(([table, rows]) => [table, rows.length])),
      tables
    };
  } finally {
    store.close();
  }
}

function importMemories(dbPath: string, inputPath: string): void {
  const archive = JSON.parse(readFileSync(inputPath, "utf8")) as Partial<MemoryArchive>;
  if (archive.format !== "psm-memory-export" || archive.version !== 1 || !isRecord(archive.tables)) {
    throw new Error("Unsupported PSM memory export format.");
  }
  mkdirSync(dirname(dbPath), { recursive: true });
  const store = new MemoryStore(dbPath);
  try {
    store.initializeSchema();
    insertRows(store, "episodic", Array.isArray(archive.tables.episodic) ? archive.tables.episodic : []);
    insertRows(store, "semantic", Array.isArray(archive.tables.semantic) ? archive.tables.semantic : []);
    insertRows(store, "archival", Array.isArray(archive.tables.archival) ? archive.tables.archival : []);
    insertRows(store, "conflicts", Array.isArray(archive.tables.conflicts) ? archive.tables.conflicts : []);
    insertRows(store, "decisions", Array.isArray(archive.tables.decisions) ? archive.tables.decisions : []);
    insertRows(store, "decay_schedule", Array.isArray(archive.tables.decay_schedule) ? archive.tables.decay_schedule : []);
  } finally {
    store.close();
  }
}

function insertRows(store: MemoryStore, table: string, rows: Record<string, unknown>[]): void {
  for (const row of rows) {
    store.insertRawRow(table, row);
  }
}

function sourceOptions(options: Record<string, string | boolean>, fallbackKind = "", fallbackLabel = ""): {
  source_kind?: string;
  source_id?: string;
  source_timestamp?: string;
  source_label?: string;
} {
  return {
    source_kind: stringOption(options, "source-kind", fallbackKind),
    source_id: stringOption(options, "source-id", ""),
    source_timestamp: stringOption(options, "source-timestamp", new Date().toISOString()),
    source_label: stringOption(options, "source-label", fallbackLabel)
  };
}

interface ReviewLogOptions {
  dbPath: string;
  logPath: string;
  date: string;
  limit: number;
}

interface ReviewLog {
  date: string;
  log_path: string;
  db: string;
  hook_events: Record<string, unknown>[];
  decisions: Record<string, unknown>[];
  instructions: string[];
}

function buildReviewLog(options: ReviewLogOptions): ReviewLog {
  return {
    date: options.date,
    log_path: options.logPath,
    db: options.dbPath,
    hook_events: readHookEvents(options.logPath, options.date, options.limit),
    decisions: readDecisionRows(options.dbPath, options.date, options.limit),
    instructions: [
      "Review each PSM hook event and memory decision.",
      "Mark decisions as good or bad outside this CLI for now, then share only what you explicitly choose.",
      "Useful labels: good_memory, too_verbose, too_private, wrong_context, missed_memory, should_ignore."
    ]
  };
}

function readHookEvents(path: string, date: string, limit: number): Record<string, unknown>[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      try {
        const parsed = JSON.parse(line) as unknown;
        return isRecord(parsed) ? parsed : {};
      } catch {
        return {};
      }
    })
    .filter((event) => typeof event.ts === "string" && event.ts.startsWith(date))
    .slice(-limit);
}

function readDecisionRows(dbPath: string, date: string, limit: number): Record<string, unknown>[] {
  if (!existsSync(dbPath)) return [];
  const store = new MemoryStore(dbPath);
  try {
    store.initializeSchema();
    return store.selectTable("decisions", Math.max(limit * 4, 100))
      .filter((row) => typeof row.created_at === "string" && row.created_at.startsWith(date))
      .slice(0, limit)
      .map(summarizeDecisionRow);
  } finally {
    store.close();
  }
}

function summarizeDecisionRow(row: Record<string, unknown>): Record<string, unknown> {
  const raw = typeof row.raw_json === "string" ? parseJsonObject(row.raw_json) : {};
  const memory = isRecord(raw.memory) ? raw.memory : typeof raw.memory === "string" ? { content: raw.memory } : {};
  return {
    id: row.id,
    created_at: row.created_at,
    user_id: row.user_id,
    action: row.action,
    route: row.route,
    memory: {
      content: typeof memory.content === "string" ? memory.content : undefined,
      confidence: memory.confidence,
      tags: memory.tags
    },
    reasoning: row.reasoning,
    review_labels: ["good_memory", "too_verbose", "too_private", "wrong_context", "missed_memory", "should_ignore"]
  };
}

function parseJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown;
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function renderReviewLog(review: ReviewLog): string {
  const lines = [
    `PSM Review Log ${review.date}`,
    `DB: ${review.db}`,
    `Hook log: ${review.log_path}`,
    "",
    "Hook events:"
  ];
  if (review.hook_events.length === 0) {
    lines.push("- none");
  } else {
    for (const event of review.hook_events) {
      lines.push(`- ${event.ts ?? ""} ${event.hook ?? "hook"} ${event.status ?? ""} ${event.reason ? `(${event.reason})` : ""} total=${formatTiming(event, "total")} model=${formatTiming(event, "model_context") || formatTiming(event, "model_remember")}`);
    }
  }
  lines.push("", "Memory decisions:");
  if (review.decisions.length === 0) {
    lines.push("- none");
  } else {
    for (const decision of review.decisions) {
      const memory = isRecord(decision.memory) ? decision.memory : {};
      lines.push(`- ${decision.created_at ?? ""} ${decision.action ?? ""}/${decision.route ?? ""}`);
      lines.push(`  memory: ${memory.content ?? ""}`);
      lines.push(`  reasoning: ${decision.reasoning ?? ""}`);
      lines.push("  feedback: [good_memory | too_verbose | too_private | wrong_context | missed_memory | should_ignore]");
    }
  }
  lines.push("", "No data is uploaded. Share only selected review notes if you choose.");
  return `${lines.join("\n")}\n`;
}

function formatTiming(event: Record<string, unknown>, key: string): string {
  const timings = isRecord(event.timings_ms) ? event.timings_ms : {};
  const value = timings[key];
  return typeof value === "number" ? `${value}ms` : "";
}

function writeHookAudit(entry: Record<string, unknown>): void {
  try {
    const path = defaultHookLogPath();
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf8");
  } catch {
    // Hook logging must never break the agent workflow.
  }
}

function configuredDbPath(options: Record<string, string | boolean>): string {
  const explicit = stringOption(options, "db", "");
  return resolvePsmDbPath({
    dbPath: explicit,
    memoryDir: stringOption(options, "memory-dir", "")
  });
}

function adminDbPath(options: Record<string, string | boolean>): string {
  return configuredDbPath(options);
}

async function resolveSetupConfig(options: Record<string, string | boolean>, setup: { interactive: boolean }): Promise<PsmConfig> {
  const current = readPsmConfig();
  const defaults = defaultPsmConfig();
  const base: PsmConfig = {
    ...current,
    memoryDir: resolvePsmMemoryDir(stringOption(options, "memory-dir", "")),
    userId: stringOption(options, "user", current.userId),
    recallTopK: intOption(options, "recall-top-k", current.recallTopK),
    embeddings: {
      ...current.embeddings,
      model: stringOption(options, "embedding-model", current.embeddings.model),
      enabled: boolOption(options, "skip-embeddings") ? false : current.embeddings.enabled
    },
    runtime: {
      ...current.runtime,
      contextSize: intOption(options, "context-size", current.runtime.contextSize),
      gpu: stringOption(options, "gpu", current.runtime.gpu),
      gpuLayers: stringOption(options, "gpu-layers", current.runtime.gpuLayers)
    },
    daemon: {
      ...current.daemon,
      enabled: boolOption(options, "daemon") || current.daemon.enabled,
      autostart: boolOption(options, "daemon") || current.daemon.autostart,
      idleTimeoutMs: intOption(options, "daemon-idle-ms", current.daemon.idleTimeoutMs),
      startupTimeoutMs: intOption(options, "daemon-startup-ms", current.daemon.startupTimeoutMs)
    }
  };

  if (!setup.interactive) return base;

  const answers = await promptForConfig(base, defaults);
  return {
    ...base,
    ...answers,
    embeddings: {
      ...base.embeddings,
      ...(answers.embeddings ?? {})
    },
    daemon: {
      ...base.daemon,
      ...(answers.daemon ?? {})
    }
  };
}

async function promptForConfig(current: PsmConfig, defaults: PsmConfig): Promise<Partial<PsmConfig>> {
  write("PSM Memory setup\n");
  write(`Config file: ${defaultPsmConfigPath()}\n`);
  write("Press Enter to accept a default.\n\n");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  try {
    const memoryDir = await ask(rl, "Memory directory", current.memoryDir || defaults.memoryDir);
    const userId = await ask(rl, "Local user id", current.userId || defaults.userId);
    const recallTopK = Number(await ask(rl, "Recall candidate count", String(current.recallTopK || defaults.recallTopK)));
    const embeddingsEnabled = yesNo(await ask(rl, "Enable embeddings", current.embeddings.enabled ? "yes" : "no"), current.embeddings.enabled);
    const embeddingModel = embeddingsEnabled
      ? await ask(rl, "Embedding model", current.embeddings.model || defaults.embeddings.model)
      : current.embeddings.model;
    const daemonEnabled = yesNo(await ask(rl, "Enable PSM daemon config", current.daemon.enabled ? "yes" : "no"), current.daemon.enabled);
    const daemonAutostart = daemonEnabled
      ? yesNo(await ask(rl, "Daemon autostart on first memory call", current.daemon.autostart ? "yes" : "no"), current.daemon.autostart)
      : current.daemon.autostart;
    const idleTimeoutMs = daemonEnabled
      ? Number(await ask(rl, "Daemon idle timeout ms", String(current.daemon.idleTimeoutMs || defaults.daemon.idleTimeoutMs)))
      : current.daemon.idleTimeoutMs;
    const startupTimeoutMs = daemonEnabled
      ? Number(await ask(rl, "Daemon startup timeout ms", String(current.daemon.startupTimeoutMs || defaults.daemon.startupTimeoutMs)))
      : current.daemon.startupTimeoutMs;

    return {
      memoryDir,
      userId,
      recallTopK: Number.isInteger(recallTopK) && recallTopK > 0 ? recallTopK : current.recallTopK,
      embeddings: {
        enabled: embeddingsEnabled,
        model: embeddingModel
      },
      daemon: {
        ...current.daemon,
        enabled: daemonEnabled,
        autostart: daemonAutostart,
        idleTimeoutMs: Number.isInteger(idleTimeoutMs) && idleTimeoutMs > 0 ? idleTimeoutMs : current.daemon.idleTimeoutMs,
        startupTimeoutMs: Number.isInteger(startupTimeoutMs) && startupTimeoutMs > 0 ? startupTimeoutMs : current.daemon.startupTimeoutMs
      }
    };
  } finally {
    rl.close();
  }
}

function ask(rl: { question(query: string, callback: (answer: string) => void): void }, label: string, fallback: string): Promise<string> {
  return new Promise((resolveAnswer) => {
    rl.question(`${label} [${fallback}]: `, (answer) => resolveAnswer(answer.trim() || fallback));
  });
}

function yesNo(value: string, fallback: boolean): boolean {
  const normalized = value.trim().toLowerCase();
  if (["y", "yes", "true", "1", "on"].includes(normalized)) return true;
  if (["n", "no", "false", "0", "off"].includes(normalized)) return false;
  return fallback;
}

function canPrompt(): boolean {
  return process.env.PSM_MEMORY_NONINTERACTIVE !== "1" && process.env.CI !== "true" && process.stdin.isTTY === true && process.stdout.isTTY === true;
}

function defaultUserId(options: Record<string, string | boolean>): string {
  const explicit = stringOption(options, "user", "");
  if (explicit) return explicit;
  return readPsmConfig().userId;
}

function renderRecallText(result: Record<string, unknown>): string {
  const memories = Array.isArray(result.memories) ? result.memories : [];
  if (memories.length === 0) return "No relevant PSM memories.\n";
  const lines = ["PSM Memory", ""];
  for (const memory of memories) {
    if (!isRecord(memory) || typeof memory.content !== "string" || !memory.content.trim()) continue;
    lines.push(`- ${memory.content.replace(/^-\s*/, "")}`);
  }
  return `${lines.join("\n")}\n`;
}

function renderRememberText(result: Record<string, unknown>): string {
  const route = typeof result.route === "string" ? result.route : "";
  if (route === "ignore" || route === "recall_only" || route === "parse_error_noop") return "No memory stored.\n";
  const written = Array.isArray(result.written) ? result.written.filter((item) => typeof item === "string") : [];
  return written.length > 0 ? `Remembered: ${written.join(", ")}.\n` : "Remembered.\n";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function output(value: unknown, pretty: boolean): void {
  process.stdout.write(`${JSON.stringify(value, null, pretty ? 2 : 0)}\n`);
}

function write(value: string): void {
  process.stdout.write(value);
}

function cliVersion(): string {
  const packageJsonPath = fileURLToPath(new URL("../package.json", import.meta.url));
  const raw = readFileSync(packageJsonPath, "utf8");
  const parsed = JSON.parse(raw) as { version?: string };
  return typeof parsed.version === "string" && parsed.version.trim() ? parsed.version : "unknown";
}

function helpText(): string {
  return `PSM Memory commands:
  version [--version | -v]
  setup [--memory-dir <path>] [--force] [--pretty]
  remember "<text>" [--json]
  recall "<question>" [--json]
  install-agent codex|claude|gemini|all[,agent...] [--pretty]
  hook recall|remember|session-start|session-end
  review [--date YYYY-MM-DD] [--pretty]
  config [--path]
  export [out.json] [--pretty]
  import <export.json> [--pretty]
  backup [--out <path>] [--pretty]
  migrate [--pretty]

The model downloads automatically during npm install. If that was skipped or failed, run "psm-memory setup".
Interactive setup writes an editable config file. Run "psm-memory config --path" to locate it.
`;
}

function advancedHelpText(): string {
  return `PSM Memory developer/debug options:
  --db <path>             Override the configured DB for admin, tests, or benchmarks.
  --top-k <n>             Override recall candidate count. Default: 5.
  --embedding-model <id>  Override embedding model. Default: ${defaultEmbeddingModel}.
  --no-embeddings         Disable vector search for debugging.
  --recall-top-k <n>      Persist the default recall candidate count during setup.
  --daemon                Enable daemon autostart config during setup.
  --daemon-idle-ms <n>    Persist daemon idle shutdown timeout during setup.
  --daemon-startup-ms <n> Persist daemon startup wait timeout during setup.
  setup --yes             Accept defaults without prompting.
  setup --skip-model      Skip PSM model download during setup.
  setup --skip-embeddings Skip embedding model preparation during setup.
  --context-size <n>      Override local GGUF runtime context size.
  --gpu <mode>            Override node-llama-cpp backend selection.
  --gpu-layers <mode>     Override GPU layer offload.

Default model path:
  ${defaultModelPath()}

The npm package downloads the default Q4_K_M GGUF model during install. If that download was skipped or failed, run "psm-memory setup".
`;
}
