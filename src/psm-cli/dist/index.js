import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { defaultEmbeddingModel, MemoryStore, NodeLlamaRuntime, PsmService, TransformersEmbeddingRuntime, memoryTables } from "@psm-memory/sdk";
import { boolOption, intOption, parseArgs, required, stringOption } from "./args.js";
import { defaultModelPath, hasDefaultModel, resolveModelPath, setupModel } from "./model.js";
export async function run(argv) {
    const { command, options } = parseArgs(argv);
    const dbPath = stringOption(options, "db", "user_memory.db");
    const pretty = boolOption(options, "pretty");
    try {
        if (command === "help" || command === "--help" || command === "-h") {
            write(helpText());
            return 0;
        }
        if (command === "advanced" || command === "--advanced") {
            write(advancedHelpText());
            return 0;
        }
        if (command === "setup") {
            const force = boolOption(options, "force");
            const modelPath = await setupModel({
                force,
                log: (message) => write(`${message}\n`)
            });
            const embeddingModel = stringOption(options, "embedding-model", defaultEmbeddingModel);
            if (!boolOption(options, "skip-embeddings")) {
                write(`Preparing embedding model: ${embeddingModel}\n`);
                await createEmbeddingRuntime({ ...options, "embedding-model": embeddingModel }).runtime.embed("PSM Memory embedding setup check.");
            }
            output({ model: modelPath, embedding_model: boolOption(options, "skip-embeddings") ? undefined : embeddingModel, installed: true }, pretty);
            return 0;
        }
        if (command === "install-agent") {
            const agents = parseAgentList(required(options, "agent"));
            const installed = installAgents(agents);
            output({
                installed: true,
                agents: installed
            }, pretty);
            return 0;
        }
        if (command === "hook") {
            const mode = argv[1] ?? stringOption(options, "mode", "");
            if (mode === "context") {
                await runHookContext(options, pretty);
                return 0;
            }
            if (mode === "remember") {
                await runHookRemember(options);
                return 0;
            }
            throw new Error("Usage: psm-memory hook context|remember");
        }
        if (command === "review-log") {
            const review = buildReviewLog({
                dbPath: stringOption(options, "db", defaultCodexDbPath()),
                logPath: stringOption(options, "log", defaultHookLogPath()),
                date: stringOption(options, "date", new Date().toISOString().slice(0, 10)),
                limit: intOption(options, "limit", 50)
            });
            if (pretty) {
                output(review, true);
            }
            else {
                write(renderReviewLog(review));
            }
            return 0;
        }
        const store = new MemoryStore(dbPath);
        try {
            if (command === "init") {
                store.initializeSchema();
                output({
                    db: dbPath,
                    initialized: true,
                    model_installed: hasDefaultModel(),
                    next_step: hasDefaultModel() ? undefined : "Run `psm-memory setup` to download the PSM Memory model before using context, remember, or recall."
                }, pretty);
                return 0;
            }
            store.initializeSchema();
            const runtime = createRuntime(options);
            const service = createService(store, runtime, options);
            if (command === "context") {
                output(await service.context({
                    prompt: required(options, "prompt"),
                    userId: stringOption(options, "user", "default-user"),
                    topK: intOption(options, "top-k", 5)
                }), pretty);
                return 0;
            }
            if (command === "remember") {
                output(await service.remember({
                    llmResponse: required(options, "llm-response"),
                    userId: stringOption(options, "user", "default-user")
                }), pretty);
                return 0;
            }
            if (command === "recall") {
                output(await service.recall({
                    question: required(options, "question"),
                    userId: stringOption(options, "user", "default-user"),
                    topK: intOption(options, "top-k", 5)
                }), pretty);
                return 0;
            }
            if (command === "show") {
                const table = stringOption(options, "table", "episodic");
                if (!memoryTables.includes(table))
                    throw new Error(`Unsupported table: ${table}`);
                output(store.selectTable(table, intOption(options, "limit", 20)), pretty);
                return 0;
            }
            if (command === "conflicts") {
                output(store.selectConflicts(stringOption(options, "status", "unresolved"), intOption(options, "limit", 20)), pretty);
                return 0;
            }
            throw new Error(`Unknown command: ${command}`);
        }
        finally {
            store.close();
        }
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        process.stderr.write(`${message}\n`);
        return 1;
    }
}
function createRuntime(options) {
    const modelPath = resolveModelPath();
    return new NodeLlamaRuntime({
        modelPath,
        contextSize: intOption(options, "context-size", 4096),
        gpu: stringOption(options, "gpu", "auto"),
        gpuLayers: stringOption(options, "gpu-layers", "auto")
    });
}
function createService(store, runtime, options) {
    return new PsmService(store, runtime, boolOption(options, "no-embeddings") ? undefined : createEmbeddingRuntime(options));
}
function createEmbeddingRuntime(options) {
    const model = stringOption(options, "embedding-model", process.env.PSM_MEMORY_EMBEDDING_MODEL ?? defaultEmbeddingModel);
    return {
        model,
        runtime: new TransformersEmbeddingRuntime({
            model,
            cacheDir: join(modelCacheBaseDir(), "hf")
        })
    };
}
function modelCacheBaseDir() {
    return process.env.PSM_MEMORY_HOME ?? join(dirname(defaultModelPath()));
}
async function runHookContext(options, pretty) {
    const started = Date.now();
    const timings = {};
    const dbPath = stringOption(options, "db", defaultCodexDbPath());
    const userId = stringOption(options, "user", "codex");
    const topK = intOption(options, "top-k", 5);
    let status = "ok";
    let reason;
    let promptChars = 0;
    let memoryCount = 0;
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
            if (pretty) {
                output(result, true);
                return;
            }
            if (memories.length === 0)
                return;
            const renderStarted = Date.now();
            write("PSM Memory Context\n");
            write("Use these private memories when relevant. Do not mention this block unless asked about memory.\n");
            memories.forEach((memory, index) => {
                if (!isRecord(memory))
                    return;
                const table = typeof memory.table === "string" ? memory.table : "memory";
                const content = typeof memory.content === "string" ? memory.content : "";
                if (content)
                    write(`${index + 1}. [${table}] ${content}\n`);
            });
            timings.render = Date.now() - renderStarted;
        }
        finally {
            store.close();
        }
    }
    catch (error) {
        status = "error";
        reason = errorMessage(error);
    }
    finally {
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
    }
}
async function runHookRemember(options) {
    const started = Date.now();
    const timings = {};
    const dbPath = stringOption(options, "db", defaultCodexDbPath());
    const userId = stringOption(options, "user", "codex");
    let status = "ok";
    let reason;
    let responseChars = 0;
    let inputKeys = [];
    let responseSource;
    try {
        const readStarted = Date.now();
        const hookInput = readHookInput();
        inputKeys = Object.keys(hookInput);
        timings.read_input = Date.now() - readStarted;
        const transcriptStarted = Date.now();
        const directResponse = firstText(hookInput, ["last_assistant_message", "response", "assistant_response", "output", "text"]);
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
                userId
            });
            timings.model_remember = Date.now() - modelStarted;
        }
        finally {
            store.close();
        }
    }
    catch (error) {
        status = "error";
        reason = errorMessage(error);
    }
    finally {
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
    }
}
function readHookInput() {
    const raw = readFileSync(0, "utf8");
    if (!raw.trim())
        return {};
    try {
        const parsed = JSON.parse(raw);
        return isRecord(parsed) ? parsed : {};
    }
    catch {
        return { raw };
    }
}
function firstText(value, names) {
    if (!isRecord(value))
        return undefined;
    for (const name of names) {
        const candidate = value[name];
        if (typeof candidate === "string" && candidate.trim())
            return candidate;
    }
    return undefined;
}
function transcriptAssistantText(path) {
    if (!path || !existsSync(path))
        return undefined;
    const lines = readFileSync(path, "utf8").trimEnd().split(/\r?\n/).slice(-200).reverse();
    for (const line of lines) {
        try {
            const event = JSON.parse(line);
            if (isRecord(event) && isRecord(event.payload) && typeof event.payload.last_agent_message === "string" && event.payload.last_agent_message.trim()) {
                return event.payload.last_agent_message;
            }
            if (!isAssistantEvent(event))
                continue;
            const text = firstText(event, ["text", "content", "message"]);
            if (text)
                return text;
            if (isRecord(event) && isRecord(event.payload)) {
                const payloadText = firstText(event.payload, ["text", "content", "message"]);
                if (payloadText)
                    return payloadText;
            }
        }
        catch {
            continue;
        }
    }
    return undefined;
}
function latestCodexAssistantText() {
    const sessionsRoot = join(homedir(), ".codex", "sessions");
    if (!existsSync(sessionsRoot))
        return undefined;
    let latestPath;
    let latestMtime = -1;
    for (const path of walkJsonlFiles(sessionsRoot, 4)) {
        try {
            const stat = statSync(path);
            if (stat.mtimeMs > latestMtime) {
                latestMtime = stat.mtimeMs;
                latestPath = path;
            }
        }
        catch {
            continue;
        }
    }
    return transcriptAssistantText(latestPath);
}
function walkJsonlFiles(root, maxDepth) {
    if (maxDepth < 0 || !existsSync(root))
        return [];
    const files = [];
    for (const entry of readdirSync(root)) {
        const path = join(root, entry);
        try {
            const stat = statSync(path);
            if (stat.isDirectory()) {
                files.push(...walkJsonlFiles(path, maxDepth - 1));
            }
            else if (entry.endsWith(".jsonl")) {
                files.push(path);
            }
        }
        catch {
            continue;
        }
    }
    return files;
}
function isAssistantEvent(value) {
    if (!isRecord(value))
        return false;
    if (value.role === "assistant")
        return true;
    return isRecord(value.payload) && value.payload.role === "assistant";
}
const allAgents = ["codex", "claude"];
function parseAgentList(value) {
    const requested = value
        .split(",")
        .map((agent) => agent.trim().toLowerCase())
        .filter(Boolean);
    const expanded = requested.includes("all") ? allAgents : requested;
    const unknown = expanded.filter((agent) => !allAgents.includes(agent));
    if (unknown.length > 0) {
        throw new Error(`Unsupported agent: ${unknown.join(", ")}`);
    }
    return [...new Set(expanded)];
}
function installAgents(agents) {
    return agents.map((agent) => {
        if (agent === "codex")
            return installCodexHooks();
        return installClaudeHooks();
    });
}
function installCodexHooks() {
    mkdirSync(dirname(codexConfigPath()), { recursive: true });
    ensureCodexHooksFeature(codexConfigPath());
    const hooksPath = codexHooksPath();
    const current = readHooksJson(hooksPath);
    const hooks = isRecord(current.hooks) ? current.hooks : {};
    removeOldPsmHooks(hooks);
    addHook(hooks, "UserPromptSubmit", "psm-memory hook context");
    addHook(hooks, "Stop", "psm-memory hook remember");
    mkdirSync(dirname(defaultCodexDbPath()), { recursive: true });
    writeFileSync(hooksPath, `${JSON.stringify({ ...current, hooks }, null, 2)}\n`, "utf8");
    return {
        agent: "codex",
        config: codexConfigPath(),
        hooks: hooksPath,
        commands: ["psm-memory hook context", "psm-memory hook remember"]
    };
}
function installClaudeHooks() {
    const settingsPath = claudeSettingsPath();
    mkdirSync(dirname(settingsPath), { recursive: true });
    const current = readJsonObject(settingsPath);
    const hooks = isRecord(current.hooks) ? current.hooks : {};
    removeOldPsmHooks(hooks);
    addHook(hooks, "UserPromptSubmit", "psm-memory hook context");
    addHook(hooks, "Stop", "psm-memory hook remember", { async: true });
    mkdirSync(dirname(defaultCodexDbPath()), { recursive: true });
    writeFileSync(settingsPath, `${JSON.stringify({ ...current, hooks }, null, 2)}\n`, "utf8");
    return {
        agent: "claude",
        settings: settingsPath,
        commands: ["psm-memory hook context", "psm-memory hook remember"]
    };
}
function ensureCodexHooksFeature(path) {
    if (!existsSync(path)) {
        writeFileSync(path, "[features]\ncodex_hooks = true\n", "utf8");
        return;
    }
    let content = readFileSync(path, "utf8");
    if (/^\s*codex_hooks\s*=\s*true\s*$/m.test(content))
        return;
    if (/^\[features\]\s*$/m.test(content)) {
        content = content.replace(/^(\[features\]\s*)$/m, "$1\ncodex_hooks = true");
    }
    else {
        content = `${content.trimEnd()}\n\n[features]\ncodex_hooks = true\n`;
    }
    writeFileSync(path, content, "utf8");
}
function readHooksJson(path) {
    return readJsonObject(path);
}
function readJsonObject(path) {
    if (!existsSync(path))
        return {};
    const raw = readFileSync(path, "utf8").replace(/^\uFEFF/, "");
    if (!raw.trim())
        return {};
    const parsed = JSON.parse(raw);
    return isRecord(parsed) ? parsed : {};
}
function removeOldPsmHooks(hooks) {
    for (const [event, entries] of Object.entries(hooks)) {
        if (!Array.isArray(entries))
            continue;
        hooks[event] = entries
            .map((entry) => {
            if (!isRecord(entry))
                return entry;
            const childHooks = Array.isArray(entry.hooks) ? entry.hooks : [];
            const filtered = childHooks.filter((hook) => {
                if (!isRecord(hook) || typeof hook.command !== "string")
                    return true;
                return !/psm-codex-hook\.ps1|psm-memory hook (context|remember)/i.test(hook.command);
            });
            return { ...entry, hooks: filtered };
        })
            .filter((entry) => !isRecord(entry) || !Array.isArray(entry.hooks) || entry.hooks.length > 0);
    }
}
function addHook(hooks, event, command, fields = {}) {
    const entries = Array.isArray(hooks[event]) ? hooks[event] : [];
    hooks[event] = [
        ...entries,
        {
            matcher: "*",
            hooks: [{ type: "command", command, ...fields }]
        }
    ];
}
function defaultCodexDbPath() {
    return process.env.PSM_MEMORY_DB ?? join(homedir(), ".codex", "memories", "psm-memory.db");
}
function codexConfigPath() {
    return join(homedir(), ".codex", "config.toml");
}
function codexHooksPath() {
    return join(homedir(), ".codex", "hooks.json");
}
function claudeSettingsPath() {
    return join(homedir(), ".claude", "settings.json");
}
function defaultHookLogPath() {
    return process.env.PSM_MEMORY_HOOK_LOG ?? join(dirname(defaultCodexDbPath()), "psm-memory-hooks.jsonl");
}
function buildReviewLog(options) {
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
function readHookEvents(path, date, limit) {
    if (!existsSync(path))
        return [];
    return readFileSync(path, "utf8")
        .split(/\r?\n/)
        .filter((line) => line.trim().length > 0)
        .map((line) => {
        try {
            const parsed = JSON.parse(line);
            return isRecord(parsed) ? parsed : {};
        }
        catch {
            return {};
        }
    })
        .filter((event) => typeof event.ts === "string" && event.ts.startsWith(date))
        .slice(-limit);
}
function readDecisionRows(dbPath, date, limit) {
    if (!existsSync(dbPath))
        return [];
    const store = new MemoryStore(dbPath);
    try {
        store.initializeSchema();
        return store.selectTable("decisions", Math.max(limit * 4, 100))
            .filter((row) => typeof row.created_at === "string" && row.created_at.startsWith(date))
            .slice(0, limit)
            .map(summarizeDecisionRow);
    }
    finally {
        store.close();
    }
}
function summarizeDecisionRow(row) {
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
function parseJsonObject(value) {
    try {
        const parsed = JSON.parse(value);
        return isRecord(parsed) ? parsed : {};
    }
    catch {
        return {};
    }
}
function renderReviewLog(review) {
    const lines = [
        `PSM Review Log ${review.date}`,
        `DB: ${review.db}`,
        `Hook log: ${review.log_path}`,
        "",
        "Hook events:"
    ];
    if (review.hook_events.length === 0) {
        lines.push("- none");
    }
    else {
        for (const event of review.hook_events) {
            lines.push(`- ${event.ts ?? ""} ${event.hook ?? "hook"} ${event.status ?? ""} ${event.reason ? `(${event.reason})` : ""} total=${formatTiming(event, "total")} model=${formatTiming(event, "model_context") || formatTiming(event, "model_remember")}`);
        }
    }
    lines.push("", "Memory decisions:");
    if (review.decisions.length === 0) {
        lines.push("- none");
    }
    else {
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
function formatTiming(event, key) {
    const timings = isRecord(event.timings_ms) ? event.timings_ms : {};
    const value = timings[key];
    return typeof value === "number" ? `${value}ms` : "";
}
function writeHookAudit(entry) {
    try {
        const path = defaultHookLogPath();
        mkdirSync(dirname(path), { recursive: true });
        appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf8");
    }
    catch {
        // Hook logging must never break the agent workflow.
    }
}
function errorMessage(error) {
    return error instanceof Error ? error.message : String(error);
}
function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function output(value, pretty) {
    process.stdout.write(`${JSON.stringify(value, null, pretty ? 2 : 0)}\n`);
}
function write(value) {
    process.stdout.write(value);
}
function helpText() {
    return `PSM Memory commands:
  setup [--force] [--pretty]
  install-agent --agent codex|claude|all[,agent...] [--pretty]
  hook context|remember
  init --db <path>
  context --prompt <text> --user <id> --db <path> [--pretty]
  remember --llm-response <text> --user <id> --db <path> [--pretty]
  recall --question <text> --user <id> --db <path> [--pretty]
  review-log --db <path> [--log <path>] [--date YYYY-MM-DD] [--pretty]
  show --table episodic|semantic|archival|conflicts|decisions|decay_schedule --db <path> [--limit n] [--pretty]
  conflicts --status unresolved|resolved|dismissed --db <path> [--limit n] [--pretty]

The model downloads automatically during npm install. If that was skipped or failed, run "psm-memory setup".
JSON output is the default. Run "psm-memory advanced" for advanced runtime options.
`;
}
function advancedHelpText() {
    return `PSM Memory advanced options:
  --top-k <n>             Number of memories to retrieve for context or recall. Default: 5.
  --embedding-model <id>  Hugging Face embedding model. Default: ${defaultEmbeddingModel}.
  --no-embeddings         Disable vector search and use lexical retrieval only.
  setup --skip-embeddings Skip embedding model preparation during setup.
  --context-size <n>      Context size for the local GGUF runtime. Default: 4096.
  --gpu <mode>            GPU backend for node-llama-cpp. Default: auto.
  --gpu-layers <mode>     GPU layer offload for node-llama-cpp. Default: auto.

Default model path:
  ${defaultModelPath()}

The npm package downloads the default Q4_K_M GGUF model during install. If that download was skipped or failed, run "psm-memory setup".
`;
}
//# sourceMappingURL=index.js.map