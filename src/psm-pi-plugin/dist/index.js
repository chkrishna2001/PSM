import { MemoryStore, NodeLlamaRuntime, PsmService } from "@psm-memory/sdk";
export function createPsmTools(options) {
    const { service, defaultUser } = createService(options);
    return {
        "psm.context": async (input) => service.context({
            prompt: requireString(input, "prompt"),
            userId: stringOr(input.user, defaultUser),
            topK: numberOr(input.top_k, options.topK ?? 5)
        }),
        "psm.remember": async (input) => service.remember({
            llmResponse: requireString(input, "llm_response"),
            userId: stringOr(input.user, defaultUser)
        }),
        "psm.recall": async (input) => service.recall({
            question: requireString(input, "question"),
            userId: stringOr(input.user, defaultUser),
            topK: numberOr(input.top_k, options.topK ?? 5)
        })
    };
}
export function createPsmHooks(options) {
    const { store, service, defaultUser } = createService(options);
    const pending = new Set();
    const enrichPrompt = async (input) => {
        const prompt = requireValue(input.prompt, "prompt");
        const userId = stringOr(input.userId, defaultUser);
        const rawContext = await service.context({
            prompt,
            userId,
            topK: input.topK ?? options.topK ?? 5
        });
        const memoryContext = renderMemoryContext(rawContext);
        const contextMessage = memoryContext ? {
            role: "system",
            content: memoryContext
        } : null;
        return {
            userId,
            prompt,
            contextMessage,
            messages: contextMessage ? [contextMessage, { role: "user", content: prompt }] : [{ role: "user", content: prompt }],
            memoryContext,
            rawContext
        };
    };
    const rememberResponse = (input) => {
        const llmResponse = renderResponseForStorage(input);
        if (!llmResponse)
            return;
        const task = service.remember({
            llmResponse,
            userId: stringOr(input.userId, defaultUser)
        }).then(() => undefined);
        pending.add(task);
        task.catch((error) => {
            options.onMemoryWriteError?.(error);
        }).finally(() => {
            pending.delete(task);
        });
    };
    return {
        enrichPrompt,
        rememberResponse,
        beforePrompt: enrichPrompt,
        afterResponse: rememberResponse,
        async flush() {
            await Promise.allSettled([...pending]);
        },
        async close() {
            await Promise.allSettled([...pending]);
            store.close();
        }
    };
}
function createService(options) {
    const store = new MemoryStore(options.dbPath);
    store.initializeSchema();
    const service = new PsmService(store, resolveRuntime(options));
    const defaultUser = options.userId ?? "default-user";
    return { store, service, defaultUser };
}
function resolveRuntime(options) {
    if (options.runtime)
        return options.runtime;
    if (options.modelPath)
        return new NodeLlamaRuntime({ modelPath: options.modelPath });
    throw new Error("PSM model runtime is required. Pass runtime or modelPath to createPsmHooks.");
}
function renderMemoryContext(rawContext) {
    const memories = Array.isArray(rawContext.memory_context) ? rawContext.memory_context : [];
    if (memories.length === 0)
        return "";
    const lines = memories.map((memory, index) => {
        const item = memory;
        const table = typeof item.table === "string" ? item.table : "memory";
        const content = typeof item.content === "string" ? item.content : "";
        return `${index + 1}. [${table}] ${content}`;
    }).filter((line) => line.trim());
    if (lines.length === 0)
        return "";
    return [
        "PSM Memory Context",
        "Use these retrieved memories as private context. Do not mention this block unless the user asks about memory.",
        "",
        ...lines
    ].join("\n");
}
function renderResponseForStorage(input) {
    const parts = [];
    if (typeof input.response === "string" && input.response.trim()) {
        parts.push(`LLM response:\n${input.response.trim()}`);
    }
    if (input.decision !== undefined && input.decision !== null) {
        parts.push(`LLM decision:\n${stringifyDecision(input.decision)}`);
    }
    return parts.join("\n\n");
}
function stringifyDecision(value) {
    if (typeof value === "string")
        return value.trim();
    try {
        return JSON.stringify(value);
    }
    catch {
        return String(value);
    }
}
function requireValue(value, key) {
    if (typeof value === "string" && value.trim())
        return value;
    throw new Error(`Missing required hook input: ${key}`);
}
function requireString(input, key) {
    const value = input[key];
    if (typeof value === "string" && value.trim())
        return value;
    throw new Error(`Missing required tool input: ${key}`);
}
function stringOr(value, fallback) {
    return typeof value === "string" && value.trim() ? value : fallback;
}
function numberOr(value, fallback) {
    const parsed = typeof value === "number" ? value : Number(value);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
//# sourceMappingURL=index.js.map