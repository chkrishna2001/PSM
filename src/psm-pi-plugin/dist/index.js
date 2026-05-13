import { HeuristicRuntime, MemoryStore, PsmService } from "psm-sdk";
export function createPsmTools(options) {
    const store = new MemoryStore(options.dbPath);
    store.initializeSchema();
    const service = new PsmService(store, options.runtime ?? new HeuristicRuntime());
    const defaultUser = options.userId ?? "default-user";
    return {
        "psm.context": async (input) => service.context({
            prompt: requireString(input, "prompt"),
            userId: stringOr(input.user, defaultUser),
            topK: numberOr(input.top_k, 5)
        }),
        "psm.remember": async (input) => service.remember({
            llmResponse: requireString(input, "llm_response"),
            userId: stringOr(input.user, defaultUser)
        }),
        "psm.recall": async (input) => service.recall({
            question: requireString(input, "question"),
            userId: stringOr(input.user, defaultUser),
            topK: numberOr(input.top_k, 5)
        })
    };
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