import { normalizeAction } from "./actions.js";
import { memoryTables } from "./types.js";
export function extractJsonObject(text) {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start < 0 || end <= start)
        return null;
    return text.slice(start, end + 1);
}
export function parseStorageDecision(rawText, fallbackContent, fallbackAction = "store_episodic") {
    const raw_json = extractJsonObject(rawText) ?? rawText.trim();
    try {
        const parsed = JSON.parse(raw_json);
        const memory = normalizeMemory(parsed.memory, fallbackContent);
        return {
            action: normalizeAction(parsed.action ?? fallbackAction),
            memory,
            reasoning: stringOr(parsed.reasoning, "Model output missing explicit reasoning; applied parser defaults."),
            confidence: numberOrUndefined(parsed.confidence ?? memory?.confidence),
            emotional_weight: numberOrUndefined(parsed.emotional_weight ?? memory?.emotional_weight),
            contradiction_score: numberOrUndefined(parsed.contradiction_score),
            raw_json
        };
    }
    catch (error) {
        return {
            action: normalizeAction(fallbackAction),
            memory: {
                content: fallbackContent,
                type: "episodic",
                confidence: 0.5,
                emotional_weight: 0.1,
                tags: ["parse_fallback"]
            },
            reasoning: `Model returned invalid JSON; stored fallback content. ${error instanceof Error ? error.message : String(error)}`,
            confidence: 0.5,
            emotional_weight: 0.1,
            raw_json,
            parse_error: error instanceof Error ? error.message : String(error)
        };
    }
}
export function parseRecallPlan(rawText, question, topK = 5) {
    const raw_json = extractJsonObject(rawText) ?? rawText.trim();
    try {
        const parsed = JSON.parse(raw_json);
        return {
            intent: stringOr(parsed.intent, "recall"),
            target_tables: normalizeTables(parsed.target_tables),
            filters: isRecord(parsed.filters) ? parsed.filters : {},
            ranking_hints: stringArray(parsed.ranking_hints),
            top_k: positiveInt(parsed.top_k, topK),
            raw_json
        };
    }
    catch (error) {
        return {
            intent: "recall",
            target_tables: ["semantic", "episodic"],
            filters: {},
            ranking_hints: keywords(question),
            top_k: topK,
            raw_json,
            parse_error: error instanceof Error ? error.message : String(error)
        };
    }
}
function normalizeMemory(value, fallbackContent) {
    if (value === null)
        return null;
    if (!isRecord(value)) {
        return { content: fallbackContent };
    }
    return {
        content: stringOr(value.content, fallbackContent),
        type: stringOrUndefined(value.type),
        strength: numberOrUndefined(value.strength),
        decay_rate: numberOrUndefined(value.decay_rate),
        emotional_weight: numberOrUndefined(value.emotional_weight),
        confidence: numberOrUndefined(value.confidence),
        tags: stringArray(value.tags),
        source_episodes: stringArray(value.source_episodes)
    };
}
function normalizeTables(value) {
    const allowed = new Set(memoryTables);
    const tables = stringArray(value).filter((table) => allowed.has(table));
    return tables.length > 0 ? tables : ["semantic", "episodic"];
}
function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function stringArray(value) {
    if (!Array.isArray(value))
        return [];
    return value.map((item) => String(item)).filter((item) => item.trim().length > 0);
}
function stringOr(value, fallback) {
    return typeof value === "string" && value.trim() ? value : fallback;
}
function stringOrUndefined(value) {
    return typeof value === "string" && value.trim() ? value : undefined;
}
function numberOrUndefined(value) {
    if (typeof value === "number" && Number.isFinite(value))
        return value;
    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value)))
        return Number(value);
    return undefined;
}
function positiveInt(value, fallback) {
    const parsed = typeof value === "number" ? value : Number(value);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
function keywords(text) {
    return text.toLowerCase().match(/[a-z0-9]{3,}/g)?.slice(0, 12) ?? [];
}
//# sourceMappingURL=json.js.map