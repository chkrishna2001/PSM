import { normalizeAction } from "./actions.js";
import { memoryTables, type MemoryPayload, type MemoryTable, type RecallPlan, type StorageDecision } from "./types.js";

export function extractJsonObject(text: string): string | null {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  return text.slice(start, end + 1);
}

export function parseStorageDecision(rawText: string, fallbackContent: string, fallbackAction = "store_episodic"): StorageDecision {
  const raw_json = extractJsonObject(rawText) ?? rawText.trim();
  try {
    const parsed = JSON.parse(raw_json) as Record<string, unknown>;
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
  } catch (error) {
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

export function parseRecallPlan(rawText: string, question: string, topK = 5): RecallPlan {
  const raw_json = extractJsonObject(rawText) ?? rawText.trim();
  try {
    const parsed = JSON.parse(raw_json) as Record<string, unknown>;
    return {
      intent: stringOr(parsed.intent, "recall"),
      target_tables: normalizeTables(parsed.target_tables),
      filters: isRecord(parsed.filters) ? parsed.filters : {},
      ranking_hints: stringArray(parsed.ranking_hints),
      top_k: positiveInt(parsed.top_k, topK),
      raw_json
    };
  } catch (error) {
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

function normalizeMemory(value: unknown, fallbackContent: string): MemoryPayload | null {
  if (value === null) return null;
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

function normalizeTables(value: unknown): MemoryTable[] {
  const allowed = new Set<string>(memoryTables);
  const tables = stringArray(value).filter((table): table is MemoryTable => allowed.has(table));
  return tables.length > 0 ? tables : ["semantic", "episodic"];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item)).filter((item) => item.trim().length > 0);
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function numberOrUndefined(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
  return undefined;
}

function positiveInt(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function keywords(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]{3,}/g)?.slice(0, 12) ?? [];
}
