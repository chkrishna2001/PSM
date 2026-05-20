import { normalizeAction } from "./actions.js";
import { memoryTables, type ContextItem, type ContextRender, type MemoryFactPayload, type MemoryPayload, type MemoryTable, type RecallPlan, type StorageDecision } from "./types.js";

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
      facts: normalizeFacts(parsed.facts),
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

function normalizeFacts(value: unknown): MemoryFactPayload[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(normalizeFact)
    .filter((fact): fact is MemoryFactPayload => fact !== null);
}

function normalizeFact(value: unknown): MemoryFactPayload | null {
  if (!isRecord(value)) return null;
  const subject = stringOrUndefined(value.subject);
  const predicate = normalizePredicate(stringOrUndefined(value.predicate));
  const valueText = stringOrUndefined(value.value_text) ?? valueToText(value.value);
  const inferenceKind = stringOrUndefined(value.inference_kind);
  const evidenceText = stringOrUndefined(value.evidence_text);
  if (!subject || !predicate || !valueText) return null;
  if (inferenceKind && inferenceKind !== "explicit") return null;
  if (!evidenceText) return null;
  return {
    subject,
    predicate,
    object: stringOrUndefined(value.object),
    value: value.value,
    value_text: valueText,
    value_json: value.value_json ?? value.value,
    fact_type: stringOrUndefined(value.fact_type),
    confidence: numberOrUndefined(value.confidence),
    inference_kind: inferenceKind ?? "explicit",
    evidence_text: evidenceText,
    temporal_expression: stringOrUndefined(value.temporal_expression),
    resolved_time: stringOrUndefined(value.resolved_time),
    resolved_time_confidence: numberOrUndefined(value.resolved_time_confidence)
  };
}

export function parseRecallPlan(rawText: string, question: string, topK = 5): RecallPlan {
  const raw_json = extractJsonObject(rawText) ?? rawText.trim();
  try {
    const parsed = JSON.parse(raw_json) as Record<string, unknown>;
    const targetTables = normalizeTables(parsed.target_tables);
    return {
      intent: stringOr(parsed.intent, "recall"),
      target_tables: targetTables.tables,
      filters: isRecord(parsed.filters) ? parsed.filters : {},
      ranking_hints: stringArray(parsed.ranking_hints),
      temporal_intent: stringOrUndefined(parsed.temporal_intent),
      top_k: Math.min(positiveInt(parsed.top_k, topK), topK),
      raw_json,
      plan_fallback: targetTables.fallback
    };
  } catch (error) {
    return {
      intent: "recall",
      target_tables: ["semantic", "episodic"],
      filters: {},
      ranking_hints: keywords(question),
      temporal_intent: undefined,
      top_k: topK,
      raw_json,
      plan_fallback: true,
      parse_error: error instanceof Error ? error.message : String(error)
    };
  }
}

export function parseContextRender(rawText: string, topK = 5): ContextRender {
  const raw_json = extractJsonObject(rawText) ?? rawText.trim();
  try {
    const parsed = JSON.parse(raw_json) as Record<string, unknown>;
    const selected_ids = stringArray(parsed.selected_ids).slice(0, topK);
    const rawItems = Array.isArray(parsed.context_items)
      ? parsed.context_items
      : Array.isArray(parsed.memory_context)
        ? parsed.memory_context
        : [];
    const context_items = rawItems
      .map(normalizeContextItem)
      .filter((item): item is ContextItem => item !== null)
      .slice(0, topK);
    return {
      context_items,
      selected_ids,
      reasoning: stringOr(parsed.reasoning, "PSM selected context items."),
      raw_json
    };
  } catch (error) {
    return {
      context_items: [],
      reasoning: `Model returned invalid context JSON. ${error instanceof Error ? error.message : String(error)}`,
      raw_json,
      parse_error: error instanceof Error ? error.message : String(error)
    };
  }
}

function normalizeMemory(value: unknown, fallbackContent: string): MemoryPayload | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" && value.trim()) {
    return { content: value };
  }
  if (!isRecord(value)) return null;
  const content = stringOrUndefined(value.content);
  if (!content) return null;
  return {
    content,
    type: stringOrUndefined(value.type),
    strength: numberOrUndefined(value.strength),
    decay_rate: numberOrUndefined(value.decay_rate),
    emotional_weight: numberOrUndefined(value.emotional_weight),
    confidence: numberOrUndefined(value.confidence),
    tags: stringArray(value.tags),
    source_episodes: stringArray(value.source_episodes),
    source_kind: stringOrUndefined(value.source_kind),
    source_id: stringOrUndefined(value.source_id),
    source_timestamp: stringOrUndefined(value.source_timestamp),
    source_label: stringOrUndefined(value.source_label),
    temporal_expression: stringOrUndefined(value.temporal_expression),
    resolved_time: stringOrUndefined(value.resolved_time),
    resolved_time_confidence: numberOrUndefined(value.resolved_time_confidence)
  };
}

function normalizeContextItem(value: unknown): ContextItem | null {
  if (typeof value === "string" && value.trim()) {
    return { table: "memory", content: value };
  }
  if (!isRecord(value)) return null;
  const content = stringOrUndefined(value.content);
  if (!content) return null;
  const table = stringOrUndefined(value.table);
  return {
    id: stringOrUndefined(value.id),
    table: table === "episodic" || table === "semantic" || table === "archival" || table === "memory_fact" ? table : "memory",
    content,
    reason: stringOrUndefined(value.reason)
  };
}

function normalizeTables(value: unknown): { tables: MemoryTable[]; fallback: boolean } {
  const allowed = new Set<string>(["semantic", "episodic", "archival"]);
  const tables = stringArray(value).filter((table): table is MemoryTable => allowed.has(table));
  return tables.length > 0 ? { tables, fallback: false } : { tables: ["semantic", "episodic"], fallback: true };
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

function valueToText(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return undefined;
}

function normalizePredicate(value: string | undefined): string | undefined {
  if (!value) return undefined;
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || undefined;
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
