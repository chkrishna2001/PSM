import { tokenize } from "./ranking.js";
import type { IndexableKind, IndexablePayload, IndexableRecord, MemoryFactPayload } from "./types.js";

const WORKFLOW_HEADER_PATTERNS: Array<{ pattern: RegExp; key: string }> = [
  { pattern: /review.*pull request|pull request review/i, key: "review-pr" },
  { pattern: /runpod.*train|gpu train|train.*runpod/i, key: "runpod-gpu-train" },
  { pattern: /grounding bar|promotion bar/i, key: "grounding-bar" }
];

export interface BuildIndexablesInput {
  llmResponse: string;
  memoryContent: string;
  tags?: string[];
  memoryTable?: string;
  memoryId?: string;
  facts?: MemoryFactPayload[];
  explicitIndexables?: IndexablePayload[];
}

export function buildIndexablesForRemember(input: BuildIndexablesInput): IndexablePayload[] {
  if (input.explicitIndexables?.length) {
    return input.explicitIndexables
      .map((row) => normalizeIndexable(row, input))
      .filter((row): row is IndexablePayload => row !== null);
  }

  const sourceText = input.llmResponse.trim() || input.memoryContent.trim();
  if (!sourceText) return [];

  const workflowKey = inferWorkflowKey(sourceText, input.tags ?? []);
  const steps = extractWorkflowSteps(sourceText);
  if (workflowKey && steps.length >= 2) {
    return [{
      kind: "workflow",
      key: workflowKey,
      target_memory_table: input.memoryTable,
      target_memory_id: input.memoryId,
      steps,
      salience: 0.95,
      reconstructive_hint: reconstructiveHint(sourceText),
      evidence_text: sourceText.slice(0, 500),
      tags: uniqueTags([`workflow:${workflowKey}`, "workflow", ...(input.tags ?? [])])
    }];
  }

  const content = cleanText(input.memoryContent || sourceText);
  const mnemonicKey = buildMnemonicKey(content, input.tags ?? []);
  const rows: IndexablePayload[] = [{
    kind: "mnemonic",
    key: mnemonicKey,
    target_memory_table: input.memoryTable,
    target_memory_id: input.memoryId,
    salience: salienceFor(content, input.tags ?? []),
    reconstructive_hint: reconstructiveHint(content),
    evidence_text: content,
    tags: uniqueTags(input.tags ?? []).slice(0, 6)
  }];

  const factKey = buildFactAnchorKey(input.facts ?? []);
  if (factKey && factKey !== mnemonicKey) {
    rows.push({
      kind: "fact_anchor",
      key: factKey,
      target_memory_table: input.memoryTable,
      target_memory_id: input.memoryId,
      salience: Math.max(rows[0].salience ?? 0.8, 0.82),
      reconstructive_hint: reconstructiveHint(content),
      evidence_text: content,
      tags: uniqueTags(input.tags ?? []).slice(0, 6)
    });
  }
  return rows;
}

export function inferWorkflowKey(text: string, tags: string[] = []): string | null {
  for (const tag of tags) {
    const match = String(tag).match(/^workflow:([a-z0-9-]+)$/i);
    if (match) return match[1].toLowerCase();
  }
  const header = text.match(/^#\s+(.+)$/m)?.[1] ?? "";
  const haystack = `${header}\n${text.slice(0, 240)}`;
  for (const entry of WORKFLOW_HEADER_PATTERNS) {
    if (entry.pattern.test(haystack)) return entry.key;
  }
  return null;
}

export function extractWorkflowSteps(text: string): string[] {
  const steps: string[] = [];
  for (const line of text.split("\n")) {
    const match = line.match(/^\s*\d+\.\s+(.+?)\s*$/);
    if (!match?.[1]) continue;
    steps.push(stepToId(match[1]));
  }
  return unique(steps);
}

export function normalizeRecallKey(query: string): string {
  return query
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export interface ScoredIndexable extends IndexableRecord {
  score: number;
}

export function rankIndexables(query: string, indexables: IndexableRecord[], topK = 5): ScoredIndexable[] {
  const normalized = normalizeRecallKey(query);
  const tokens = tokenize(query);
  return indexables
    .map((row) => ({ ...row, score: indexableScore(normalized, tokens, row) }))
    .filter((row) => row.score >= 0.35)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function indexableScore(normalizedQuery: string, queryTokens: string[], row: IndexableRecord): number {
  let score = row.salience ?? 0.5;
  const key = row.key.toLowerCase();
  if (key && (normalizedQuery === key || normalizedQuery.includes(key) || key.includes(normalizedQuery))) {
    score += 0.55;
  }
  if (row.kind === "workflow" && /review|workflow|procedure|how do i/i.test(normalizedQuery.replace(/-/g, " "))) {
    score += 0.1;
  }
  const searchable = [row.key, row.reconstructive_hint ?? "", ...(row.tags ?? []), ...(row.steps ?? [])].join(" ");
  const haystack = new Set(tokenize(searchable));
  const overlap = queryTokens.filter((token) => haystack.has(token)).length;
  if (queryTokens.length > 0) {
    score += overlap / queryTokens.length * 0.35;
  }
  return Number(score.toFixed(6));
}

function normalizeIndexable(row: IndexablePayload, input: BuildIndexablesInput): IndexablePayload | null {
  const key = cleanKey(row.key);
  if (!key) return null;
  return {
    kind: row.kind ?? "mnemonic",
    key,
    target_memory_table: row.target_memory_table ?? input.memoryTable,
    target_memory_id: row.target_memory_id ?? input.memoryId,
    steps: Array.isArray(row.steps) ? row.steps.map((step) => stepToId(String(step))) : undefined,
    salience: clamp01(row.salience ?? 0.85),
    reconstructive_hint: row.reconstructive_hint ?? reconstructiveHint(input.memoryContent),
    evidence_text: row.evidence_text ?? input.memoryContent,
    tags: uniqueTags(row.tags ?? input.tags ?? [])
  };
}

function buildMnemonicKey(content: string, tags: string[]): string {
  const tagTokens = tags.flatMap((tag) => meaningfulTokens(String(tag).replace(/_/g, " ")));
  const contentTokens = meaningfulTokens(content);
  const tokens = unique([...tagTokens, ...contentTokens]).slice(0, 4);
  return tokens.length > 0 ? tokens.join("-") : "memory-anchor";
}

function buildFactAnchorKey(facts: MemoryFactPayload[]): string {
  const fact = facts.find((item) => item.subject && item.predicate && item.value_text);
  if (!fact) return "";
  return unique([
    ...meaningfulTokens(fact.subject ?? ""),
    ...meaningfulTokens(fact.predicate ?? ""),
    ...meaningfulTokens(fact.value_text ?? "")
  ]).slice(0, 4).join("-");
}

function stepToId(step: string): string {
  return step
    .toLowerCase()
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || "step";
}

function salienceFor(content: string, tags: string[]): number {
  const lower = content.toLowerCase();
  let score = 0.68;
  if (/\b\d{4}\b|yesterday|last week|workflow|review|procedure/.test(lower)) score += 0.08;
  if (/decision|prefer|constraint|indexable|mnemonic|recall/.test(lower)) score += 0.12;
  if (tags.length > 0) score += 0.04;
  return Number(Math.min(score, 0.98).toFixed(2));
}

function reconstructiveHint(content: string): string {
  const sentence = content.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? content;
  return sentence.length <= 160 ? sentence : `${sentence.slice(0, 157).trim()}...`;
}

function meaningfulTokens(text: string): string[] {
  const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "into", "said", "user", "memory"]);
  return cleanText(text)
    .toLowerCase()
    .match(/[a-z0-9]+/g)
    ?.filter((token) => token.length > 2 && !stop.has(token)) ?? [];
}

function cleanText(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function cleanKey(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "");
}

function uniqueTags(tags: string[]): string[] {
  return [...new Set(tags.map((tag) => String(tag).trim()).filter(Boolean).map((tag) => tag.replace(/\s+/g, "_")))];
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
