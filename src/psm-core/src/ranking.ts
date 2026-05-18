import type { MemoryRecord, RankedMemory } from "./types.js";

const stopwords = new Set([
  "the", "and", "for", "that", "this", "with", "you", "your", "what", "when", "where", "why", "how", "are",
  "was", "were", "has", "have", "had", "from", "about", "into", "onto", "then", "than", "they", "them",
  "does", "did", "doing", "done", "will", "would", "could", "should", "their", "there", "here", "also"
]);

export interface HybridRankOptions {
  topK: number;
  vectorScores?: Map<string, number>;
  preferredTables?: MemoryRecord["table"][];
  minScore?: number;
}

export function rankMemories(query: string, memories: MemoryRecord[], topK: number): RankedMemory[] {
  return hybridRankMemories(query, memories, { topK });
}

export function hybridRankMemories(query: string, memories: MemoryRecord[], options: HybridRankOptions): RankedMemory[] {
  const qTokens = tokenize(query);
  const qNumbers = numbers(query);
  const qEntities = entityTokens(query);
  const temporalQuestion = /\bwhen\b|\bdate\b|\byear\b|\bmonth\b|\bday\b|\btime\b/i.test(query);
  const preferredTables = new Set(options.preferredTables ?? []);
  const ranked = dedupe(memories).map((memory) => {
    const tags = parseJson(memory.tags);
    const sourceEpisodes = parseJson(memory.source_episodes);
    const searchable = [
      memory.content,
      Array.isArray(tags) ? tags.join(" ") : "",
      memory.source_kind ?? "",
      memory.source_id ?? "",
      memory.source_label ?? "",
      memory.temporal_expression ?? "",
      memory.resolved_time ?? "",
      memory.source_timestamp ?? ""
    ].join(" ");
    const memoryTokens = tokenize(searchable);
    const memorySet = new Set(memoryTokens);
    const vectorScore = options.vectorScores?.get(memoryKey(memory)) ?? 0;
    const exactCoverage = qTokens.length === 0 ? 0 : qTokens.filter((token) => memorySet.has(token)).length / qTokens.length;
    const rareExact = qTokens.filter((token) => token.length >= 5 && memorySet.has(token)).length;
    const numberScore = overlapRatio(qNumbers, numbers(searchable));
    const temporalScore = temporalQuestion ? temporalSignal(searchable) : 0;
    const entityScore = overlapRatio(qEntities, memoryTokens);
    const tagScore = lexicalScore(qTokens, tokenize(Array.isArray(tags) ? tags.join(" ") : ""));
    const tableBoost = preferredTables.has(memory.table) ? 0.08 : 0;
    const score =
      0.9 * lexicalScore(qTokens, memoryTokens) +
      0.75 * exactCoverage +
      0.18 * rareExact +
      0.55 * numberScore +
      0.25 * temporalScore +
      0.4 * entityScore +
      0.25 * tagScore +
      0.8 * vectorScore +
      tableBoost +
      0.12 * (memory.confidence ?? 0.5) +
      0.08 * (memory.strength ?? 0.5) +
      0.04 * (memory.table === "semantic" ? 1 : 0);
    return {
      ...memory,
      score: Number(score.toFixed(6)),
      metadata: {
        tags,
        source_episodes: sourceEpisodes,
        ranking: {
          lexical: Number(lexicalScore(qTokens, memoryTokens).toFixed(6)),
          exact_coverage: Number(exactCoverage.toFixed(6)),
          rare_exact: rareExact,
          number: Number(numberScore.toFixed(6)),
          temporal: Number(temporalScore.toFixed(6)),
          entity: Number(entityScore.toFixed(6)),
          tag: Number(tagScore.toFixed(6)),
          vector: Number(vectorScore.toFixed(6)),
          preferred_table: tableBoost > 0
        }
      }
    };
  });
  return ranked
    .filter((memory) => memory.score >= (options.minScore ?? 0))
    .sort((a, b) => b.score - a.score)
    .slice(0, options.topK);
}

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .match(/[a-z0-9]+/g)
    ?.map(normalizeToken)
    .filter((token) => token.length > 2 && !stopwords.has(token)) ?? [];
}

function lexicalScore(queryTokens: string[], memoryTokens: string[]): number {
  if (queryTokens.length === 0 || memoryTokens.length === 0) return 0;
  const memorySet = new Set(memoryTokens);
  const overlap = queryTokens.filter((token) => memorySet.has(token)).length;
  return overlap / Math.sqrt(queryTokens.length * memoryTokens.length);
}

function overlapRatio(queryValues: string[], memoryValues: string[]): number {
  if (queryValues.length === 0 || memoryValues.length === 0) return 0;
  const memorySet = new Set(memoryValues);
  return queryValues.filter((value) => memorySet.has(value)).length / queryValues.length;
}

function parseJson(value: string | null | undefined): unknown {
  if (!value) return [];
  try {
    return JSON.parse(value);
  } catch {
    return [];
  }
}

function normalizeToken(token: string): string {
  if (token.endsWith("ies") && token.length > 4) return `${token.slice(0, -3)}y`;
  if (token.endsWith("es") && token.length > 4) return token.slice(0, -2);
  if (token.endsWith("s") && token.length > 3) return token.slice(0, -1);
  return token;
}

function numbers(text: string): string[] {
  return text.match(/\b\d{2,4}\b/g) ?? [];
}

function temporalSignal(text: string): number {
  if (/\b\d{4}\b/.test(text)) return 1;
  if (/\b\d{1,2}\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b/i.test(text)) return 0.9;
  if (/\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b/i.test(text)) return 0.7;
  if (/\b(yesterday|today|tomorrow|last year|last week|last month|next year|next week|next month)\b/i.test(text)) return 0.45;
  return 0;
}

function entityTokens(text: string): string[] {
  return text.match(/\b[A-Z][a-zA-Z]{2,}\b/g)?.map((token) => normalizeToken(token.toLowerCase())) ?? [];
}

function memoryKey(memory: MemoryRecord): string {
  return `${memory.table}:${memory.id}`;
}

function dedupe(memories: MemoryRecord[]): MemoryRecord[] {
  const seen = new Set<string>();
  const result: MemoryRecord[] = [];
  for (const memory of memories) {
    const key = memoryKey(memory);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(memory);
  }
  return result;
}
