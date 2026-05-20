import { fallbackAgentContextItems, renderAgentMemoryContext } from "./context.js";
import { buildContextPlanPrompt, buildContextRenderPrompt, buildRecallPlanPrompt, buildStoragePrompt, buildStorageRepairPrompt } from "./prompts.js";
import { parseContextRender, parseRecallPlan, parseStorageDecision } from "./json.js";
import { hybridRankMemories, tokenize } from "./ranking.js";
import { MemoryStore } from "./store.js";
import { normalizeFactTemporalFields, normalizeMemoryTemporalFields } from "./temporal.js";
import type { ContextItem, ContextRequest, EmbeddingRuntime, MemoryFactRecord, MemoryRecord, ModelRuntime, RankedMemory, RecallRequest, RememberRequest, WrittenMemoryRef } from "./types.js";

export class PsmService {
  constructor(
    private readonly store: MemoryStore,
    private readonly runtime: ModelRuntime,
    private readonly embeddings?: { model: string; runtime: EmbeddingRuntime }
  ) {}

  async context(request: ContextRequest): Promise<Record<string, unknown>> {
    const topK = request.topK ?? 5;
    const raw = await this.runtime.generateJson(buildContextPlanPrompt(request.prompt, topK), { temperature: 0, maxTokens: 256 });
    const plan = parseRecallPlan(raw, request.prompt, topK);
    const recallTables = memoryTablesForRecall(plan.target_tables);
    const searchQuery = [...plan.ranking_hints, request.prompt].join(" ");
    const candidates = await this.contextCandidates(request.userId, searchQuery, recallTables, Math.max(100, plan.top_k * 10));
    const ranked = hybridRankMemories(searchQuery, candidates.memories, {
      topK: plan.top_k,
      vectorScores: candidates.vectorScores,
      preferredTables: recallTables,
      minScore: 0.15
    });
    const rankedFacts = rankFacts(searchQuery, candidates.facts, plan.top_k);
    this.store.updateAccess(ranked);
    const contextItems = exactContextItems(ranked, plan.top_k, "Exact DB-backed memory context.", rankedFacts);
    const rendered = await this.renderContext(request.prompt, contextItems, plan.top_k);
    return {
      user_id: request.userId,
      prompt: request.prompt,
      recall_plan: plan,
      context_items: contextItems,
      agent_context_items: rendered.items,
      agent_context: rendered.context,
      agent_context_reasoning: rendered.reasoning,
      agent_context_parse_error: rendered.parse_error,
      context_reasoning: "Context is retrieved from exact DB-backed memory rows. Agent context may be model-shaped, but only from selected grounded rows.",
      context_parse_error: plan.parse_error,
      plan_fallback: plan.plan_fallback === true,
      memory_context: ranked.map(memoryContextRow),
      fact_context: rankedFacts.map(factContextRow),
      grounding: {
        mode: "grounded_db_rows",
        generated_text_allowed: true,
        item_count: contextItems.length
      }
    };
  }

  async recall(request: RecallRequest): Promise<Record<string, unknown>> {
    const topK = request.topK ?? 5;
    const raw = await this.runtime.generateJson(buildRecallPlanPrompt(request.question, topK), { temperature: 0, maxTokens: 256 });
    const plan = parseRecallPlan(raw, request.question, topK);
    const searchQuery = [...plan.ranking_hints, request.question].join(" ");
    const recallTables = memoryTablesForRecall(plan.target_tables);
    const candidates = await this.contextCandidates(request.userId, searchQuery, recallTables, Math.max(100, plan.top_k * 10));
    const ranked = hybridRankMemories(searchQuery, candidates.memories, {
      topK: plan.top_k,
      vectorScores: candidates.vectorScores,
      preferredTables: recallTables,
      minScore: 0.15
    });
    const rankedFacts = rankFacts(searchQuery, candidates.facts, plan.top_k);
    this.store.updateAccess(ranked);
    return {
      user_id: request.userId,
      question: request.question,
      recall_plan: plan,
      plan_fallback: plan.plan_fallback === true,
      facts: rankedFacts.map(factContextRow),
      memories: ranked.map(memoryContextRow)
    };
  }

  async remember(request: RememberRequest): Promise<Record<string, unknown>> {
    const existing = this.store.selectMemories(request.userId, ["semantic", "episodic"], 50);
    const raw = await this.runtime.generateJson(buildStoragePrompt(request.llmResponse, existing, request.source), { temperature: 0, maxTokens: 1024 });
    let decision = parseStorageDecision(raw, request.llmResponse, "store_episodic");
    let repairedRaw: string | undefined;
    if (decision.parse_error) {
      repairedRaw = await this.runtime.generateJson(buildStorageRepairPrompt(request.llmResponse, decision.raw_json), { temperature: 0, maxTokens: 1024 });
      decision = parseStorageDecision(repairedRaw, request.llmResponse, "store_episodic");
    }
    if (decision.parse_error) {
      return {
        user_id: request.userId,
        action: "ignore",
        route: "parse_error_noop",
        written: [],
        memory: null,
        reasoning: decision.reasoning,
        raw_model_json: decision.raw_json,
        repair_attempted: Boolean(repairedRaw),
        parse_error: decision.parse_error
      };
    }
    if (decision.memory && request.source) {
      decision = {
        ...decision,
        memory: {
          ...decision.memory,
          source_kind: request.source.source_kind ?? decision.memory.source_kind,
          source_id: request.source.source_id ?? decision.memory.source_id,
          source_timestamp: request.source.source_timestamp ?? decision.memory.source_timestamp,
          source_label: request.source.source_label ?? decision.memory.source_label
        }
      };
    }
    if (decision.memory) {
      decision = {
        ...decision,
        memory: normalizeMemoryTemporalFields(decision.memory, decision.memory.source_timestamp ?? request.source?.source_timestamp)
      };
    }
    if (decision.facts?.length) {
      const sourceTimestamp = decision.memory?.source_timestamp ?? request.source?.source_timestamp;
      decision = {
        ...decision,
        facts: decision.facts.map((fact) => normalizeFactTemporalFields(fact, sourceTimestamp))
      };
    }
    const result = this.store.applyDecision(request.userId, request.source?.source_id ?? "llm-response", decision, request.extraTags);
    await this.embedWrittenMemories(request.userId, result.memory_refs);
    return {
      user_id: request.userId,
      action: result.action,
      route: result.route,
      written: result.written,
      memory: decision.memory,
      reasoning: decision.reasoning,
      raw_model_json: decision.raw_json,
      repair_attempted: Boolean(repairedRaw),
      parse_error: decision.parse_error
    };
  }

  private async embedWrittenMemories(userId: string, refs: WrittenMemoryRef[]): Promise<void> {
    if (!this.embeddings) return;
    for (const ref of refs) {
      const embedding = await this.embeddings.runtime.embed(ref.content);
      this.store.upsertMemoryEmbedding(ref, userId, this.embeddings.model, embedding);
    }
  }

  private async renderContext(prompt: string, contextItems: ContextItem[], topK: number): Promise<{ items: ContextItem[]; context: string; reasoning: string; parse_error?: string }> {
    if (contextItems.length === 0) {
      return { items: [], context: "", reasoning: "No relevant memory context." };
    }

    const candidateById = new Map(contextItems.map((item) => [item.id, item]));
    const raw = await this.runtime.generateJson(buildContextRenderPrompt(prompt, contextItems, topK), { temperature: 0, maxTokens: 512 });
    const render = parseContextRender(raw, topK);
    const renderedItems: ContextItem[] = render.context_items
      .flatMap((item) => {
        const source = item.id ? candidateById.get(item.id) : undefined;
        if (!source) return [];
        if (!isGroundedContextContent(item.content, source)) return [];
        return [{
          ...source,
          table: item.table,
          content: item.content,
          reason: item.reason ?? source.reason
        }];
      });

    const items = renderedItems.length > 0 ? renderedItems : fallbackAgentContextItems(contextItems);
    return {
      items,
      context: renderAgentMemoryContext(items),
      reasoning: renderedItems.length > 0 ? render.reasoning : `Used deterministic grounded context fallback. ${render.reasoning}`,
      parse_error: render.parse_error
    };
  }

  private async contextCandidates(userId: string, query: string, preferredTables: MemoryRecord["table"][], limit: number): Promise<{ memories: MemoryRecord[]; facts: ScoredFact[]; vectorScores: Map<string, number> }> {
    const tables = allRecallTables();
    const lexicalCandidates = this.store.selectMemories(userId, tables, limit);
    const factCandidates = rankFacts(query, this.store.selectMemoryFacts(userId, limit), limit);
    if (!this.embeddings) {
      return { memories: lexicalCandidates, facts: factCandidates, vectorScores: new Map() };
    }

    const queryEmbedding = await this.embeddings.runtime.embed(query);
    const vectorScores = new Map<string, number>();
    const scored = this.store.selectEmbeddingRows(userId, this.embeddings.model)
      .map((row) => {
        const table = String(row.memory_table) as MemoryRecord["table"];
        if (!tables.includes(table)) return null;
        const embedding = parseEmbedding(row.embedding_json);
        if (!embedding) return null;
        return {
          table,
          id: String(row.memory_id),
          score: cosineSimilarity(queryEmbedding, embedding)
        };
      })
      .filter((row): row is { table: MemoryRecord["table"]; id: string; score: number } => row !== null)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    const memories = scored
      .map((row) => {
        vectorScores.set(`${row.table}:${row.id}`, row.score);
        return this.store.getMemory(row.table, row.id);
      })
      .filter((memory): memory is MemoryRecord => memory !== undefined);
    return { memories: mergeMemories(lexicalCandidates, memories), facts: factCandidates, vectorScores };
  }
}

interface ScoredFact extends MemoryFactRecord {
  score: number;
}

function isGroundedContextContent(content: string, source: ContextItem): boolean {
  const renderedTokens = tokenize(content);
  if (renderedTokens.length === 0) return false;
  const sourceText = [
    source.content,
    source.source_id ?? "",
    source.source_timestamp ?? "",
    source.resolved_time ?? "",
    source.temporal_expression ?? ""
  ].join(" ");
  const sourceTokens = new Set(tokenize(sourceText));
  const overlap = renderedTokens.filter((token) => sourceTokens.has(token)).length;
  return overlap >= Math.min(2, renderedTokens.length);
}

function exactContextItems(ranked: RankedMemory[], topK: number, reason: string, facts: ScoredFact[] = []): ContextItem[] {
  const factItems = facts.slice(0, topK).map(factContextItem);
  const memoryItems = ranked.slice(0, Math.max(0, topK - factItems.length)).map((memory) => ({
    id: `${memory.table}:${memory.id}`,
    memory_id: memory.id,
    table: memory.table,
    content: `${metadataPrefix(memory)} ${memory.content}`.trim(),
    reason,
    source_kind: memory.source_kind ?? undefined,
    source_id: memory.source_id ?? undefined,
    source_timestamp: memory.source_timestamp ?? undefined,
    source_label: memory.source_label ?? undefined,
    saved_at: memory.created_at,
    temporal_expression: memory.temporal_expression ?? undefined,
    resolved_time: memory.resolved_time ?? undefined,
    resolved_time_confidence: memory.resolved_time_confidence,
    score: memory.score
  }));
  return [...factItems, ...memoryItems];
}

function metadataPrefix(memory: RankedMemory): string {
  const fields = [
    memory.table,
    memory.created_at ? `saved_at=${memory.created_at}` : "",
    memory.source_timestamp ? `source_time=${memory.source_timestamp}` : "",
    memory.resolved_time ? `resolved_time=${memory.resolved_time}` : "",
    memory.source_label ? `source=${memory.source_label}` : memory.source_id ? `source=${memory.source_id}` : ""
  ].filter(Boolean);
  return `[${fields.join(" | ")}]`;
}

function memoryContextRow(memory: RankedMemory): Record<string, unknown> {
  return {
    table: memory.table,
    id: memory.id,
    content: memory.content,
    score: memory.score,
    created_at: memory.created_at,
    source_kind: memory.source_kind,
    source_id: memory.source_id,
    source_timestamp: memory.source_timestamp,
    source_label: memory.source_label,
    temporal_expression: memory.temporal_expression,
    resolved_time: memory.resolved_time,
    resolved_time_confidence: memory.resolved_time_confidence,
    metadata: memory.metadata
  };
}

function rankFacts(query: string, facts: MemoryFactRecord[], topK: number): ScoredFact[] {
  const qTokens = tokenize(query);
  if (qTokens.length === 0) return [];
  return facts
    .map((fact) => ({ ...fact, score: factScore(qTokens, fact) }))
    .filter((fact) => fact.score >= 0.2)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function factScore(queryTokens: string[], fact: MemoryFactRecord): number {
  const searchable = [
    fact.subject,
    fact.predicate,
    fact.object ?? "",
    fact.value_text,
    fact.fact_type ?? "",
    fact.evidence_text ?? "",
    fact.temporal_expression ?? "",
    fact.resolved_time ?? "",
    fact.source_id ?? ""
  ].join(" ");
  const memoryTokens = new Set(tokenize(searchable));
  const overlap = queryTokens.filter((token) => memoryTokens.has(token)).length;
  const coverage = overlap / queryTokens.length;
  const predicateHit = queryTokens.some((token) => tokenize(fact.predicate).includes(token)) ? 0.3 : 0;
  const subjectHit = queryTokens.some((token) => tokenize(fact.subject).includes(token)) ? 0.25 : 0;
  return Number((coverage + predicateHit + subjectHit + 0.1 * (fact.confidence ?? 0.75)).toFixed(6));
}

function factContextItem(fact: ScoredFact): ContextItem {
  return {
    id: `memory_fact:${fact.id}`,
    memory_id: fact.source_memory_id ?? undefined,
    table: "memory_fact",
    content: factContextContent(fact),
    reason: "Exact DB-backed extracted fact.",
    source_id: fact.source_id ?? undefined,
    source_timestamp: fact.source_timestamp ?? undefined,
    temporal_expression: fact.temporal_expression ?? undefined,
    resolved_time: fact.resolved_time ?? undefined,
    resolved_time_confidence: fact.resolved_time_confidence,
    score: fact.score
  };
}

function factContextContent(fact: MemoryFactRecord): string {
  const fields = [
    "fact",
    `predicate=${fact.predicate}`,
    fact.fact_type ? `type=${fact.fact_type}` : "",
    fact.inference_kind ? `inference=${fact.inference_kind}` : "",
    fact.source_timestamp ? `source_time=${fact.source_timestamp}` : "",
    fact.resolved_time ? `resolved_time=${fact.resolved_time}` : ""
  ].filter(Boolean);
  const evidence = fact.evidence_text ? ` Evidence: ${fact.evidence_text}.` : "";
  const source = fact.source_memory_table && fact.source_memory_id ? ` Source memory: ${fact.source_memory_table}:${fact.source_memory_id}.` : "";
  return `[${fields.join(" | ")}] Fact: ${fact.subject} ${fact.predicate} ${fact.value_text}.${evidence}${source}`;
}

function factContextRow(fact: ScoredFact): Record<string, unknown> {
  return {
    table: "memory_fact",
    id: fact.id,
    subject: fact.subject,
    predicate: fact.predicate,
    value_text: fact.value_text,
    fact_type: fact.fact_type,
    confidence: fact.confidence,
    inference_kind: fact.inference_kind,
    evidence_text: fact.evidence_text,
    source_memory_table: fact.source_memory_table,
    source_memory_id: fact.source_memory_id,
    source_id: fact.source_id,
    source_timestamp: fact.source_timestamp,
    temporal_expression: fact.temporal_expression,
    resolved_time: fact.resolved_time,
    resolved_time_confidence: fact.resolved_time_confidence,
    score: fact.score
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseEmbedding(value: unknown): number[] | null {
  try {
    const parsed = JSON.parse(String(value)) as unknown;
    return Array.isArray(parsed) && parsed.every((item) => typeof item === "number") ? parsed : null;
  } catch {
    return null;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  if (length === 0) return 0;
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let i = 0; i < length; i++) {
    dot += a[i] * b[i];
    aNorm += a[i] * a[i];
    bNorm += b[i] * b[i];
  }
  if (aNorm === 0 || bNorm === 0) return 0;
  return dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
}

function memoryTablesForRecall(tables: string[]): MemoryRecord["table"][] {
  const filtered = tables.filter((table): table is MemoryRecord["table"] => table === "episodic" || table === "semantic" || table === "archival");
  return filtered.length > 0 ? filtered : ["semantic", "episodic"];
}

function allRecallTables(): MemoryRecord["table"][] {
  return ["semantic", "episodic", "archival"];
}

function mergeMemories(...groups: MemoryRecord[][]): MemoryRecord[] {
  const seen = new Set<string>();
  const result: MemoryRecord[] = [];
  for (const group of groups) {
    for (const memory of group) {
      const key = `${memory.table}:${memory.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      result.push(memory);
    }
  }
  return result;
}
