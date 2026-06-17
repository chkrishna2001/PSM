import { fallbackAgentContextItems, renderAgentMemoryContext } from "./context.js";
import { applyStorageGuards, groundingOverlapScore } from "./grounding-guards.js";
import { buildIndexablesForRemember, rankIndexables, type ScoredIndexable } from "./indexables.js";
import { buildContextPlanPrompt, buildContextRenderPrompt, buildRecallPlanPrompt, buildStoragePrompt, buildStorageRepairPrompt } from "./prompts.js";
import { parseContextRender, parseRecallPlan, parseStorageDecision } from "./json.js";
import { hybridRankMemories, tokenize } from "./ranking.js";
import { chunkSourceId, segmentLlmResponse } from "./segment-remember.js";
import { MemoryStore } from "./store.js";
import { normalizeFactTemporalFields, normalizeMemoryTemporalFields } from "./temporal.js";
import type { ContextItem, ContextRequest, EmbeddingRuntime, IndexableRecord, MemoryFactRecord, MemoryRecord, ModelRuntime, RankedMemory, RecallRequest, RememberRequest, StorageDecision, WrittenMemoryRef } from "./types.js";
import { routeForAction } from "./actions.js";
import type { SegmentSplitReason } from "./segment-remember.js";

export interface RememberChunkedRequest extends RememberRequest {
  maxChunkTokens?: number;
  minChunkTokens?: number;
}

export interface ChunkRememberResult {
  chunk_index: number;
  chunk_text: string;
  estimated_tokens: number;
  split_reason: SegmentSplitReason;
  source_id: string;
  action?: string;
  route?: string;
  written: string[];
  guard_rejected: boolean;
  deduped: boolean;
  content_grounded: boolean;
  grounding_overlap: number;
  grounding_required: number;
  remember_result: Record<string, unknown>;
}

const contextMinScore = 0.15;
const recallMinScore = 0.35;

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
    const renderCandidateK = Math.max(plan.top_k * 4, 20);
    const candidates = await this.contextCandidates(request.userId, searchQuery, recallTables, Math.max(100, renderCandidateK * 10));
    const ranked = hybridRankMemories(searchQuery, candidates.memories, {
      topK: renderCandidateK,
      vectorScores: candidates.vectorScores,
      preferredTables: recallTables,
      minScore: contextMinScore
    });
    const rankedFacts = rankFacts(searchQuery, candidates.facts, renderCandidateK);
    this.store.updateAccess(ranked);
    const contextItems = exactContextItems(ranked, renderCandidateK, "Exact DB-backed memory context.", rankedFacts);
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
      minScore: recallMinScore
    });
    const rankedFacts = rankFacts(searchQuery, candidates.facts, plan.top_k);
    const rankedIndexables = rankIndexables(request.question, this.store.selectIndexables(request.userId, 100), plan.top_k);
    this.store.updateAccess(ranked);
    return {
      user_id: request.userId,
      question: request.question,
      recall_plan: plan,
      plan_fallback: plan.plan_fallback === true,
      facts: rankedFacts.map(factContextRow),
      memories: ranked.map(memoryContextRow),
      indexables: rankedIndexables.map((row) => indexableContextRow(row, this.store)),
      workflows: rankedIndexables.filter((row) => row.kind === "workflow").map((row) => workflowContextRow(row, this.store))
    };
  }

  async remember(request: RememberRequest): Promise<Record<string, unknown>> {
    const existing = request.includeExistingMemories === false ? [] : this.store.selectMemories(request.userId, ["semantic", "episodic"], 50);
    const raw = await this.runtime.generateJson(buildStoragePrompt(request.llmResponse, existing, request.source, request.userMessage), { temperature: 0, maxTokens: 1024 });
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
    const guarded = applyStorageGuards(request.llmResponse, decision);
    if (guarded.rejected) {
      return {
        user_id: request.userId,
        action: "ignore",
        route: guarded.guard_route ?? "grounding_reject",
        written: [],
        memory: null,
        reasoning: guarded.guard_reason ?? "Storage guard rejected ungrounded output.",
        raw_model_json: decision.raw_json,
        repair_attempted: Boolean(repairedRaw),
        parse_error: decision.parse_error,
        guard_rejected: true
      };
    }
    if (wouldStoreDecision(decision) && !(decision.indexables?.length)) {
      decision = {
        ...decision,
        indexables: buildIndexablesForRemember({
          llmResponse: request.llmResponse,
          memoryContent: decision.memory?.content?.trim() ?? request.llmResponse,
          tags: [...(decision.memory?.tags ?? []), ...(request.extraTags ?? [])],
          facts: decision.facts ?? []
        })
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
      parse_error: decision.parse_error,
      indexables: decision.indexables ?? []
    };
  }

  async rememberChunked(request: RememberChunkedRequest): Promise<Record<string, unknown>> {
    const segments = segmentLlmResponse(request.llmResponse, {
      maxChunkTokens: request.maxChunkTokens,
      minChunkTokens: request.minChunkTokens
    });
    const baseSourceId = request.source?.source_id ?? "llm-response";
    const chunkResults: ChunkRememberResult[] = [];

    for (const segment of segments) {
      const sourceId = chunkSourceId(baseSourceId, segment.index);
      const rememberResult = await this.remember({
        ...request,
        llmResponse: segment.text,
        source: {
          ...request.source,
          source_id: sourceId,
          source_kind: request.source?.source_kind ?? "llm_response_chunk",
          source_label: request.source?.source_label
            ? `${request.source.source_label} [chunk ${segment.index + 1}/${segments.length}]`
            : `llm-response chunk ${segment.index + 1}/${segments.length}`
        },
        extraTags: [
          ...(request.extraTags ?? []),
          `chunk_index:${segment.index}`,
          `chunk_total:${segments.length}`,
          `chunk_split:${segment.splitReason}`
        ]
      });

      const written = Array.isArray(rememberResult.written) ? rememberResult.written.map(String) : [];
      const route = typeof rememberResult.route === "string" ? rememberResult.route : "";
      const memory = isRecord(rememberResult.memory) ? rememberResult.memory : null;
      const memoryContent = typeof memory?.content === "string" ? memory.content : "";
      const overlap = groundingOverlapScore(segment.text, memoryContent);
      chunkResults.push({
        chunk_index: segment.index,
        chunk_text: segment.text,
        estimated_tokens: segment.estimatedTokens,
        split_reason: segment.splitReason,
        source_id: sourceId,
        action: typeof rememberResult.action === "string" ? rememberResult.action : undefined,
        route,
        written,
        guard_rejected: rememberResult.guard_rejected === true,
        deduped: route === "dedupe_skip",
        content_grounded: written.length > 0 && overlap.grounded,
        grounding_overlap: overlap.overlap,
        grounding_required: overlap.required,
        remember_result: rememberResult
      });
    }

    const storedChunks = chunkResults.filter((chunk) => chunk.written.length > 0);
    return {
      user_id: request.userId,
      chunked: true,
      chunk_count: segments.length,
      chunks_stored: storedChunks.length,
      chunks: chunkResults,
      action: storedChunks.length > 0 ? "store_episodic" : "ignore",
      route: storedChunks.length > 0 ? "chunked_remember" : "ignore",
      written: storedChunks.flatMap((chunk) => chunk.written)
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

    const items = renderedItems.length > 0 ? renderedItems : fallbackAgentContextItems(contextItems).slice(0, topK);
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
  const temporalQuestion = queryTokens.some((token) => ["when", "date", "year", "month", "time"].includes(token));
  const temporalBoost = temporalQuestion && fact.fact_type === "temporal_fact" ? 0.4 : 0;
  return Number((coverage + predicateHit + subjectHit + 0.1 * (fact.confidence ?? 0.75) + temporalBoost).toFixed(6));
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

function wouldStoreDecision(decision: StorageDecision): boolean {
  const route = routeForAction(decision.action);
  if (route === "ignore" || route === "recall_only") return false;
  return Boolean(decision.memory?.content?.trim());
}

function indexableContextRow(row: ScoredIndexable, store: MemoryStore): Record<string, unknown> {
  const linked = linkedMemoryForIndexable(row, store);
  return {
    id: row.id,
    kind: row.kind,
    key: row.key,
    steps: row.steps,
    salience: row.salience,
    score: row.score,
    reconstructive_hint: row.reconstructive_hint,
    evidence_text: row.evidence_text,
    tags: row.tags,
    target_memory_table: row.target_memory_table,
    target_memory_id: row.target_memory_id,
    linked_memory_content: linked?.content ?? null
  };
}

function workflowContextRow(row: ScoredIndexable, store: MemoryStore): Record<string, unknown> {
  const linked = linkedMemoryForIndexable(row, store);
  return {
    key: row.key,
    steps: row.steps,
    procedure: linked?.content ?? row.reconstructive_hint ?? row.evidence_text ?? "",
    score: row.score,
    target_memory_id: row.target_memory_id
  };
}

function linkedMemoryForIndexable(row: IndexableRecord, store: MemoryStore): MemoryRecord | undefined {
  if (!row.target_memory_table || !row.target_memory_id) return undefined;
  const table = row.target_memory_table;
  if (table !== "episodic" && table !== "semantic" && table !== "archival") return undefined;
  return store.getMemory(table, row.target_memory_id);
}
