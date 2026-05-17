import { buildContextPlanPrompt, buildRecallPlanPrompt, buildStoragePrompt, buildStorageRepairPrompt } from "./prompts.js";
import { parseRecallPlan, parseStorageDecision } from "./json.js";
import { rankMemories } from "./ranking.js";
import { MemoryStore } from "./store.js";
import type { ContextItem, ContextRequest, EmbeddingRuntime, MemoryRecord, ModelRuntime, RankedMemory, RecallRequest, RememberRequest, WrittenMemoryRef } from "./types.js";

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
    const memories = await this.contextCandidates(request.userId, searchQuery, recallTables, Math.max(100, plan.top_k * 10));
    const ranked = rankMemories(searchQuery, memories, plan.top_k);
    this.store.updateAccess(ranked);
    const contextItems = exactContextItems(ranked, plan.top_k, "Exact DB-backed memory context.");
    return {
      user_id: request.userId,
      prompt: request.prompt,
      recall_plan: plan,
      context_items: contextItems,
      context_reasoning: "Rendered from exact DB-backed memory rows. PSM may plan retrieval, but injected memory text is never free-form generated.",
      context_parse_error: plan.parse_error,
      plan_fallback: plan.plan_fallback === true,
      memory_context: ranked.map(memoryContextRow),
      grounding: {
        mode: "exact_db_rows",
        generated_text_allowed: false,
        item_count: contextItems.length
      }
    };
  }

  async recall(request: RecallRequest): Promise<Record<string, unknown>> {
    const topK = request.topK ?? 5;
    const raw = await this.runtime.generateJson(buildRecallPlanPrompt(request.question, topK), { temperature: 0, maxTokens: 256 });
    const plan = parseRecallPlan(raw, request.question, topK);
    const searchQuery = [...plan.ranking_hints, request.question].join(" ");
    const memories = await this.contextCandidates(request.userId, searchQuery, memoryTablesForRecall(plan.target_tables), Math.max(100, plan.top_k * 10));
    const ranked = rankMemories(searchQuery, memories, plan.top_k);
    this.store.updateAccess(ranked);
    return {
      user_id: request.userId,
      question: request.question,
      recall_plan: plan,
      plan_fallback: plan.plan_fallback === true,
      memories: ranked.map(memoryContextRow)
    };
  }

  async remember(request: RememberRequest): Promise<Record<string, unknown>> {
    const existing = this.store.selectMemories(request.userId, ["semantic", "episodic"], 50);
    const raw = await this.runtime.generateJson(buildStoragePrompt(request.llmResponse, existing, request.source), { temperature: 0, maxTokens: 256 });
    let decision = parseStorageDecision(raw, request.llmResponse, "store_episodic");
    let repairedRaw: string | undefined;
    if (decision.parse_error) {
      repairedRaw = await this.runtime.generateJson(buildStorageRepairPrompt(request.llmResponse, decision.raw_json), { temperature: 0, maxTokens: 384 });
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
          ...request.source,
          ...decision.memory,
          source_kind: decision.memory.source_kind ?? request.source.source_kind,
          source_id: decision.memory.source_id ?? request.source.source_id,
          source_timestamp: decision.memory.source_timestamp ?? request.source.source_timestamp,
          source_label: decision.memory.source_label ?? request.source.source_label
        }
      };
    }
    const result = this.store.applyDecision(request.userId, request.source?.source_id ?? "llm-response", decision);
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

  private async contextCandidates(userId: string, query: string, tables: MemoryRecord["table"][], limit: number): Promise<MemoryRecord[]> {
    if (!this.embeddings) {
      return this.store.selectMemories(userId, tables, limit);
    }

    const queryEmbedding = await this.embeddings.runtime.embed(query);
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
      .map((row) => this.store.getMemory(row.table, row.id))
      .filter((memory): memory is MemoryRecord => memory !== undefined);
    return memories.length > 0 ? memories : this.store.selectMemories(userId, tables, limit);
  }
}

function exactContextItems(ranked: RankedMemory[], topK: number, reason: string): ContextItem[] {
  return ranked.slice(0, topK).map((memory) => ({
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
