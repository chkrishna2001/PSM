import { randomUUID } from "node:crypto";
import Database from "better-sqlite3";
import { routeForAction } from "./actions.js";
import type { MemoryAction, MemoryPayload, MemoryRecord, MemoryTable, RankedMemory, StorageDecision, WrittenMemoryRef } from "./types.js";
import { memoryTables } from "./types.js";

type DbRow = Record<string, unknown>;

export class MemoryStore {
  private readonly db: Database.Database;

  constructor(readonly dbPath: string) {
    this.db = new Database(dbPath);
    this.db.exec("PRAGMA foreign_keys = ON;");
  }

  initializeSchema(): void {
    this.db.exec(`
CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO schema_version(version) VALUES (1);
CREATE TABLE IF NOT EXISTS episodic (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  strength REAL NOT NULL,
  decay_rate REAL NOT NULL,
  emotional_weight REAL NOT NULL,
  confidence REAL NOT NULL,
  tags TEXT,
  source_kind TEXT,
  source_id TEXT,
  source_timestamp TEXT,
  source_label TEXT,
  temporal_expression TEXT,
  resolved_time TEXT,
  resolved_time_confidence REAL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_accessed TEXT,
  promoted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS semantic (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  strength REAL NOT NULL,
  decay_rate REAL NOT NULL,
  emotional_weight REAL NOT NULL,
  confidence REAL NOT NULL,
  tags TEXT,
  source_episodes TEXT,
  source_kind TEXT,
  source_id TEXT,
  source_timestamp TEXT,
  source_label TEXT,
  temporal_expression TEXT,
  resolved_time TEXT,
  resolved_time_confidence REAL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_accessed TEXT
);
CREATE TABLE IF NOT EXISTS archival (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  summary TEXT,
  original_type TEXT,
  source_id TEXT,
  archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS conflicts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  existing_memory_id TEXT,
  existing_memory_type TEXT,
  conflicting_content TEXT NOT NULL,
  conflict_reason TEXT,
  status TEXT NOT NULL DEFAULT 'unresolved',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decay_schedule (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  memory_key TEXT NOT NULL,
  next_decay TEXT NOT NULL,
  decay_rate REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  source TEXT NOT NULL,
  action TEXT NOT NULL,
  route TEXT NOT NULL,
  reasoning TEXT,
  raw_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS memory_embeddings (
  memory_table TEXT NOT NULL,
  memory_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  embedding_json TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (memory_table, memory_id, model)
);
CREATE INDEX IF NOT EXISTS idx_episodic_user_created ON episodic(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_user_created ON semantic(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conflicts_status_created ON conflicts(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_user_created ON decisions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_user_model ON memory_embeddings(user_id, model);
`);
    this.ensureMemoryMetadataColumns();
  }

  applyDecision(userId: string, source: string, decision: StorageDecision, extraTags: string[] = []): { action: MemoryAction; route: string; written: string[]; memory_refs: WrittenMemoryRef[] } {
    const route = routeForAction(decision.action);
    this.insertDecision(userId, source, decision.action, route, decision.reasoning, decision.raw_json);
    const memory = withExtraTags(decision.memory ?? { content: source }, extraTags);
    const content = memory.content?.trim() || source;
    const written: string[] = [];
    const memory_refs: WrittenMemoryRef[] = [];

    switch (route) {
      case "ignore":
      case "recall_only":
        return { action: decision.action, route, written, memory_refs };
      case "semantic_upsert":
      case "update_with_supersede":
        memory_refs.push({ table: "semantic", id: this.insertSemantic(userId, content, memory, [source]), content });
        written.push("semantic");
        return { action: decision.action, route, written, memory_refs };
      case "decay_existing_then_insert":
        this.insertDecaySchedule(userId, content, memory.decay_rate ?? 0.03);
        memory_refs.push({ table: "episodic", id: this.insertEpisodic(userId, content, memory), content });
        written.push("decay_schedule", "episodic");
        return { action: decision.action, route, written, memory_refs };
      case "conflict_log_and_hold":
        this.insertConflict(userId, content, decision.reasoning || "PSM flagged potential conflict");
        written.push("conflicts");
        if (decision.action === "flag_and_store") {
          memory_refs.push({ table: "episodic", id: this.insertEpisodic(userId, content, memory), content });
          written.push("episodic");
        }
        return { action: decision.action, route, written, memory_refs };
      default:
        memory_refs.push({ table: "episodic", id: this.insertEpisodic(userId, content, memory), content });
        written.push("episodic");
        return { action: decision.action, route, written, memory_refs };
    }
  }

  insertEpisodic(userId: string, content: string, memory: MemoryPayload = {}): string {
    const id = randomUUID();
    this.db.prepare(`
INSERT INTO episodic (
  id, user_id, content, strength, decay_rate, emotional_weight, confidence, tags,
  source_kind, source_id, source_timestamp, source_label, temporal_expression, resolved_time, resolved_time_confidence,
  promoted
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)`).run(
      id,
      userId,
      content,
      memory.strength ?? 0.75,
      memory.decay_rate ?? 0.02,
      memory.emotional_weight ?? 0.2,
      memory.confidence ?? 0.8,
      JSON.stringify(memory.tags ?? []),
      memory.source_kind ?? null,
      memory.source_id ?? null,
      memory.source_timestamp ?? null,
      memory.source_label ?? null,
      memory.temporal_expression ?? null,
      memory.resolved_time ?? null,
      memory.resolved_time_confidence ?? null
    );
    return id;
  }

  insertSemantic(userId: string, content: string, memory: MemoryPayload = {}, sourceEpisodes: string[] = []): string {
    const id = randomUUID();
    this.db.prepare(`
INSERT INTO semantic (
  id, user_id, content, strength, decay_rate, emotional_weight, confidence, tags, source_episodes,
  source_kind, source_id, source_timestamp, source_label, temporal_expression, resolved_time, resolved_time_confidence
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      id,
      userId,
      content,
      memory.strength ?? 0.85,
      memory.decay_rate ?? 0.005,
      memory.emotional_weight ?? 0.2,
      memory.confidence ?? 0.85,
      JSON.stringify(memory.tags ?? []),
      JSON.stringify(memory.source_episodes ?? sourceEpisodes),
      memory.source_kind ?? null,
      memory.source_id ?? null,
      memory.source_timestamp ?? null,
      memory.source_label ?? null,
      memory.temporal_expression ?? null,
      memory.resolved_time ?? null,
      memory.resolved_time_confidence ?? null
    );
    return id;
  }

  insertConflict(userId: string, content: string, reason: string): string {
    const id = randomUUID();
    this.db.prepare(`
INSERT INTO conflicts (id, user_id, conflicting_content, conflict_reason, status)
VALUES (?, ?, ?, ?, 'unresolved')`).run(id, userId, content, reason);
    return id;
  }

  insertDecaySchedule(userId: string, memoryKey: string, decayRate: number): string {
    const id = randomUUID();
    this.db.prepare(`
INSERT INTO decay_schedule (id, user_id, memory_key, next_decay, decay_rate)
VALUES (?, ?, ?, datetime('now', '+1 day'), ?)`).run(id, userId, memoryKey, decayRate);
    return id;
  }

  insertDecision(userId: string, source: string, action: string, route: string, reasoning: string, rawJson: string): string {
    const id = randomUUID();
    this.db.prepare(`
INSERT INTO decisions (id, user_id, source, action, route, reasoning, raw_json)
VALUES (?, ?, ?, ?, ?, ?, ?)`).run(id, userId, source, action, route, reasoning, rawJson);
    return id;
  }

  upsertMemoryEmbedding(ref: WrittenMemoryRef, userId: string, model: string, embedding: number[]): void {
    this.db.prepare(`
INSERT INTO memory_embeddings (memory_table, memory_id, user_id, model, dimensions, embedding_json, content_hash, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
ON CONFLICT(memory_table, memory_id, model) DO UPDATE SET
  dimensions = excluded.dimensions,
  embedding_json = excluded.embedding_json,
  content_hash = excluded.content_hash,
  updated_at = CURRENT_TIMESTAMP`).run(
      ref.table,
      ref.id,
      userId,
      model,
      embedding.length,
      JSON.stringify(embedding),
      hashContent(ref.content)
    );
  }

  selectEmbeddingRows(userId: string, model: string): DbRow[] {
    return this.db.prepare("SELECT * FROM memory_embeddings WHERE user_id = ? AND model = ?").all(userId, model);
  }

  getMemory(table: "episodic" | "semantic" | "archival", id: string): MemoryRecord | undefined {
    if (table === "episodic") {
      const row = this.db.prepare("SELECT *, 'episodic' as memory_table FROM episodic WHERE id = ?").get(id);
      return row ? asMemoryRecord(row) : undefined;
    }
    if (table === "semantic") {
      const row = this.db.prepare("SELECT *, 'semantic' as memory_table FROM semantic WHERE id = ?").get(id);
      return row ? asMemoryRecord(row) : undefined;
    }
    const row = this.db.prepare("SELECT id, user_id, content, NULL as strength, NULL as decay_rate, NULL as emotional_weight, NULL as confidence, NULL as tags, NULL as source_episodes, NULL as source_kind, NULL as source_id, NULL as source_timestamp, NULL as source_label, NULL as temporal_expression, NULL as resolved_time, NULL as resolved_time_confidence, 'archival' as memory_table, archived_at as created_at, NULL as last_accessed FROM archival WHERE id = ?").get(id);
    return row ? asMemoryRecord(row) : undefined;
  }

  selectTable(table: MemoryTable, limit: number): DbRow[] {
    if (!memoryTables.includes(table)) throw new Error(`Unsupported table: ${table}`);
    return this.db.prepare(`SELECT * FROM ${table} ORDER BY rowid DESC LIMIT ?`).all(limit);
  }

  insertRawRow(table: string, row: DbRow): void {
    if (!memoryTables.includes(table as MemoryTable)) throw new Error(`Unsupported table: ${table}`);
    const columns = tableColumns(this.db, table).filter((column) => column !== "rowid");
    const selected = columns.filter((column) => row[column] !== undefined);
    if (selected.length === 0) return;
    const placeholders = selected.map(() => "?").join(", ");
    const updates = selected
      .filter((column) => column !== "id")
      .map((column) => `${column} = excluded.${column}`)
      .join(", ");
    const conflict = selected.includes("id") && updates ? ` ON CONFLICT(id) DO UPDATE SET ${updates}` : selected.includes("id") ? " ON CONFLICT(id) DO NOTHING" : "";
    this.db.prepare(`INSERT INTO ${table} (${selected.join(", ")}) VALUES (${placeholders})${conflict}`).run(...selected.map((column) => rawValue(row[column])));
  }

  selectConflicts(status: string, limit: number): DbRow[] {
    return this.db.prepare("SELECT * FROM conflicts WHERE status = ? ORDER BY created_at DESC LIMIT ?").all(status, limit);
  }

  selectMemories(userId: string, tables: MemoryTable[] = ["semantic", "episodic"], limit = 100): MemoryRecord[] {
    const rows: MemoryRecord[] = [];
    for (const table of tables) {
      if (table === "episodic") {
        rows.push(...this.db.prepare("SELECT *, 'episodic' as memory_table FROM episodic WHERE user_id = ? ORDER BY created_at DESC LIMIT ?").all(userId, limit).map(asMemoryRecord));
      } else if (table === "semantic") {
        rows.push(...this.db.prepare("SELECT *, 'semantic' as memory_table FROM semantic WHERE user_id = ? ORDER BY created_at DESC LIMIT ?").all(userId, limit).map(asMemoryRecord));
      } else if (table === "archival") {
        rows.push(...this.db.prepare("SELECT id, user_id, content, NULL as strength, NULL as decay_rate, NULL as emotional_weight, NULL as confidence, NULL as tags, NULL as source_episodes, NULL as source_kind, NULL as source_id, NULL as source_timestamp, NULL as source_label, NULL as temporal_expression, NULL as resolved_time, NULL as resolved_time_confidence, 'archival' as memory_table, archived_at as created_at, NULL as last_accessed FROM archival WHERE user_id = ? ORDER BY archived_at DESC LIMIT ?").all(userId, limit).map(asMemoryRecord));
      }
    }
    return rows;
  }

  updateAccess(memories: RankedMemory[]): void {
    for (const memory of memories) {
      if (memory.table === "episodic" || memory.table === "semantic") {
        this.db.prepare(`UPDATE ${memory.table} SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?`).run(memory.id);
      }
    }
  }

  close(): void {
    this.db.close();
  }

  private ensureMemoryMetadataColumns(): void {
    const columns: Array<[string, string]> = [
      ["source_kind", "TEXT"],
      ["source_id", "TEXT"],
      ["source_timestamp", "TEXT"],
      ["source_label", "TEXT"],
      ["temporal_expression", "TEXT"],
      ["resolved_time", "TEXT"],
      ["resolved_time_confidence", "REAL"]
    ];
    for (const table of ["episodic", "semantic"]) {
      const existing = new Set(this.db.prepare(`PRAGMA table_info(${table})`).all().map((row) => String((row as DbRow).name)));
      for (const [column, type] of columns) {
        if (!existing.has(column)) {
          this.db.prepare(`ALTER TABLE ${table} ADD COLUMN ${column} ${type}`).run();
        }
      }
    }
  }
}

function hashContent(value: string): string {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}

function asMemoryRecord(row: DbRow): MemoryRecord {
  return {
    id: String(row.id),
    user_id: String(row.user_id),
    content: String(row.content),
    strength: asNumber(row.strength),
    decay_rate: asNumber(row.decay_rate),
    emotional_weight: asNumber(row.emotional_weight),
    confidence: asNumber(row.confidence),
    tags: row.tags == null ? null : String(row.tags),
    source_episodes: row.source_episodes == null ? null : String(row.source_episodes),
    source_kind: row.source_kind == null ? null : String(row.source_kind),
    source_id: row.source_id == null ? null : String(row.source_id),
    source_timestamp: row.source_timestamp == null ? null : String(row.source_timestamp),
    source_label: row.source_label == null ? null : String(row.source_label),
    temporal_expression: row.temporal_expression == null ? null : String(row.temporal_expression),
    resolved_time: row.resolved_time == null ? null : String(row.resolved_time),
    resolved_time_confidence: asNumber(row.resolved_time_confidence),
    table: row.memory_table as MemoryRecord["table"],
    created_at: row.created_at == null ? undefined : String(row.created_at),
    last_accessed: row.last_accessed == null ? null : String(row.last_accessed)
  };
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function tableColumns(db: Database.Database, table: string): string[] {
  return db.prepare(`PRAGMA table_info(${table})`).all().map((row) => String((row as DbRow).name));
}

function rawValue(value: unknown): unknown {
  if (Array.isArray(value) || (typeof value === "object" && value !== null)) return JSON.stringify(value);
  return value;
}

function withExtraTags(memory: MemoryPayload, extraTags: string[]): MemoryPayload {
  if (extraTags.length === 0) return memory;
  return {
    ...memory,
    tags: [...(memory.tags ?? []), ...extraTags]
  };
}
