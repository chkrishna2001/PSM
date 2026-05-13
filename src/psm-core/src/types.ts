export const memoryTables = ["episodic", "semantic", "archival", "conflicts", "decisions", "decay_schedule"] as const;

export type MemoryTable = (typeof memoryTables)[number];

export type MemoryAction =
  | "ignore"
  | "store"
  | "store_episodic"
  | "promote"
  | "promote_semantic"
  | "update"
  | "update_existing"
  | "rank"
  | "decay"
  | "decay_and_update"
  | "flag_conflict"
  | "flag_and_store"
  | "flag_and_update"
  | "detect_interference";

export type MemoryRoute =
  | "ignore"
  | "recall_only"
  | "episodic_insert"
  | "semantic_upsert"
  | "update_with_supersede"
  | "decay_existing_then_insert"
  | "conflict_log_and_hold";

export interface MemoryPayload {
  content?: string;
  type?: string;
  strength?: number;
  decay_rate?: number;
  emotional_weight?: number;
  confidence?: number;
  tags?: string[];
  source_episodes?: string[];
}

export interface StorageDecision {
  action: MemoryAction;
  memory: MemoryPayload | null;
  reasoning: string;
  confidence?: number;
  emotional_weight?: number;
  contradiction_score?: number;
  raw_json: string;
  parse_error?: string;
}

export interface RecallPlan {
  intent: string;
  target_tables: MemoryTable[];
  filters: Record<string, unknown>;
  ranking_hints: string[];
  top_k: number;
  raw_json: string;
  parse_error?: string;
}

export interface MemoryRecord {
  id: string;
  user_id: string;
  content: string;
  strength?: number;
  decay_rate?: number;
  emotional_weight?: number;
  confidence?: number;
  tags?: string | null;
  source_episodes?: string | null;
  table: "episodic" | "semantic" | "archival";
  created_at?: string;
  last_accessed?: string | null;
}

export interface RankedMemory extends MemoryRecord {
  score: number;
  metadata: Record<string, unknown>;
}

export interface ModelRuntime {
  generateJson(prompt: string, options?: GenerateOptions): Promise<string>;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
}

export interface ContextRequest {
  prompt: string;
  userId: string;
  topK?: number;
}

export interface RecallRequest {
  question: string;
  userId: string;
  topK?: number;
}

export interface RememberRequest {
  llmResponse: string;
  userId: string;
}
