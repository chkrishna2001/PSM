import type { MemoryAction, MemoryPayload, MemoryRecord, MemoryTable, RankedMemory, StorageDecision, WrittenMemoryRef } from "./types.js";
type DbRow = Record<string, unknown>;
export declare class MemoryStore {
    readonly dbPath: string;
    private readonly db;
    constructor(dbPath: string);
    initializeSchema(): void;
    applyDecision(userId: string, source: string, decision: StorageDecision, extraTags?: string[]): {
        action: MemoryAction;
        route: string;
        written: string[];
        memory_refs: WrittenMemoryRef[];
    };
    insertEpisodic(userId: string, content: string, memory?: MemoryPayload): string;
    insertSemantic(userId: string, content: string, memory?: MemoryPayload, sourceEpisodes?: string[]): string;
    insertConflict(userId: string, content: string, reason: string): string;
    insertDecaySchedule(userId: string, memoryKey: string, decayRate: number): string;
    insertDecision(userId: string, source: string, action: string, route: string, reasoning: string, rawJson: string): string;
    upsertMemoryEmbedding(ref: WrittenMemoryRef, userId: string, model: string, embedding: number[]): void;
    selectEmbeddingRows(userId: string, model: string): DbRow[];
    getMemory(table: "episodic" | "semantic" | "archival", id: string): MemoryRecord | undefined;
    selectTable(table: MemoryTable, limit: number): DbRow[];
    selectConflicts(status: string, limit: number): DbRow[];
    selectMemories(userId: string, tables?: MemoryTable[], limit?: number): MemoryRecord[];
    updateAccess(memories: RankedMemory[]): void;
    close(): void;
}
export {};
