import type { MemoryRecord, RankedMemory } from "./types.js";
export declare function rankMemories(query: string, memories: MemoryRecord[], topK: number): RankedMemory[];
export declare function tokenize(text: string): string[];
