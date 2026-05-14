import { MemoryStore } from "./store.js";
import type { ContextRequest, EmbeddingRuntime, ModelRuntime, RecallRequest, RememberRequest } from "./types.js";
export declare class PsmService {
    private readonly store;
    private readonly runtime;
    private readonly embeddings?;
    constructor(store: MemoryStore, runtime: ModelRuntime, embeddings?: {
        model: string;
        runtime: EmbeddingRuntime;
    } | undefined);
    context(request: ContextRequest): Promise<Record<string, unknown>>;
    recall(request: RecallRequest): Promise<Record<string, unknown>>;
    remember(request: RememberRequest): Promise<Record<string, unknown>>;
    private embedWrittenMemories;
    private contextCandidates;
}
