import { MemoryStore } from "./store.js";
import type { ContextRequest, ModelRuntime, RecallRequest, RememberRequest } from "./types.js";
export declare class PsmService {
    private readonly store;
    private readonly runtime;
    constructor(store: MemoryStore, runtime: ModelRuntime);
    context(request: ContextRequest): Promise<Record<string, unknown>>;
    recall(request: RecallRequest): Promise<Record<string, unknown>>;
    remember(request: RememberRequest): Promise<Record<string, unknown>>;
}
