import type { EmbeddingRuntime } from "./types.js";
export interface TransformersEmbeddingRuntimeOptions {
    model?: string;
    cacheDir?: string;
}
export declare const defaultEmbeddingModel = "Xenova/all-MiniLM-L6-v2";
export declare class TransformersEmbeddingRuntime implements EmbeddingRuntime {
    readonly model: string;
    private readonly cacheDir?;
    private extractorPromise?;
    constructor(options?: TransformersEmbeddingRuntimeOptions);
    embed(text: string): Promise<number[]>;
    private loadExtractor;
    private createExtractor;
}
