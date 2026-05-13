import type { GenerateOptions, ModelRuntime } from "./types.js";
export interface NodeLlamaRuntimeOptions {
    modelPath: string;
    contextSize?: number;
    gpu?: "auto" | "cuda" | "vulkan" | "metal" | false;
    gpuLayers?: "auto" | "max" | number;
    log?: (message: string) => void;
}
export declare class NodeLlamaRuntime implements ModelRuntime {
    private completionPromise?;
    private readonly modelPath;
    private readonly contextSize;
    private readonly gpu;
    private readonly gpuLayers;
    private readonly log?;
    constructor(optionsOrModelPath: NodeLlamaRuntimeOptions | string, contextSize?: number);
    generateJson(prompt: string, options?: GenerateOptions): Promise<string>;
    private loadCompletion;
    private createCompletion;
}
