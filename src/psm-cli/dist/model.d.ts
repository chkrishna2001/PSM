export declare const defaultModel: {
    readonly repo: "chkrishna2001/psm-memory-qwen-1.5b-gguf";
    readonly file: "gguf/psm-memory-qwen-1.5b-q4_k_m.gguf";
    readonly filename: "psm-memory-qwen-1.5b-q4_k_m.gguf";
    readonly size: 986047808;
    readonly sha256: "05a35ea07f27514e20db9f55e28d4e6f51a15c1684125067a0b65f9b483cc6e3";
};
export declare function modelCacheDir(): string;
export declare function defaultModelPath(): string;
export declare function hasDefaultModel(): boolean;
export declare function resolveModelPath(): string;
export declare function setupModel(options?: {
    force?: boolean;
    log?: (message: string) => void;
}): Promise<string>;
