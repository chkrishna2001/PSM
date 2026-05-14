export const defaultEmbeddingModel = "Xenova/all-MiniLM-L6-v2";
export class TransformersEmbeddingRuntime {
    model;
    cacheDir;
    extractorPromise;
    constructor(options = {}) {
        this.model = options.model ?? defaultEmbeddingModel;
        this.cacheDir = options.cacheDir;
    }
    async embed(text) {
        const extractor = await this.loadExtractor();
        const output = await extractor(text, { pooling: "mean", normalize: true });
        return extractVector(output);
    }
    async loadExtractor() {
        this.extractorPromise ??= this.createExtractor();
        return this.extractorPromise;
    }
    async createExtractor() {
        const mod = (await import("@huggingface/transformers"));
        if (this.cacheDir && mod.env) {
            mod.env.cacheDir = this.cacheDir;
        }
        return mod.pipeline("feature-extraction", this.model);
    }
}
function extractVector(output) {
    if (isVector(output))
        return output;
    if (isRecord(output)) {
        const data = output.data;
        if (data instanceof Float32Array || data instanceof Float64Array || Array.isArray(data)) {
            return Array.from(data);
        }
        if (typeof output.tolist === "function") {
            return flattenVector(output.tolist());
        }
    }
    return flattenVector(output);
}
function flattenVector(value) {
    if (isVector(value))
        return value;
    if (Array.isArray(value)) {
        if (value.length === 1)
            return flattenVector(value[0]);
        if (value.every((item) => typeof item === "number"))
            return value;
    }
    throw new Error("Unable to extract embedding vector from model output.");
}
function isVector(value) {
    return Array.isArray(value) && value.every((item) => typeof item === "number" && Number.isFinite(item));
}
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
//# sourceMappingURL=embeddings.js.map