export class NodeLlamaRuntime {
    completionPromise;
    modelPath;
    contextSize;
    gpu;
    gpuLayers;
    log;
    constructor(optionsOrModelPath, contextSize = 4096) {
        if (typeof optionsOrModelPath === "string") {
            this.modelPath = optionsOrModelPath;
            this.contextSize = contextSize;
            this.gpu = "auto";
            this.gpuLayers = "auto";
        }
        else {
            this.modelPath = optionsOrModelPath.modelPath;
            this.contextSize = optionsOrModelPath.contextSize ?? 4096;
            this.gpu = optionsOrModelPath.gpu ?? "auto";
            this.gpuLayers = optionsOrModelPath.gpuLayers ?? "auto";
            this.log = optionsOrModelPath.log;
        }
    }
    async generateJson(prompt, options = {}) {
        const completion = await this.loadCompletion();
        return completion.generateCompletion(prompt, {
            maxTokens: options.maxTokens ?? 256,
            temperature: options.temperature ?? 0,
            topK: options.topK ?? 20,
            topP: options.topP ?? 1,
            trimWhitespaceSuffix: true,
            customStopTriggers: ["<|im_end|>", "\n\n<|"]
        });
    }
    loadCompletion() {
        this.completionPromise ??= this.createCompletion();
        return this.completionPromise;
    }
    async createCompletion() {
        const mod = (await import("node-llama-cpp"));
        const getLlama = mod.getLlama;
        const Completion = mod.LlamaCompletion;
        if (!getLlama || !Completion) {
            throw new Error("node-llama-cpp does not expose getLlama and LlamaCompletion.");
        }
        if (mod.getLlamaGpuTypes) {
            const supported = await mod.getLlamaGpuTypes("supported");
            this.log?.(`node-llama-cpp supported GPU backends: ${supported.length ? supported.join(", ") : "none"}`);
        }
        const llama = await getLlama({
            gpu: this.gpu,
            build: "auto",
            logger: (level, message) => this.log?.(`[${String(level)}] ${message.trim()}`)
        });
        this.log?.(`node-llama-cpp selected backend: ${llama.gpu ?? "unknown"}`);
        const model = await llama.loadModel({
            modelPath: this.modelPath,
            gpuLayers: this.gpuLayers
        });
        this.log?.(`node-llama-cpp model GPU layers: ${model.gpuLayers ?? "unknown"}`);
        const context = await model.createContext({ contextSize: this.contextSize });
        return new Completion({ contextSequence: context.getSequence() });
    }
}
//# sourceMappingURL=llama-runtime.js.map