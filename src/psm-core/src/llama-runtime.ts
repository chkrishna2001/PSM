import type { GenerateOptions, ModelRuntime } from "./types.js";

type LlamaModule = {
  getLlama?: (options?: Record<string, unknown>) => Promise<{
    gpu?: string;
    loadModel(options: { modelPath: string; gpuLayers?: "auto" | "max" | number }): Promise<{
      gpuLayers?: number;
      createContext(options?: { contextSize?: number }): Promise<{
        getSequence(): unknown;
        dispose?(): void;
      }>;
      dispose?(): void;
    }>;
    dispose?(): void;
  }>;
  getLlamaGpuTypes?: (include: "supported" | "allValid") => Promise<string[]>;
  LlamaCompletion?: new (options: { contextSequence: unknown }) => {
    generateCompletion(prompt: string, options?: Record<string, unknown>): Promise<string>;
    dispose?(options?: { disposeSequence?: boolean }): void;
  };
};

export interface NodeLlamaRuntimeOptions {
  modelPath: string;
  contextSize?: number;
  gpu?: "auto" | "cuda" | "vulkan" | "metal" | false;
  gpuLayers?: "auto" | "max" | number;
  log?: (message: string) => void;
}

export class NodeLlamaRuntime implements ModelRuntime {
  private completionPromise?: Promise<{ generateCompletion(prompt: string, options?: Record<string, unknown>): Promise<string> }>;

  private readonly modelPath: string;
  private readonly contextSize: number;
  private readonly gpu: NodeLlamaRuntimeOptions["gpu"];
  private readonly gpuLayers: NodeLlamaRuntimeOptions["gpuLayers"];
  private readonly log?: (message: string) => void;

  constructor(optionsOrModelPath: NodeLlamaRuntimeOptions | string, contextSize = 4096) {
    if (typeof optionsOrModelPath === "string") {
      this.modelPath = optionsOrModelPath;
      this.contextSize = contextSize;
      this.gpu = "auto";
      this.gpuLayers = "auto";
    } else {
      this.modelPath = optionsOrModelPath.modelPath;
      this.contextSize = optionsOrModelPath.contextSize ?? 4096;
      this.gpu = optionsOrModelPath.gpu ?? "auto";
      this.gpuLayers = optionsOrModelPath.gpuLayers ?? "auto";
      this.log = optionsOrModelPath.log;
    }
  }

  async generateJson(prompt: string, options: GenerateOptions = {}): Promise<string> {
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

  private loadCompletion(): Promise<{ generateCompletion(prompt: string, options?: Record<string, unknown>): Promise<string> }> {
    this.completionPromise ??= this.createCompletion();
    return this.completionPromise;
  }

  private async createCompletion(): Promise<{ generateCompletion(prompt: string, options?: Record<string, unknown>): Promise<string> }> {
    const mod = (await import("node-llama-cpp")) as LlamaModule;
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
      logger: (level: unknown, message: string) => this.log?.(`[${String(level)}] ${message.trim()}`)
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
