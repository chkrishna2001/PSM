import type { EmbeddingRuntime } from "./types.js";

type TransformersModule = {
  pipeline(task: "feature-extraction", model: string, options?: Record<string, unknown>): Promise<(text: string, options?: Record<string, unknown>) => Promise<unknown>>;
  env?: {
    cacheDir?: string;
    allowLocalModels?: boolean;
  };
};

export interface TransformersEmbeddingRuntimeOptions {
  model?: string;
  cacheDir?: string;
}

export const defaultEmbeddingModel = "Xenova/all-MiniLM-L6-v2";

export class TransformersEmbeddingRuntime implements EmbeddingRuntime {
  readonly model: string;
  private readonly cacheDir?: string;
  private extractorPromise?: Promise<(text: string, options?: Record<string, unknown>) => Promise<unknown>>;

  constructor(options: TransformersEmbeddingRuntimeOptions = {}) {
    this.model = options.model ?? defaultEmbeddingModel;
    this.cacheDir = options.cacheDir;
  }

  async embed(text: string): Promise<number[]> {
    const extractor = await this.loadExtractor();
    const output = await extractor(text, { pooling: "mean", normalize: true });
    return extractVector(output);
  }

  private async loadExtractor(): Promise<(text: string, options?: Record<string, unknown>) => Promise<unknown>> {
    this.extractorPromise ??= this.createExtractor();
    return this.extractorPromise;
  }

  private async createExtractor(): Promise<(text: string, options?: Record<string, unknown>) => Promise<unknown>> {
    const mod = (await import("@huggingface/transformers")) as TransformersModule;
    if (this.cacheDir && mod.env) {
      mod.env.cacheDir = this.cacheDir;
    }
    return mod.pipeline("feature-extraction", this.model);
  }
}

function extractVector(output: unknown): number[] {
  if (isVector(output)) return output;
  if (isRecord(output)) {
    const data = output.data;
    if (data instanceof Float32Array || data instanceof Float64Array || Array.isArray(data)) {
      return Array.from(data as ArrayLike<number>);
    }
    if (typeof output.tolist === "function") {
      return flattenVector(output.tolist());
    }
  }
  return flattenVector(output);
}

function flattenVector(value: unknown): number[] {
  if (isVector(value)) return value;
  if (Array.isArray(value)) {
    if (value.length === 1) return flattenVector(value[0]);
    if (value.every((item) => typeof item === "number")) return value as number[];
  }
  throw new Error("Unable to extract embedding vector from model output.");
}

function isVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number" && Number.isFinite(item));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
