import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { GenerateOptions, ModelRuntime } from "./types.js";

export interface TraceModelRuntimeOptions {
  runtime: ModelRuntime;
  path: string;
  enabled?: boolean;
  source?: string;
}

export class TraceModelRuntime implements ModelRuntime {
  constructor(private readonly options: TraceModelRuntimeOptions) {}

  async generateJson(prompt: string, options: GenerateOptions = {}): Promise<string> {
    const started = Date.now();
    let output = "";
    let error: string | undefined;
    try {
      output = await this.options.runtime.generateJson(prompt, options);
      return output;
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
      throw caught;
    } finally {
      if (this.options.enabled !== false) {
        appendPsmTrace(this.options.path, {
          ts: new Date().toISOString(),
          source: this.options.source,
          operation: inferPsmOperation(prompt),
          options,
          prompt,
          output,
          error,
          duration_ms: Date.now() - started
        });
      }
    }
  }
}

export function appendPsmTrace(path: string, entry: Record<string, unknown>): void {
  try {
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf8");
  } catch {
    // Tracing must never break memory operations.
  }
}

export function inferPsmOperation(prompt: string): string {
  const match = prompt.match(/"operation"\s*:\s*"([^"]+)"/);
  return match?.[1] ?? "unknown";
}
