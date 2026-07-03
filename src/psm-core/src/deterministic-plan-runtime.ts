import type { GenerateOptions, ModelRuntime } from "./types.js";

/** ponytail: recall/context plans without a separate planner LLM — DB rank + grounded rows only */
export class DeterministicPlanRuntime implements ModelRuntime {
  constructor(private readonly storage?: ModelRuntime) {}

  async warmup(): Promise<void> {
    const warmable = this.storage as { warmup?: () => Promise<void> } | undefined;
    if (warmable?.warmup) await warmable.warmup();
  }

  async generateJson(prompt: string, options: GenerateOptions = {}): Promise<string> {
    const storagePayload = extractJsonPayload(prompt);
    if (storagePayload && isStorageOperation(storagePayload)) {
      if (!this.storage) {
        throw new Error("PSM storage runtime required for remember() prompts.");
      }
      return this.storage.generateJson(prompt, options);
    }

    const operation = stringField(storagePayload, "operation");
    const topK = positiveInt(
      storagePayload?.requested_top_k ?? storagePayload?.top_k ?? storagePayload?.max_items,
      options.maxTokens && options.maxTokens <= 32 ? options.maxTokens : 5
    );

    if (operation === "context_plan" || operation === "recall_plan") {
      return JSON.stringify({
        intent: "recall",
        target_tables: ["episodic", "semantic", "archival"],
        filters: {},
        ranking_hints: [],
        temporal_intent: undefined,
        top_k: topK
      });
    }

    if (operation === "render_context") {
      // service.ts falls back to exact DB-backed context_items when render is empty
      return JSON.stringify({
        context_items: [],
        selected_ids: [],
        reasoning: "deterministic render fallback"
      });
    }

    throw new Error(`DeterministicPlanRuntime: unsupported operation ${operation ?? "unknown"}`);
  }
}

function extractJsonPayload(prompt: string): Record<string, unknown> | null {
  const marker = "<|user|>";
  const chunk = prompt.includes(marker) ? prompt.slice(prompt.lastIndexOf(marker)) : prompt;
  for (const line of chunk.split("\n").reverse()) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) continue;
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      continue;
    }
  }
  return null;
}

function isStorageOperation(payload: Record<string, unknown>): boolean {
  const operation = stringField(payload, "operation");
  if (operation === "remember_llm_response" || operation === "repair_remember_json") return true;
  return "conversation" in payload;
}

function stringField(payload: Record<string, unknown> | null, key: string): string | undefined {
  const value = payload?.[key];
  return typeof value === "string" && value.trim() ? value : undefined;
}

function positiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
