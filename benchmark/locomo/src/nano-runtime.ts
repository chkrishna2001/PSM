import type { GenerateOptions, ModelRuntime } from "@psm-memory/sdk";
import { NanoClient } from "./nano-client.js";

const instruction = "Perform the PSM memory operation for the current input. Return JSON only using the target schema.";

export class NanoModelRuntime implements ModelRuntime {
  constructor(private readonly nano: NanoClient) {}

  async generateJson(prompt: string, _options: GenerateOptions = {}): Promise<string> {
    const payload = promptPayload(prompt);
    const operation = String(payload.operation ?? "");
    if (operation === "remember_llm_response") {
      const prediction = await this.nano.predict({
        id: `psm-service:${sourceId(payload)}`,
        instruction,
        input: storageInput(payload)
      });
      return JSON.stringify(prediction);
    }
    if (operation === "repair_remember_json") {
      return String(payload.invalid_model_output ?? "{}");
    }
    if (operation === "recall_plan" || operation === "context_plan") {
      const question = String(payload.question ?? payload.user_prompt ?? "");
      return JSON.stringify({
        intent: "recall",
        target_tables: ["semantic", "episodic"],
        filters: {},
        ranking_hints: keywords(question),
        temporal_intent: temporalIntent(question),
        top_k: Number(payload.requested_top_k ?? 5)
      });
    }
    if (Array.isArray(payload.candidate_context_items)) {
      return JSON.stringify({
        context_items: payload.candidate_context_items.slice(0, Number(payload.max_items ?? 5)).map((item: unknown) => item),
        reasoning: "Selected highest ranked grounded context items."
      });
    }
    return "{}";
  }
}

function promptPayload(prompt: string): Record<string, unknown> {
  const matches = [...prompt.matchAll(/\{[\s\S]*?\}(?=\s*<\|assistant\|>|\s*$)/g)];
  for (let i = matches.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(matches[i][0]);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Try the previous JSON-looking block.
    }
  }
  const start = prompt.indexOf("{");
  const end = prompt.lastIndexOf("}");
  if (start >= 0 && end > start) {
    try {
      const parsed = JSON.parse(prompt.slice(start, end + 1));
      if (isRecord(parsed)) return parsed;
    } catch {
      return {};
    }
  }
  return {};
}

function storageInput(payload: Record<string, unknown>): Record<string, unknown> {
  const conversation = Array.isArray(payload.conversation) ? payload.conversation : [];
  const current = lastMessage(conversation);
  const source = isRecord(payload.source) ? payload.source : {};
  return {
    prior_context: conversation.slice(0, -1).map((item) => ({
      speaker: String(item.role ?? ""),
      text: String(item.content ?? "")
    })),
    memory_store: Array.isArray(payload.memory_store) ? payload.memory_store : [],
    operation: "remember",
    source_kind: String(source.source_kind ?? ""),
    source_id: String(source.source_id ?? ""),
    current_turn: {
      speaker: String(source.source_label ?? current.role ?? ""),
      text: String(current.content ?? ""),
      dia_id: diaId(String(source.source_id ?? "")),
      session: "",
      timestamp: String(source.source_timestamp ?? ""),
      image_caption: "",
      image_query: ""
    }
  };
}

function lastMessage(conversation: unknown[]): Record<string, unknown> {
  const item = conversation[conversation.length - 1];
  return isRecord(item) ? item : {};
}

function sourceId(payload: Record<string, unknown>): string {
  const source = isRecord(payload.source) ? payload.source : {};
  return String(source.source_id ?? "prediction");
}

function diaId(value: string): string {
  return value.match(/(?:^|:)(D\d+:\d+)$/)?.[1] ?? "";
}

function keywords(value: string): string[] {
  return [...new Set(value.toLowerCase().match(/[a-z0-9]+/g)?.filter((token) => token.length > 2).slice(0, 12) ?? [])];
}

function temporalIntent(value: string): string | undefined {
  return /\bwhen\b|\bdate\b|\byear\b|\bmonth\b|\bday\b|\btime\b/i.test(value) ? "temporal" : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
