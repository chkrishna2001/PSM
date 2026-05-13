import type { ModelRuntime } from "./types.js";

export class HeuristicRuntime implements ModelRuntime {
  async generateJson(prompt: string): Promise<string> {
    const payloadText = extractPayloadText(prompt);
    const text = payloadText.toLowerCase();
    if (prompt.includes("recall_plan") || prompt.includes("context_plan")) {
      return JSON.stringify({
        intent: "recall",
        target_tables: ["semantic", "episodic"],
        filters: {},
        ranking_hints: Array.from(new Set(text.match(/[a-z0-9]{4,}/g) ?? [])).slice(0, 12),
        top_k: 5
      });
    }
    if (text.includes("never mind") || text.includes("ignore")) {
      return JSON.stringify({ action: "ignore", memory: null, reasoning: "Low-value response." });
    }
    if (text.includes("conflict") || text.includes("contradict")) {
      return JSON.stringify({
        action: "flag_conflict",
        memory: { content: payloadText, type: "episodic", confidence: 0.65, tags: ["heuristic"] },
        reasoning: "Potential conflict language detected."
      });
    }
    return JSON.stringify({
      action: "store_episodic",
      memory: { content: payloadText, type: "episodic", strength: 0.75, decay_rate: 0.02, emotional_weight: 0.2, confidence: 0.8, tags: ["heuristic"] },
      reasoning: "Fallback runtime stored meaningful assistant output as episodic memory."
    });
  }
}

function extractPayloadText(prompt: string): string {
  const jsonStart = prompt.indexOf("{");
  const jsonEnd = prompt.lastIndexOf("}");
  if (jsonStart >= 0 && jsonEnd > jsonStart) {
    try {
      const payload = JSON.parse(prompt.slice(jsonStart, jsonEnd + 1)) as Record<string, unknown>;
      return findText(payload) || "memory";
    } catch {
      return "memory";
    }
  }
  return "memory";
}

function findText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map(findText).filter(Boolean).join(" ");
  }
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    for (const key of ["question", "user_prompt", "conversation", "content"]) {
      const found = findText(record[key]);
      if (found) return found;
    }
    return Object.values(record).map(findText).filter(Boolean).join(" ");
  }
  return "";
}
