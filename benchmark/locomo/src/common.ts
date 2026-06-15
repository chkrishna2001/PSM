import { readFileSync } from "node:fs";
import type { ModelRuntime } from "@psm-memory/sdk";
import type { CliOptions, LocomoSample, LocomoTurn } from "./types.js";

export function parseOptions(argv: string[]): CliOptions {
  const options: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      options[key] = next;
      i++;
    } else {
      options[key] = true;
    }
  }

  return {
    data: stringOption(options, "data", "benchmark/locomo/data/locomo10.json"),
    db: stringOption(options, "db", "benchmark/locomo/results/locomo-psm-memory.db"),
    server: stringOption(options, "server", "http://127.0.0.1:8080"),
    out: stringOption(options, "out", "benchmark/locomo/results/locomo-results.json"),
    limit: intOption(options, "limit", 0),
    batchSize: intOption(options, "batch-size", 10),
    topK: intOption(options, "top-k", 3)
  };
}

export function loadSamples(path: string): LocomoSample[] {
  return JSON.parse(readFileSync(path, "utf8")) as LocomoSample[];
}

export function flattenTurns(sample: LocomoSample): LocomoTurn[] {
  const conversation = sample.conversation ?? {};
  return Object.keys(conversation)
    .filter((key) => /^session_\d+$/.test(key))
    .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]))
    .flatMap((key) => {
      const turns = conversation[key];
      return Array.isArray(turns) ? turns.map((turn) => ({ ...turn, session: key } as LocomoTurn)) : [];
    });
}

export function locomoSourceTimestamp(sample: LocomoSample, session: string | undefined): string | undefined {
  if (!session) return undefined;
  const dateTime = sample.conversation?.[`${session}_date_time`];
  if (typeof dateTime === "string" && dateTime.trim()) return dateTime.trim();
  const sessionNumber = session.match(/^session_(\d+)$/)?.[1];
  if (!sessionNumber) return undefined;
  const date = sample.event_summary?.[`events_session_${sessionNumber}`]?.date;
  return typeof date === "string" && date.trim() ? date.trim() : undefined;
}

/** Product-shaped plain text: one turn utterance, no transcript wrapper or benchmark metadata. */
export function buildLocomoProductText(input: { turns: LocomoTurn[]; index: number }): string {
  const turn = input.turns[input.index];
  const speaker = String(turn.speaker ?? "Unknown").trim();
  const utterance = String(turn.text ?? "").trim();
  const imageBits = [
    turn.query ? `Image query: ${turn.query}.` : "",
    turn.blip_caption ? `Image caption: ${turn.blip_caption}.` : ""
  ].filter(Boolean);

  const base = utterance
    ? `${speaker} said ${quoteText(utterance)}.`
    : `${speaker} shared an image.`;
  return imageBits.length > 0 ? `${base} ${imageBits.join(" ")}`.replace(/\s+/g, " ").trim() : base;
}

/** Long-form transcript text (legacy benchmark shape; not product-shaped). */
export function buildLocomoPsmText(input: { sample: LocomoSample; turns: LocomoTurn[]; index: number; windowSize: number }): string {
  const turn = input.turns[input.index];
  const speaker = String(turn.speaker ?? "Unknown").trim();
  const utterance = String(turn.text ?? "").trim();
  const session = String(turn.session ?? "");
  const sourceTimestamp = locomoSourceTimestamp(input.sample, session);
  const windowSize = Number.isInteger(input.windowSize) && input.windowSize >= 0 ? input.windowSize : 2;
  const windowStart = Math.max(0, input.index - windowSize);
  const priorTurns = input.turns.slice(windowStart, input.index);
  const imageBits = [
    turn.query ? `image query: ${turn.query}` : "",
    turn.blip_caption ? `image caption: ${turn.blip_caption}` : ""
  ].filter(Boolean);

  const parts: string[] = [];
  if (priorTurns.length > 0) {
    const prior = priorTurns
      .map((item) => `${item.speaker ?? "Unknown"} said ${quoteText(item.text)}`)
      .join("; ");
    parts.push(`Earlier in the conversation: ${prior}.`);
  }
  if (sourceTimestamp) {
    parts.push(`Session time: ${sourceTimestamp}.`);
  }
  if (imageBits.length > 0) {
    parts.push(imageBits.join(" "));
  }
  parts.push(`${speaker} said ${quoteText(utterance)}.`);
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

export function buildLocomoRememberText(input: { sample: LocomoSample; turns: LocomoTurn[]; index: number; windowSize: number }): string {
  const turn = input.turns[input.index];
  const sampleId = String(input.sample.sample_id ?? "unknown");
  const session = String(turn.session ?? "");
  const diaId = String(turn.dia_id ?? "");
  const sourceTimestamp = locomoSourceTimestamp(input.sample, session);
  const windowStart = Math.max(0, input.index - input.windowSize);
  const windowEnd = input.index;
  const nearbyTurns = input.turns
    .slice(windowStart, windowEnd)
    .map((item) => renderTurnLine(item));
  const imageLines = renderImageLines(turn);
  return [
    `Source id: ${sampleId}:${diaId}`,
    `Sample id: ${sampleId}`,
    `Session: ${session || "unknown"}`,
    `Session time: ${sourceTimestamp ?? "unknown"}`,
    `Current speaker: ${turn.speaker ?? "unknown"}`,
    `Current utterance: ${quoteText(turn.text)}`,
    ...imageLines,
    "Previous context:",
    ...(nearbyTurns.length > 0 ? nearbyTurns : ["- none"]),
  ].join("\n");
}

function renderTurnLine(turn: LocomoTurn): string {
  const fields = [
    `${turn.speaker ?? "Unknown"} said: ${quoteText(turn.text)}`,
    turn.query ? `image query: ${turn.query}` : "",
    turn.blip_caption ? `image caption: ${turn.blip_caption}` : ""
  ].filter(Boolean);
  return `- [prior ${turn.session ?? "unknown"} ${turn.dia_id ?? "unknown"}] ${fields.join("; ")}`;
}

function renderImageLines(turn: LocomoTurn): string[] {
  const lines: string[] = [];
  if (turn.query) lines.push(`Image query: ${turn.query}`);
  if (turn.blip_caption) lines.push(`Image caption: ${turn.blip_caption}`);
  if (turn.img_url?.length) lines.push(`Image URLs: ${turn.img_url.join(", ")}`);
  return lines;
}

function quoteText(value: string | undefined): string {
  const text = value?.trim();
  return text ? `"${text}"` : "\"\"";
}

export function parseTags(value: string | null | undefined): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

export function tagValue(tags: string[], key: string): string {
  const prefix = `${key}:`;
  return tags.find((tag) => tag.startsWith(prefix))?.slice(prefix.length) ?? "";
}

export function createLocomoIngestRuntime(runtime: ModelRuntime): ModelRuntime {
  return {
    async generateJson(prompt, options) {
      const raw = await runtime.generateJson(prompt, options);
      return hasInvalidLocomoMemoryContent(raw, prompt) ? "invalid locomo memory content" : raw;
    }
  };
}

function hasInvalidLocomoMemoryContent(raw: string, prompt: string): boolean {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed)) return false;
    const memory = parsed.memory;
    const content = typeof memory === "string"
      ? memory.trim()
      : isRecord(memory) && typeof memory.content === "string"
        ? memory.content.trim()
        : "";
    if (!content) return false;
    if (isLocomoWrapperContent(content)) return true;
    const currentUtterance = extractCurrentUtterance(prompt);
    if (!currentUtterance) return false;
    return !isGroundedInCurrentUtterance(content, currentUtterance);
  } catch {
    return false;
  }
}

function extractCurrentUtterance(prompt: string): string {
  const match = prompt.match(/^Current utterance:\s*"([\s\S]*?)"$/m);
  if (match?.[1]?.trim()) return match[1].trim();

  const unescapedPrompt = prompt.replace(/\\n/g, "\n").replace(/\\"/g, "\"");
  const unescapedMatch = unescapedPrompt.match(/^Current utterance:\s*"([\s\S]*?)"$/m);
  if (unescapedMatch?.[1]?.trim()) return unescapedMatch[1].trim();

  const encodedContent = prompt.match(/"conversation":\[\{"role":"assistant","content":"((?:\\.|[^"\\])*)"\}\]/)?.[1];
  if (!encodedContent) return "";
  try {
    const decodedContent = JSON.parse(`"${encodedContent}"`) as unknown;
    if (typeof decodedContent !== "string") return "";
    return decodedContent.match(/^Current utterance:\s*"([\s\S]*?)"$/m)?.[1]?.trim() ?? "";
  } catch {
    return "";
  }
}

function isGroundedInCurrentUtterance(content: string, currentUtterance: string): boolean {
  const currentTokens = meaningfulTokens(currentUtterance);
  const contentTokens = meaningfulTokens(content);
  if (currentTokens.length === 0 || contentTokens.length === 0) return false;
  const contentSet = new Set(contentTokens);
  const overlap = currentTokens.filter((token) => contentSet.has(token));
  if (overlap.length >= Math.min(2, currentTokens.length)) return true;
  const quoted = content.match(/"([^"]{4,})"/g)?.some((quote) => currentUtterance.includes(quote.slice(1, -1))) ?? false;
  return quoted;
}

function meaningfulTokens(text: string): string[] {
  const stop = new Set(["the", "and", "but", "you", "your", "for", "with", "that", "this", "what", "have", "been", "said", "asked", "mentioned", "about", "from", "into", "they", "them", "their", "good", "really"]);
  return text.toLowerCase().match(/[a-z0-9+]{3,}/g)?.filter((token) => !stop.has(token)) ?? [];
}

function isLocomoWrapperContent(content: string): boolean {
  const lower = content.toLowerCase();
  return content.startsWith("{")
    || lower.includes("locomo benchmark conversation turn")
    || lower.includes("normal conversation-memory input")
    || lower.includes("rendered from the benchmark dataset")
    || lower.includes("store only durable memories")
    || lower.includes("current turn to remember:")
    || lower.includes("current utterance:")
    || lower.includes("previous context:")
    || lower.includes("extraction guidance:")
    || lower.includes("do not store")
    || lower.includes("preserve source ids")
    || lower.startsWith("user ")
    || lower.includes(" user ")
    || lower.includes("\"operation\":\"locomo_remember_turn\"")
    || lower.includes("\"operation\": \"locomo_remember_turn\"");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringOption(options: Record<string, string | boolean>, key: string, fallback: string): string {
  const value = options[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intOption(options: Record<string, string | boolean>, key: string, fallback: number): number {
  const value = options[key];
  const parsed = typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}
