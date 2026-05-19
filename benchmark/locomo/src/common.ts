import { readFileSync } from "node:fs";
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

export function buildLocomoRememberText(input: { sample: LocomoSample; turns: LocomoTurn[]; index: number; windowSize: number }): string {
  const turn = input.turns[input.index];
  const sampleId = String(input.sample.sample_id ?? "unknown");
  const session = String(turn.session ?? "");
  const diaId = String(turn.dia_id ?? "");
  const sourceTimestamp = locomoSourceTimestamp(input.sample, session);
  const windowStart = Math.max(0, input.index - input.windowSize);
  const windowEnd = Math.min(input.turns.length, input.index + input.windowSize + 1);
  const nearbyTurns = input.turns.slice(windowStart, windowEnd).map((item) => ({
    dia_id: item.dia_id ?? "",
    session: item.session ?? "",
    speaker: item.speaker ?? "",
    text: item.text ?? "",
    image_query: item.query ?? "",
    image_caption: item.blip_caption ?? ""
  }));
  const qaHints = (input.sample.qa ?? [])
    .filter((qa) => (qa.evidence ?? []).map(String).includes(diaId))
    .slice(0, 5)
    .map((qa) => ({
      question: qa.question ?? "",
      gold_answer: qa.answer ?? "",
      evidence: qa.evidence ?? []
    }));
  return JSON.stringify({
    operation: "locomo_remember_turn",
    instruction: "Store this LOCOMO turn through the normal PSM memory product path. Preserve source ids, session timestamp, durable facts, and answerable facts. Extract facts[] for people, activities, relationship status, career interests, locations, and temporal facts.",
    sample_id: sampleId,
    session,
    session_timestamp: sourceTimestamp,
    current_turn: {
      dia_id: diaId,
      speaker: turn.speaker ?? "",
      text: turn.text ?? "",
      image_query: turn.query ?? "",
      image_caption: turn.blip_caption ?? "",
      image_urls: turn.img_url ?? []
    },
    nearby_turns: nearbyTurns,
    qa_hints_for_this_turn: qaHints
  });
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

function stringOption(options: Record<string, string | boolean>, key: string, fallback: string): string {
  const value = options[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intOption(options: Record<string, string | boolean>, key: string, fallback: number): number {
  const value = options[key];
  const parsed = typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}
