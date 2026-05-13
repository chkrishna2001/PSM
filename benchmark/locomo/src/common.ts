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
    .flatMap((key) => (conversation[key] ?? []).map((turn) => ({ ...turn, session: key } as LocomoTurn)));
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
