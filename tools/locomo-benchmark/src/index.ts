import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { MemoryStore, rankMemories, type MemoryRecord } from "psm-sdk";

interface LocomoTurn {
  dia_id?: string;
  speaker?: string;
  text?: string;
}

interface LocomoQa {
  category?: string;
  question?: string;
  answer?: string;
  evidence?: string[];
}

interface LocomoSample {
  sample_id?: string;
  conversation?: Record<string, LocomoTurn[]>;
  qa?: LocomoQa[];
}

interface Args {
  data: string;
  db: string;
  out: string;
  limit: number;
  topK: number;
  ingest: boolean;
}

export async function main(argv: string[]): Promise<number> {
  const args = parseArgs(argv);
  if (!existsSync(args.data)) {
    throw new Error(`LOCOMO data not found: ${args.data}. Place locomo10.json there or pass --data.`);
  }
  const samples = JSON.parse(readFileSync(args.data, "utf8")) as LocomoSample[];
  const store = new MemoryStore(args.db);
  store.initializeSchema();

  if (args.ingest) {
    ingestSamples(samples, store, args.limit);
  }

  const records = evaluate(samples, store, args.topK, args.limit);
  const summary = summarize(records, args.topK);
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify({ summary, records }, null, 2), "utf8");
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\nWrote ${args.out}\n`);
  store.close();
  return 0;
}

function ingestSamples(samples: LocomoSample[], store: MemoryStore, limit: number): void {
  let count = 0;
  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? "unknown");
    for (const turn of flattenTurns(sample)) {
      if (limit > 0 && count >= limit) return;
      store.insertEpisodic(`locomo-${sampleId}`, `${turn.speaker ?? "speaker"}: ${turn.text ?? ""}`, {
        tags: [
          `locomo_sample_id:${sampleId}`,
          `locomo_dia_id:${turn.dia_id ?? ""}`,
          `speaker:${turn.speaker ?? ""}`
        ],
        confidence: 0.8,
        strength: 0.75
      });
      count++;
    }
  }
}

function evaluate(samples: LocomoSample[], store: MemoryStore, topK: number, limit: number): Record<string, unknown>[] {
  const records: Record<string, unknown>[] = [];
  let seen = 0;
  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? "unknown");
    const userId = `locomo-${sampleId}`;
    const memories = store.selectMemories(userId, ["semantic", "episodic"], 1000);
    for (const qa of sample.qa ?? []) {
      if (limit > 0 && seen >= limit) return records;
      const question = String(qa.question ?? "");
      const evidence = qa.evidence ?? [];
      const ranked = rankMemories(question, memories, topK);
      records.push({
        sample_id: sampleId,
        category: String(qa.category ?? "unknown"),
        question,
        answer: String(qa.answer ?? ""),
        evidence,
        selected_ids: ranked.map((memory) => locomoDiaId(memory)).filter(Boolean),
        hit_at_1: hitAt(evidence, ranked, 1),
        hit_at_k: hitAt(evidence, ranked, topK)
      });
      seen++;
    }
  }
  return records;
}

function flattenTurns(sample: LocomoSample): LocomoTurn[] {
  const conversation = sample.conversation ?? {};
  return Object.keys(conversation)
    .filter((key) => /^session_\d+$/.test(key))
    .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]))
    .flatMap((key) => conversation[key] ?? []);
}

function hitAt(evidence: string[], memories: MemoryRecord[], k: number): boolean {
  const selected = memories.slice(0, k).map((memory) => locomoDiaId(memory));
  return evidence.some((id) => selected.includes(id));
}

function locomoDiaId(memory: MemoryRecord): string {
  const tags = parseTags(memory.tags);
  const tag = tags.find((item) => item.startsWith("locomo_dia_id:"));
  return tag ? tag.slice("locomo_dia_id:".length) : "";
}

function parseTags(value: string | null | undefined): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function summarize(records: Record<string, unknown>[], topK: number): Record<string, unknown> {
  const count = records.length || 1;
  return {
    questions: records.length,
    hit_at_1: records.filter((record) => record.hit_at_1 === true).length / count,
    [`hit_at_${topK}`]: records.filter((record) => record.hit_at_k === true).length / count
  };
}

function parseArgs(argv: string[]): Args {
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
    data: stringOption(options, "data", join("benchmark", "locomo", "data", "locomo10.json")),
    db: stringOption(options, "db", "locomo-psm-memory.db"),
    out: stringOption(options, "out", join("results", "locomo", "typescript-results.json")),
    limit: intOption(options, "limit", 0),
    topK: intOption(options, "top-k", 3),
    ingest: options.ingest === true
  };
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

if (process.argv[1]?.endsWith("index.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
