import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { buildStoragePrompt, MemoryStore, parseStorageDecision } from "@psm-memory/sdk";
import { flattenTurns, loadSamples, parseOptions } from "./common.js";
import { LlamaServerRuntime } from "./llama-server-runtime.js";

interface IngestStats {
  db: string;
  data: string;
  server: string;
  seen: number;
  stored: number;
  ignored: number;
  failed: number;
  errors: Array<{ source: string; error: string }>;
}

export async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const samples = loadSamples(options.data);
  const runtime = new LlamaServerRuntime(options.server);
  const store = new MemoryStore(options.db);
  store.initializeSchema();

  const stats: IngestStats = {
    db: options.db,
    data: options.data,
    server: options.server,
    seen: 0,
    stored: 0,
    ignored: 0,
    failed: 0,
    errors: []
  };

  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? "unknown");
    const userId = `locomo-${sampleId}`;
    const turns = flattenTurns(sample);
    for (let i = 0; i < turns.length; i += options.batchSize) {
      const batch = turns.slice(i, i + options.batchSize);
      const pending: Array<Promise<void>> = [];
      for (const turn of batch) {
        if (options.limit > 0 && stats.seen + pending.length >= options.limit) break;
        const ordinal: number = stats.seen + pending.length;
        pending.push(ingestTurn(runtime, store, stats, userId, sampleId, turn, ordinal));
      }
      stats.seen += pending.length;
      await Promise.all(pending);
      process.stdout.write(`ingested ${stats.seen} | stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}\n`);
      if (options.limit > 0 && stats.seen >= options.limit) return finish(store, stats);
    }
  }

  return finish(store, stats);
}

async function ingestTurn(
  runtime: LlamaServerRuntime,
  store: MemoryStore,
  stats: IngestStats,
  userId: string,
  sampleId: string,
  turn: { dia_id?: string; speaker?: string; text?: string },
  ordinal: number
): Promise<void> {
  const diaId = String(turn.dia_id ?? "");
  const source = `${sampleId}:${diaId || ordinal}`;
  const text = `${turn.speaker ?? "speaker"}: ${turn.text ?? ""}`;
  try {
    const raw = await runtime.generateJson(buildStoragePrompt(text), { temperature: 0, maxTokens: 192 });
    const decision = parseStorageDecision(raw, text, "store_episodic");
    const result = store.applyDecision(userId, source, decision, [
      `locomo_sample_id:${sampleId}`,
      `locomo_dia_id:${diaId}`,
      `locomo_speaker:${turn.speaker ?? ""}`
    ]);
    if (result.route === "ignore" || result.route === "recall_only") stats.ignored++;
    else stats.stored++;
  } catch (error) {
    stats.failed++;
    const message = error instanceof Error ? error.message : String(error);
    stats.errors.push({ source, error: message });
    store.insertDecision(userId, source, "error", "error", message, JSON.stringify({ error: message }));
  }
}

function finish(store: MemoryStore, stats: IngestStats): number {
  mkdirSync(dirname("benchmark/locomo/results/ingest-summary.json"), { recursive: true });
  writeFileSync("benchmark/locomo/results/ingest-summary.json", JSON.stringify(stats, null, 2), "utf8");
  store.close();
  process.stdout.write(`${JSON.stringify(stats, null, 2)}\n`);
  return stats.failed > 0 ? 1 : 0;
}

if (process.argv[1]?.endsWith("ingest.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
