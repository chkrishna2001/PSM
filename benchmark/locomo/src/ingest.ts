import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, PsmService } from "@psm-memory/sdk";
import { buildLocomoRememberText, createLocomoIngestRuntime, flattenTurns, loadSamples, locomoSourceTimestamp, parseOptions } from "./common.js";
import { LlamaServerRuntime } from "./llama-server-runtime.js";
import type { LocomoSample, LocomoTurn } from "./types.js";

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
  const windowSize = Number(getOption(argv, "window-size", "2"));
  const samples = loadSamples(options.data);
  const runtime = createLocomoIngestRuntime(new LlamaServerRuntime(options.server));
  const store = new MemoryStore(options.db);
  store.initializeSchema();
  const service = new PsmService(store, runtime);

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
        const index = i + pending.length;
        const ordinal: number = stats.seen + pending.length;
        pending.push(ingestTurn(service, store, stats, userId, sample, turns, index, windowSize, sampleId, turn, ordinal));
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
  service: PsmService,
  store: MemoryStore,
  stats: IngestStats,
  userId: string,
  sample: LocomoSample,
  turns: LocomoTurn[],
  index: number,
  windowSize: number,
  sampleId: string,
  turn: LocomoTurn,
  ordinal: number
): Promise<void> {
  const diaId = String(turn.dia_id ?? "");
  const source = `${sampleId}:${diaId || ordinal}`;
  try {
    const result = await service.remember({
      userId,
      llmResponse: buildLocomoRememberText({ sample, turns, index, windowSize: Number.isInteger(windowSize) && windowSize >= 0 ? windowSize : 2 }),
      source: {
        source_kind: "locomo_turn",
        source_id: source,
        source_timestamp: locomoSourceTimestamp(sample, turn.session),
        source_label: `LOCOMO ${sampleId} ${diaId || ordinal}`
      },
      extraTags: [
        `locomo_sample_id:${sampleId}`,
        `locomo_dia_id:${diaId}`,
        `locomo_speaker:${turn.speaker ?? ""}`,
        `locomo_session:${turn.session ?? ""}`
      ]
    });
    recordRememberResult(stats, source, result);
  } catch (error) {
    stats.failed++;
    const message = error instanceof Error ? error.message : String(error);
    stats.errors.push({ source, error: message });
    store.insertDecision(userId, source, "error", "error", message, JSON.stringify({ error: message }));
  }
}

function recordRememberResult(stats: IngestStats, source: string, result: Record<string, unknown>): void {
  const route = typeof result.route === "string" ? result.route : "";
  const parseError = typeof result.parse_error === "string" ? result.parse_error : "";
  const written = Array.isArray(result.written) ? result.written : [];
  if (parseError) {
    stats.failed++;
    stats.errors.push({ source, error: parseError });
  } else if (route === "ignore" || route === "recall_only") {
    stats.ignored++;
  } else if (written.length > 0) {
    stats.stored++;
  } else {
    stats.ignored++;
  }
}

function finish(store: MemoryStore, stats: IngestStats): number {
  mkdirSync(dirname("benchmark/locomo/results/ingest-summary.json"), { recursive: true });
  writeFileSync("benchmark/locomo/results/ingest-summary.json", JSON.stringify(stats, null, 2), "utf8");
  store.close();
  process.stdout.write(`${JSON.stringify(stats, null, 2)}\n`);
  return stats.failed > 0 ? 1 : 0;
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

if (process.argv[1]?.endsWith("ingest.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
