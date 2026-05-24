import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, NodeLlamaRuntime, PsmService } from "@psm-memory/sdk";
import { buildLocomoRememberText, createLocomoIngestRuntime, flattenTurns, loadSamples, locomoSourceTimestamp, parseOptions } from "./common.js";

interface IngestStats {
  db: string;
  data: string;
  model: string;
  seen: number;
  stored: number;
  ignored: number;
  failed: number;
  errors: Array<{ source: string; error: string }>;
}

export async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const model = getOption(argv, "model", "models/psm-q4_k_m.gguf");
  const gpu = getOption(argv, "gpu", "auto") as "auto" | "cuda" | "vulkan" | "metal";
  const gpuLayers = parseGpuLayers(getOption(argv, "gpu-layers", "auto"));
  const contextSize = Number(getOption(argv, "context-size", "4096"));
  const windowSize = Number(getOption(argv, "window-size", "2"));
  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  store.initializeSchema();
  const runtime = createLocomoIngestRuntime(new NodeLlamaRuntime({
    modelPath: model,
    gpu,
    gpuLayers,
    contextSize: Number.isInteger(contextSize) && contextSize > 0 ? contextSize : 4096,
    log: (message) => process.stderr.write(`${message}\n`)
  }));
  const service = new PsmService(store, runtime);

  const stats: IngestStats = {
    db: options.db,
    data: options.data,
    model,
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
    for (let index = 0; index < turns.length; index++) {
      const turn = turns[index];
      if (options.limit > 0 && stats.seen >= options.limit) return finish(store, stats);
      const diaId = String(turn.dia_id ?? "");
      const source = `${sampleId}:${diaId || stats.seen}`;
      stats.seen++;
      try {
        const result = await service.remember({
          userId,
          llmResponse: buildLocomoRememberText({ sample, turns, index, windowSize: Number.isInteger(windowSize) && windowSize >= 0 ? windowSize : 2 }),
          userMessage: `${turn.speaker ?? "Unknown"} said: ${turn.text ?? ""}`.trim(),
          source: {
            source_kind: "locomo_turn",
            source_id: source,
            source_timestamp: locomoSourceTimestamp(sample, turn.session),
            source_label: `LOCOMO ${sampleId} ${diaId || stats.seen}`
          },
          includeExistingMemories: false,
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
      if (stats.seen % options.batchSize === 0) {
        process.stdout.write(`ingested ${stats.seen} | stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}\n`);
      }
    }
  }

  return finish(store, stats);
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
  mkdirSync(dirname("benchmark/locomo/results/ingest-node-summary.json"), { recursive: true });
  writeFileSync("benchmark/locomo/results/ingest-node-summary.json", JSON.stringify(stats, null, 2), "utf8");
  store.close();
  process.stdout.write(`${JSON.stringify(stats, null, 2)}\n`);
  return stats.failed > 0 ? 1 : 0;
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

function parseGpuLayers(value: string): "auto" | "max" | number {
  if (value === "auto" || value === "max") return value;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : "auto";
}

if (process.argv[1]?.endsWith("ingest-node.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
