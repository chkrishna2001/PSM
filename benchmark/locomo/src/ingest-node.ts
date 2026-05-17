import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, NodeLlamaRuntime, parseStorageDecision } from "@psm-memory/sdk";
import { flattenTurns, loadSamples, parseOptions } from "./common.js";
import { buildLocomoMemoryPrompt } from "./storage-prompt.js";

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
  const runtime = new NodeLlamaRuntime({
    modelPath: model,
    gpu,
    gpuLayers,
    contextSize: Number.isInteger(contextSize) && contextSize > 0 ? contextSize : 4096,
    log: (message) => process.stderr.write(`${message}\n`)
  });

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
      const text = `${turn.speaker ?? "speaker"}: ${turn.text ?? ""}`;
      stats.seen++;
      try {
        const prompt = buildLocomoMemoryPrompt({ sample, turns, index, windowSize: Number.isInteger(windowSize) && windowSize >= 0 ? windowSize : 2 });
        const raw = await runtime.generateJson(prompt, { temperature: 0, maxTokens: 256 });
        const decision = parseStorageDecision(raw, text, "store_episodic");
        const result = store.applyDecision(userId, source, decision, [
          `locomo_sample_id:${sampleId}`,
          `locomo_dia_id:${diaId}`,
          `locomo_speaker:${turn.speaker ?? ""}`,
          `locomo_session:${turn.session ?? ""}`
        ]);
        if (result.route === "ignore" || result.route === "recall_only") stats.ignored++;
        else stats.stored++;
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
