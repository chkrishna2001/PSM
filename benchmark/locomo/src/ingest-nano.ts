import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, type StorageDecision } from "@psm-memory/sdk";
import { flattenTurns, loadSamples, locomoSourceTimestamp, parseOptions } from "./common.js";
import { NanoClient, type NanoPrediction } from "./nano-client.js";
import type { LocomoSample, LocomoTurn } from "./types.js";

interface NanoIngestStats {
  db: string;
  data: string;
  checkpoint: string;
  seen: number;
  stored: number;
  ignored: number;
  failed: number;
  errors: Array<{ source: string; error: string }>;
}

const instruction = "Perform the PSM memory operation for the current input. Return JSON only using the target schema.";

export async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const checkpoint = getOption(argv, "nano-checkpoint", "hf-download/nano-psm-primary-10m-retention-dominant-codex-selector-v3-checkpoints/checkpoint-best.pt");
  const config = getOption(argv, "nano-config", "nano-psm/configs/primary-10m.json");
  const script = getOption(argv, "nano-script", "nano-psm/src/nano_psm/predict.py");
  const python = getOption(argv, "python", "python");
  const device = getOption(argv, "device", "auto");
  const windowSize = Number(getOption(argv, "window-size", "2"));
  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  const nano = new NanoClient({ python, script, config, checkpoint, device });
  store.initializeSchema();

  const stats: NanoIngestStats = {
    db: options.db,
    data: options.data,
    checkpoint,
    seen: 0,
    stored: 0,
    ignored: 0,
    failed: 0,
    errors: []
  };

  try {
    for (const sample of samples) {
      const sampleId = String(sample.sample_id ?? "unknown");
      const userId = `locomo-${sampleId}`;
      const turns = flattenTurns(sample);
      for (let index = 0; index < turns.length; index++) {
        if (options.limit > 0 && stats.seen >= options.limit) return finish(store, nano, stats);
        const turn = turns[index];
        const diaId = String(turn.dia_id ?? "");
        const source = `${sampleId}:${diaId || stats.seen}`;
        stats.seen++;
        try {
          const prediction = await nano.predict({
            id: `locomo:${source}`,
            instruction,
            input: nanoLocomoInput(sample, turns, index, Number.isInteger(windowSize) && windowSize >= 0 ? windowSize : 2),
          });
          const result = store.applyDecision(userId, source, toStorageDecision(prediction, sample, turn, source), [
            `locomo_sample_id:${sampleId}`,
            `locomo_dia_id:${diaId}`,
            `locomo_speaker:${turn.speaker ?? ""}`,
            `locomo_session:${turn.session ?? ""}`
          ]);
          recordApplyResult(stats, source, prediction, result);
        } catch (error) {
          stats.failed++;
          const message = error instanceof Error ? error.message : String(error);
          stats.errors.push({ source, error: message });
          store.insertDecision(userId, source, "error", "error", message, JSON.stringify({ error: message }));
        }
        if (stats.seen % options.batchSize === 0) {
          process.stdout.write(`nano-ingested ${stats.seen} | stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}\n`);
        }
      }
    }
    return finish(store, nano, stats);
  } finally {
    nano.close();
  }
}

function nanoLocomoInput(sample: LocomoSample, turns: LocomoTurn[], index: number, windowSize: number): Record<string, unknown> {
  const turn = turns[index];
  const sampleId = String(sample.sample_id ?? "unknown");
  const diaId = String(turn.dia_id ?? "");
  const session = String(turn.session ?? "");
  return {
    prior_context: turns.slice(Math.max(0, index - windowSize), index).map((item) => ({
      speaker: item.speaker ?? "",
      text: item.text ?? "",
      dia_id: item.dia_id ?? "",
      session: item.session ?? "",
      image_caption: item.blip_caption ?? "",
    })),
    memory_store: [],
    operation: "remember",
    source_kind: "locomo",
    source_id: `${sampleId}:${diaId}`,
    current_turn: {
      speaker: turn.speaker ?? "",
      text: turn.text ?? "",
      dia_id: diaId,
      session,
      timestamp: locomoSourceTimestamp(sample, turn.session),
      image_caption: turn.blip_caption ?? "",
      image_query: turn.query ?? "",
    }
  };
}

function toStorageDecision(prediction: NanoPrediction, sample: LocomoSample, turn: LocomoTurn, sourceId: string): StorageDecision {
  const memory = prediction.memory ? {
    ...prediction.memory,
    content: String(prediction.memory.content ?? turn.text ?? ""),
    source_kind: "locomo",
    source_id: sourceId,
    source_timestamp: locomoSourceTimestamp(sample, turn.session),
    source_label: `LOCOMO ${sourceId}`,
  } : null;
  const decision = {
    action: normalizeAction(prediction.action),
    memory,
    facts: Array.isArray(prediction.facts) ? prediction.facts : [],
    reasoning: prediction.reasoning ?? "Nano PSM structured prediction.",
    confidence: typeof prediction.confidence === "number" ? prediction.confidence : undefined,
    raw_json: JSON.stringify(prediction),
    parse_error: prediction.parse_error,
  } as StorageDecision;
  if (decision.action !== "ignore" && !decision.memory?.content?.trim()) {
    return {
      ...decision,
      action: "ignore",
      memory: null,
      reasoning: "Nano PSM predicted a store action but produced no durable content.",
    };
  }
  return decision;
}

function normalizeAction(action: string): StorageDecision["action"] {
  if (action === "recall_context") return "ignore";
  if (action === "store") return "store_episodic";
  if (action === "promote") return "promote_semantic";
  if (action === "update") return "update_existing";
  if (action === "flag_contradiction") return "flag_conflict";
  if (action === "flag_and_update") return "flag_and_store";
  if (["ignore", "store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"].includes(action)) {
    return action as StorageDecision["action"];
  }
  return "ignore";
}

function recordApplyResult(
  stats: NanoIngestStats,
  source: string,
  prediction: NanoPrediction,
  result: { route: string; written: string[] },
): void {
  if (prediction.parse_error) {
    stats.failed++;
    stats.errors.push({ source, error: prediction.parse_error });
  } else if (result.route === "ignore" || result.route === "recall_only") {
    stats.ignored++;
  } else if (result.written.length > 0) {
    stats.stored++;
  } else {
    stats.ignored++;
  }
}

function finish(store: MemoryStore, nano: NanoClient, stats: NanoIngestStats): number {
  mkdirSync(dirname("benchmark/locomo/results/ingest-nano-summary.json"), { recursive: true });
  writeFileSync("benchmark/locomo/results/ingest-nano-summary.json", JSON.stringify(stats, null, 2), "utf8");
  nano.close();
  store.close();
  process.stdout.write(`${JSON.stringify(stats, null, 2)}\n`);
  return stats.failed > 0 ? 1 : 0;
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

if (process.argv[1]?.endsWith("ingest-nano.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
