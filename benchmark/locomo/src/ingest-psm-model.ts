import { mkdirSync, writeFileSync } from "node:fs";
import { platform } from "node:os";
import path from "node:path";
import { MemoryStore, PsmModelRuntime, PsmService } from "@psm-memory/sdk";
import {
  buildLocomoProductText,
  buildLocomoPsmText,
  buildLocomoRememberText,
  createLocomoIngestRuntime,
  flattenTurns,
  loadSamples,
  locomoSourceTimestamp,
  parseOptions
} from "./common.js";

interface IngestStats {
  db: string;
  data: string;
  checkpoint: string;
  device: string;
  seen: number;
  stored: number;
  ignored: number;
  failed: number;
  errors: Array<{ source: string; error: string }>;
}

interface DebugRecord {
  source: string;
  llm_response: string;
  user_message?: string;
  action?: string;
  route?: string;
  raw_model_json?: string;
  parse_error?: string;
  written?: string[];
}

export async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const checkpoint = getOption(argv, "checkpoint", "psm-model/checkpoints/real-v3-50m-full-v2-step-048000.pt");
  const python = getOption(argv, "python", platform() === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python");
  const device = getOption(argv, "device", "cpu");
  const outputFormat = getOption(argv, "output-format", "tagged") as "tagged" | "json" | "at_tag";
  const windowSize = Number(getOption(argv, "window-size", "2"));
  const inputFormat = getOption(argv, "input-format", "psm");
  const repoRoot = path.resolve(getOption(argv, "repo-root", process.cwd()));
  const useLocomoWrapper = inputFormat === "locomo";
  const useNarrative = inputFormat === "psm-narrative";
  const debugRaw = argv.includes("--debug-raw");
  const debugOut = getOption(argv, "debug-out", options.db.replace(/\.db$/i, "-ingest-debug.json"));

  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  store.initializeSchema();
  const modelRuntime = new PsmModelRuntime({
    checkpoint: path.resolve(repoRoot, checkpoint),
    python: resolvePythonExecutable(repoRoot, python),
    repoRoot,
    outputFormat,
    device
  });
  const runtime = useLocomoWrapper ? createLocomoIngestRuntime(modelRuntime) : modelRuntime;
  const service = new PsmService(store, runtime);

  const stats: IngestStats = {
    db: options.db,
    data: options.data,
    checkpoint,
    device,
    seen: 0,
    stored: 0,
    ignored: 0,
    failed: 0,
    errors: []
  };
  const debugRecords: DebugRecord[] = [];

  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? "unknown");
    const userId = `locomo-${sampleId}`;
    const turns = flattenTurns(sample);
    for (let index = 0; index < turns.length; index++) {
      const turn = turns[index];
      if (options.limit > 0 && stats.seen >= options.limit) return finish(store, stats, debugRaw ? debugOut : undefined, debugRecords);
      const diaId = String(turn.dia_id ?? "");
      const source = `${sampleId}:${diaId || stats.seen}`;
      stats.seen++;
      try {
        const rememberInput = {
          sample,
          turns,
          index,
          windowSize: Number.isInteger(windowSize) && windowSize >= 0 ? windowSize : 2
        };
        const llmResponse = useLocomoWrapper
          ? buildLocomoRememberText(rememberInput)
          : useNarrative
            ? buildLocomoPsmText(rememberInput)
            : buildLocomoProductText(rememberInput);
        const userMessage = useLocomoWrapper
          ? `${turn.speaker ?? "Unknown"} said: ${turn.text ?? ""}`.trim()
          : undefined;
        const result = await service.remember({
          userId,
          llmResponse,
          ...(userMessage ? { userMessage } : {}),
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
        if (debugRaw) {
          debugRecords.push({
            source,
            llm_response: llmResponse,
            ...(userMessage ? { user_message: userMessage } : {}),
            action: typeof result.action === "string" ? result.action : undefined,
            route: typeof result.route === "string" ? result.route : undefined,
            raw_model_json: typeof result.raw_model_json === "string" ? result.raw_model_json : undefined,
            parse_error: typeof result.parse_error === "string" ? result.parse_error : undefined,
            written: Array.isArray(result.written) ? result.written.map(String) : undefined
          });
        }
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

  return finish(store, stats, debugRaw ? debugOut : undefined, debugRecords);
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

function finish(store: MemoryStore, stats: IngestStats, debugOut?: string, debugRecords: DebugRecord[] = []): number {
  const out = "benchmark/locomo/results/ingest-psm-model-summary.json";
  mkdirSync(path.dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(stats, null, 2), "utf8");
  if (debugOut) {
    mkdirSync(path.dirname(debugOut), { recursive: true });
    writeFileSync(debugOut, JSON.stringify(debugRecords, null, 2), "utf8");
  }
  store.close();
  process.stdout.write(`${JSON.stringify(stats, null, 2)}\n`);
  return stats.failed > 0 ? 1 : 0;
}

function resolvePythonExecutable(repoRoot: string, python: string): string {
  const looksAbsolute = /^[A-Za-z]:[\\/]/.test(python) || python.startsWith("/");
  if (looksAbsolute || (!python.includes("/") && !python.includes("\\"))) {
    return python;
  }
  return path.resolve(repoRoot, python);
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

if (process.argv[1]?.endsWith("ingest-psm-model.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
