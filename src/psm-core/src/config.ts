import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, platform, userInfo } from "node:os";
import { dirname, join, resolve } from "node:path";
import { defaultEmbeddingModel } from "./embeddings.js";

export interface PsmDaemonConfig {
  enabled: boolean;
  host: string;
  autostart: boolean;
  idleTimeoutMs: number;
  startupTimeoutMs: number;
}

export interface PsmRuntimeConfig {
  contextSize: number;
  gpu: string;
  gpuLayers: string;
}

export interface PsmEmbeddingConfig {
  enabled: boolean;
  model: string;
}

export interface PsmTraceConfig {
  enabled: boolean;
  path: string;
}

export interface PsmModelConfig {
  enabled: boolean;
  checkpoint: string;
  python: string;
  device: string;
  outputFormat: "tagged" | "json" | "at_tag";
}

export interface PsmConfig {
  memoryDir: string;
  userId: string;
  recallTopK: number;
  embeddings: PsmEmbeddingConfig;
  runtime: PsmRuntimeConfig;
  psmModel: PsmModelConfig;
  daemon: PsmDaemonConfig;
  trace: PsmTraceConfig;
}

export function defaultPsmMemoryDir(): string {
  if (platform() === "win32") {
    return join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "psm-memory");
  }
  if (platform() === "darwin") {
    return join(homedir(), "Library", "Application Support", "psm-memory");
  }
  return join(process.env.XDG_DATA_HOME || join(homedir(), ".local", "share"), "psm-memory");
}

export function defaultPsmConfigPath(): string {
  return join(defaultPsmMemoryDir(), "config.json");
}

export function defaultPsmUserId(): string {
  try {
    return userInfo().username || "local-user";
  } catch {
    return process.env.USERNAME || process.env.USER || "local-user";
  }
}

export function defaultPsmConfig(): PsmConfig {
  return {
    memoryDir: defaultPsmMemoryDir(),
    userId: defaultPsmUserId(),
    recallTopK: 5,
    embeddings: {
      enabled: true,
      model: defaultEmbeddingModel
    },
    runtime: {
      contextSize: 4096,
      gpu: "cpu",
      gpuLayers: "0"
    },
    psmModel: {
      enabled: false,
      checkpoint: "psm-model/checkpoints/real-v3-50m-full-v2.pt",
      python: process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python",
      device: "cpu",
      outputFormat: "tagged"
    },
    daemon: {
      enabled: false,
      host: "127.0.0.1",
      autostart: false,
      idleTimeoutMs: 900_000,
      startupTimeoutMs: 60_000
    },
    trace: {
      enabled: false,
      path: join(defaultPsmMemoryDir(), "psm-model-io.jsonl")
    }
  };
}

export function readPsmConfig(): PsmConfig {
  const defaults = defaultPsmConfig();
  try {
    const path = defaultPsmConfigPath();
    if (!existsSync(path)) return defaults;
    const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (!isRecord(parsed)) return defaults;
    const embeddings = isRecord(parsed.embeddings) ? parsed.embeddings : {};
    const runtime = isRecord(parsed.runtime) ? parsed.runtime : {};
    const psmModel = isRecord(parsed.psmModel) ? parsed.psmModel : {};
    const daemon = isRecord(parsed.daemon) ? parsed.daemon : {};
    const trace = isRecord(parsed.trace) ? parsed.trace : {};
    return {
      memoryDir: stringValue(parsed.memoryDir, defaults.memoryDir),
      userId: stringValue(parsed.userId, defaults.userId),
      recallTopK: positiveIntValue(parsed.recallTopK, defaults.recallTopK),
      embeddings: {
        enabled: booleanValue(embeddings.enabled, defaults.embeddings.enabled),
        model: stringValue(embeddings.model, defaults.embeddings.model)
      },
      runtime: {
        contextSize: positiveIntValue(runtime.contextSize, defaults.runtime.contextSize),
        gpu: stringValue(runtime.gpu, defaults.runtime.gpu),
        gpuLayers: stringValue(runtime.gpuLayers, defaults.runtime.gpuLayers)
      },
      psmModel: {
        enabled: booleanValue(psmModel.enabled, defaults.psmModel.enabled),
        checkpoint: stringValue(psmModel.checkpoint, defaults.psmModel.checkpoint),
        python: stringValue(psmModel.python, defaults.psmModel.python),
        device: stringValue(psmModel.device, defaults.psmModel.device),
        outputFormat: stringValue(psmModel.outputFormat, defaults.psmModel.outputFormat) as PsmModelConfig["outputFormat"]
      },
      daemon: {
        enabled: booleanValue(daemon.enabled, defaults.daemon.enabled),
        host: stringValue(daemon.host, defaults.daemon.host),
        autostart: booleanValue(daemon.autostart, defaults.daemon.autostart),
        idleTimeoutMs: positiveIntValue(daemon.idleTimeoutMs, defaults.daemon.idleTimeoutMs),
        startupTimeoutMs: positiveIntValue(daemon.startupTimeoutMs, defaults.daemon.startupTimeoutMs)
      },
      trace: {
        enabled: booleanValue(trace.enabled, defaults.trace.enabled),
        path: stringValue(trace.path, join(stringValue(parsed.memoryDir, defaults.memoryDir), "psm-model-io.jsonl"))
      }
    };
  } catch {
    return defaults;
  }
}

export function writePsmConfig(config: Partial<PsmConfig>): PsmConfig {
  const merged = mergePsmConfig(config);
  const path = defaultPsmConfigPath();
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(merged, null, 2)}\n`, "utf8");
  return merged;
}

export function mergePsmConfig(config: Partial<PsmConfig>): PsmConfig {
  const current = readPsmConfig();
  return {
    ...current,
    ...config,
    memoryDir: config.memoryDir ? resolve(config.memoryDir) : current.memoryDir,
    embeddings: {
      ...current.embeddings,
      ...(config.embeddings ?? {})
    },
    runtime: {
      ...current.runtime,
      ...(config.runtime ?? {})
    },
    psmModel: {
      ...current.psmModel,
      ...(config.psmModel ?? {})
    },
    daemon: {
      ...current.daemon,
      ...(config.daemon ?? {})
    },
    trace: {
      ...current.trace,
      ...(config.trace ?? {})
    }
  };
}

export function resolvePsmMemoryDir(override?: string): string {
  if (override?.trim()) return resolve(override);
  if (process.env.PSM_MEMORY_DIR?.trim()) return resolve(process.env.PSM_MEMORY_DIR);
  return readPsmConfig().memoryDir;
}

export function resolvePsmDbPath(options: { dbPath?: string; memoryDir?: string } = {}): string {
  if (options.dbPath?.trim()) return options.dbPath;
  if (process.env.PSM_MEMORY_DB?.trim()) return process.env.PSM_MEMORY_DB;
  return join(resolvePsmMemoryDir(options.memoryDir), "psm-memory.db");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (value === "true") return true;
  if (value === "false") return false;
  return fallback;
}

function positiveIntValue(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
