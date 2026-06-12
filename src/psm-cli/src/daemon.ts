import { createServer, request as httpRequest } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { defaultEmbeddingModel, HybridPsmRuntime, MemoryStore, NodeLlamaRuntime, PsmModelRuntime, PsmService, readPsmConfig, resolvePsmDbPath, TraceModelRuntime, TransformersEmbeddingRuntime, type EmbeddingRuntime, type ModelRuntime, type PsmConfig } from "@psm-memory/sdk";
import { defaultModelPath, resolveModelPath } from "./model.js";

export interface DaemonRequest {
  operation: "recall" | "remember" | "context";
  payload: Record<string, unknown>;
}

interface DaemonState {
  pid: number;
  host: string;
  port: number;
  startedAt: string;
  lastSeenAt: string;
}

export async function startDaemon(): Promise<void> {
  const config = readPsmConfig();
  const dbPath = resolvePsmDbPath();
  mkdirSync(dirname(dbPath), { recursive: true });

  const store = new MemoryStore(dbPath);
  store.initializeSchema();

  const runtime = createRuntime(config);
  await warmupRuntime(runtime);
  const service = new PsmService(store, runtime, config.embeddings.enabled ? createEmbeddingRuntime() : undefined);
  let lastRequestAt = Date.now();

  const server = createServer(async (req, res) => {
    lastRequestAt = Date.now();
    writeDaemonState(serverPort(server), config.daemon.host);
    try {
      if (req.method === "GET" && req.url === "/health") {
        writeJson(res, 200, {
          ok: true,
          db: dbPath,
          model: defaultModelPath(),
          psm_model: psmModelEnabled(config) ? resolve(config.psmModel.checkpoint) : undefined,
          pid: process.pid
        });
        return;
      }

      if (req.method !== "POST" || req.url !== "/v1") {
        writeJson(res, 404, { error: "not_found" });
        return;
      }

      const body = await readJsonBody(req);
      const operation = body.operation;
      const payload = isRecord(body.payload) ? body.payload : {};

      if (operation === "recall") {
        writeJson(res, 200, await service.recall({
          question: stringValue(payload.question, "question"),
          userId: stringValue(payload.userId, config.userId),
          topK: numberValue(payload.topK, config.recallTopK)
        }));
        return;
      }

      if (operation === "context") {
        writeJson(res, 200, await service.context({
          prompt: stringValue(payload.prompt, "prompt"),
          userId: stringValue(payload.userId, config.userId),
          topK: numberValue(payload.topK, config.recallTopK)
        }));
        return;
      }

      if (operation === "remember") {
        writeJson(res, 200, await service.remember({
          llmResponse: stringValue(payload.llmResponse, "llmResponse"),
          userMessage: optionalStringValue(payload.userMessage),
          userId: stringValue(payload.userId, config.userId),
          source: isRecord(payload.source) ? payload.source : undefined
        }));
        return;
      }

      writeJson(res, 400, { error: "unsupported_operation" });
    } catch (error) {
      writeJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, config.daemon.host, () => {
      writeDaemonState(serverPort(server), config.daemon.host);
      resolve();
    });
  });

  const idleCheck = setInterval(() => {
    if (Date.now() - lastRequestAt >= config.daemon.idleTimeoutMs) {
      shutdown();
    }
  }, Math.min(60_000, Math.max(1_000, Math.floor(config.daemon.idleTimeoutMs / 4))));

  const shutdown = () => {
    clearInterval(idleCheck);
    removeDaemonStateForCurrentProcess();
    server.close(() => {
      store.close();
      process.exit(0);
    });
  };

  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}

export async function callDaemon(request: DaemonRequest): Promise<Record<string, unknown> | null> {
  const config = readPsmConfig();
  if (!config.daemon.enabled || !config.daemon.autostart) return null;
  const state = await ensureDaemon();
  return daemonRequestRaw(state, "POST", "/v1", request);
}

export async function dispatchDaemonRemember(
  request: DaemonRequest,
  onComplete: (result: { ok: true; body: Record<string, unknown> } | { ok: false; error: string }) => void
): Promise<boolean> {
  const config = readPsmConfig();
  if (!config.daemon.enabled || !config.daemon.autostart) return false;
  try {
    const state = await ensureDaemon();
    void daemonRequestRaw(state, "POST", "/v1", request)
      .then((body) => onComplete({ ok: true, body }))
      .catch((error) => onComplete({
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      }));
    return true;
  } catch (error) {
    onComplete({
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
    return true;
  }
}

async function ensureDaemon(): Promise<DaemonState> {
  const current = readDaemonState();
  if (current && await isHealthy(current)) return current;

  removeDaemonState();
  startDetachedDaemon();

  const timeoutAt = Date.now() + readPsmConfig().daemon.startupTimeoutMs;
  while (Date.now() < timeoutAt) {
    const state = readDaemonState();
    if (state && await isHealthy(state)) return state;
    await sleep(250);
  }

  throw new Error("PSM daemon is enabled but could not start before startupTimeoutMs.");
}

function startDetachedDaemon(): void {
  const cliPath = process.argv[1];
  const child = spawn(process.execPath, [cliPath, "daemon-run"], {
    detached: true,
    stdio: "ignore",
    windowsHide: true
  });
  child.unref();
}

async function isHealthy(state: DaemonState): Promise<boolean> {
  try {
    await daemonRequestRaw(state, "GET", "/health");
    return true;
  } catch {
    return false;
  }
}

function createEmbeddingRuntime(): { model: string; runtime: EmbeddingRuntime } {
  const config = readPsmConfig();
  const model = process.env.PSM_MEMORY_EMBEDDING_MODEL ?? config.embeddings.model ?? defaultEmbeddingModel;
  return {
    model,
    runtime: new TransformersEmbeddingRuntime({
      model,
      cacheDir: join(dirname(defaultModelPath()), "hf")
    })
  };
}

function resolveRepoRoot(): string {
  const cwd = process.cwd();
  if (existsSync(resolve(cwd, "psm-model", "src", "psm_model"))) return cwd;
  return cwd;
}

function psmModelEnabled(config: PsmConfig): boolean {
  if (process.env.PSM_MODEL === "1" || process.env.PSM_MODEL === "true") return true;
  return config.psmModel.enabled;
}

function createRuntime(config: PsmConfig): ModelRuntime {
  const primary = new NodeLlamaRuntime({
    modelPath: resolveModelPath(),
    contextSize: config.runtime.contextSize,
    gpu: config.runtime.gpu as "auto",
    gpuLayers: config.runtime.gpuLayers as "auto"
  });
  const repoRoot = resolveRepoRoot();
  const runtime: ModelRuntime = psmModelEnabled(config)
    ? new HybridPsmRuntime(
        primary,
        new PsmModelRuntime({
          checkpoint: resolve(repoRoot, config.psmModel.checkpoint),
          python: config.psmModel.python,
          device: config.psmModel.device,
          outputFormat: config.psmModel.outputFormat,
          repoRoot
        })
      )
    : primary;
  return traceEnabled(config) ? new TraceModelRuntime({
    runtime,
    path: tracePath(config),
    source: "psm-daemon"
  }) : runtime;
}

async function warmupRuntime(runtime: ModelRuntime): Promise<void> {
  const warmable = runtime as { warmup?: () => Promise<void> };
  if (warmable.warmup) await warmable.warmup();
}

function traceEnabled(config: PsmConfig): boolean {
  if (process.env.PSM_MEMORY_TRACE === "1" || process.env.PSM_MEMORY_TRACE === "true") return true;
  return config.trace.enabled;
}

function tracePath(config: PsmConfig): string {
  return process.env.PSM_MEMORY_TRACE_PATH ?? config.trace.path;
}

function daemonRequestRaw(state: DaemonState, method: "GET" | "POST", path: string, body?: unknown): Promise<Record<string, unknown>> {
  const rawBody = body === undefined ? undefined : JSON.stringify(body);
  return new Promise((resolve, reject) => {
    const req = httpRequest({
      hostname: state.host,
      port: state.port,
      path,
      method,
      headers: rawBody ? {
        "content-type": "application/json",
        "content-length": String(Buffer.byteLength(rawBody))
      } : undefined
    }, (res) => {
      let data = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        data += chunk;
      });
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data || "{}") as unknown;
          if ((res.statusCode ?? 500) >= 400) {
            reject(new Error(isRecord(parsed) && typeof parsed.error === "string" ? parsed.error : `HTTP ${res.statusCode}`));
            return;
          }
          resolve(isRecord(parsed) ? parsed : {});
        } catch (error) {
          reject(error);
        }
      });
    });
    req.once("error", reject);
    if (rawBody) req.write(rawBody);
    req.end();
  });
}

function daemonStatePath(): string {
  return join(readPsmConfig().memoryDir, "daemon.json");
}

function readDaemonState(): DaemonState | null {
  try {
    const path = daemonStatePath();
    if (!existsSync(path)) return null;
    const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (!isRecord(parsed)) return null;
    const pid = Number(parsed.pid);
    const port = Number(parsed.port);
    const host = typeof parsed.host === "string" ? parsed.host : "";
    if (!Number.isInteger(pid) || !Number.isInteger(port) || !host) return null;
    return {
      pid,
      port,
      host,
      startedAt: typeof parsed.startedAt === "string" ? parsed.startedAt : "",
      lastSeenAt: typeof parsed.lastSeenAt === "string" ? parsed.lastSeenAt : ""
    };
  } catch {
    return null;
  }
}

function writeDaemonState(port: number, host: string): void {
  const existing = readDaemonState();
  const state: DaemonState = {
    pid: process.pid,
    host,
    port,
    startedAt: existing !== null && existing.pid === process.pid && existing.startedAt ? existing.startedAt : new Date().toISOString(),
    lastSeenAt: new Date().toISOString()
  };
  const path = daemonStatePath();
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function removeDaemonStateForCurrentProcess(): void {
  const state = readDaemonState();
  if (state?.pid === process.pid) removeDaemonState();
}

function removeDaemonState(): void {
  try {
    const path = daemonStatePath();
    if (existsSync(path)) unlinkSync(path);
  } catch {
    // Stale state should not prevent daemon startup.
  }
}

function serverPort(server: { address(): string | { port?: number } | null }): number {
  const address = server.address();
  if (typeof address === "object" && address && typeof address.port === "number") return address.port;
  throw new Error("PSM daemon failed to resolve listening port.");
}

function readJsonBody(req: { on(event: "data" | "end" | "error", callback: (...args: unknown[]) => void): void }): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += String(chunk);
    });
    req.on("end", () => {
      try {
        const parsed = JSON.parse(data || "{}") as unknown;
        resolve(isRecord(parsed) ? parsed : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function writeJson(res: { statusCode: number; setHeader(name: string, value: string): void; end(data: string): void }, status: number, value: unknown): void {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(`${JSON.stringify(value)}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function optionalStringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function numberValue(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
