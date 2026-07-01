import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { isAbsolute, resolve } from "node:path";

interface PendingCall {
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
}

const servers = new Map<string, RememberServer>();

export class RememberServer {
  private readonly proc: ChildProcessWithoutNullStreams;
  private readonly ready: Promise<void>;
  private readonly pending: PendingCall[];
  private chain: Promise<void> = Promise.resolve();

  private constructor(proc: ChildProcessWithoutNullStreams, ready: Promise<void>, pending: PendingCall[]) {
    this.proc = proc;
    this.ready = ready;
    this.pending = pending;
    proc.on("exit", (code) => {
      const error = new Error(`remember server exited with status ${code ?? "unknown"}`);
      while (this.pending.length > 0) {
        this.pending.shift()?.reject(error);
      }
    });
  }

  static get(options: {
    python: string;
    checkpoint: string;
    repoRoot: string;
    outputFormat: string;
    device: string;
    maxNewTokens: number;
    env: NodeJS.ProcessEnv;
    hfBinaryAdapter?: string;
    hfExtractAdapter?: string;
    hfModelKey?: string;
  }): RememberServer {
    const hfMode = Boolean(options.hfBinaryAdapter && options.hfExtractAdapter);
    const key = [
      options.python,
      hfMode ? "hf-two-pass" : options.checkpoint,
      options.hfBinaryAdapter ?? "",
      options.hfExtractAdapter ?? "",
      options.outputFormat,
      options.device,
      options.maxNewTokens
    ].join("|");
    const existing = servers.get(key);
    if (existing) return existing;

    const bareName =
      !options.python.includes("/") && !options.python.includes("\\");
    const pythonPath = isAbsolute(options.python) || bareName
      ? options.python
      : resolve(options.repoRoot, options.python);
    const spawnArgs = hfMode
      ? [
          "-m",
          "psm_model.hf_remember_server",
          "--binary-adapter",
          options.hfBinaryAdapter!,
          "--extract-adapter",
          options.hfExtractAdapter!,
          "--model",
          options.hfModelKey ?? "qwen0.5b",
          "--device",
          options.device,
          "--max-new-tokens",
          String(options.maxNewTokens)
        ]
      : [
          "-m",
          "psm_model.remember_server",
          options.checkpoint,
          "--output-format",
          options.outputFormat,
          "--device",
          options.device,
          "--max-new-tokens",
          String(options.maxNewTokens)
        ];
    const proc = spawn(pythonPath, spawnArgs, {
      cwd: options.repoRoot,
      env: options.env,
      stdio: ["pipe", "pipe", "pipe"]
    });
    let stderrTail = "";
    proc.stderr.on("data", (chunk: Buffer) => {
      stderrTail = (stderrTail + chunk.toString()).slice(-4000);
      process.stderr.write(chunk);
    });
    const pending: PendingCall[] = [];
    const forgetServer = () => {
      servers.delete(key);
    };
    const reader = createInterface({ input: proc.stdout });
    const ready = new Promise<void>((resolveReady, rejectReady) => {
      let readySettled = false;
      const settleReady = (ok: boolean, error?: Error) => {
        if (readySettled) return;
        readySettled = true;
        if (!ok) forgetServer();
        const detail = stderrTail.trim();
        if (ok) resolveReady();
        else {
          rejectReady(
            error ??
              new Error(detail ? `remember server failed to start: ${detail}` : "remember server failed to start")
          );
        }
      };
      reader.on("line", (line: string) => {
        if (!readySettled) {
          try {
            const message = JSON.parse(line) as { ready?: boolean };
            if (message.ready) {
              settleReady(true);
              return;
            }
            settleReady(false, new Error(`remember server bad ready line: ${line}`));
          } catch {
            settleReady(false, new Error(`remember server bad ready line: ${line}`));
          }
          return;
        }
        const waiter = pending.shift();
        if (!waiter) return;
        try {
          const parsed = JSON.parse(line) as Record<string, unknown>;
          if (typeof parsed.error === "string") {
            waiter.reject(new Error(parsed.error));
            return;
          }
          waiter.resolve(parsed);
        } catch (error) {
          waiter.reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
      proc.stdout.on("close", () => {
        if (!readySettled) {
          const detail = stderrTail.trim();
          settleReady(
            false,
            new Error(detail ? `remember server stdout closed before ready: ${detail}` : "remember server stdout closed before ready")
          );
        }
      });
      proc.on("error", (error) => {
        settleReady(false, error instanceof Error ? error : new Error(String(error)));
      });
      proc.on("exit", (code, signal) => {
        const detail = stderrTail.trim();
        settleReady(
          false,
          new Error(
            detail
              ? `remember server exited with status ${code ?? signal ?? "unknown"} before ready: ${detail}`
              : `remember server exited with status ${code ?? signal ?? "unknown"} before ready`
          )
        );
      });
    });
    const server = new RememberServer(proc, ready, pending);
    servers.set(key, server);
    return server;
  }

  async ensureReady(): Promise<void> {
    await this.ready;
  }

  async remember(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
    await this.ready;
    const task = this.chain.then(() => this.writeRequest(payload));
    this.chain = task.then(() => undefined, () => undefined);
    return task;
  }

  private writeRequest(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      this.proc.stdin.write(`${JSON.stringify({ payload })}\n`, (error) => {
        if (error) {
          this.pending.pop();
          reject(error);
        }
      });
    });
  }
}
