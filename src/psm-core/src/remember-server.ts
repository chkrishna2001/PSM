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
  }): RememberServer {
    const key = [
      options.python,
      options.checkpoint,
      options.outputFormat,
      options.device,
      options.maxNewTokens
    ].join("|");
    const existing = servers.get(key);
    if (existing) return existing;

    const pythonPath = isAbsolute(options.python) ? options.python : resolve(options.repoRoot, options.python);
    const proc = spawn(
      pythonPath,
      [
        "-m",
        "psm_model.remember_server",
        options.checkpoint,
        "--output-format",
        options.outputFormat,
        "--device",
        options.device,
        "--max-new-tokens",
        String(options.maxNewTokens)
      ],
      {
        cwd: options.repoRoot,
        env: options.env,
        stdio: ["pipe", "pipe", "pipe"]
      }
    );
    const pending: PendingCall[] = [];
    const reader = createInterface({ input: proc.stdout });
    const ready = new Promise<void>((resolveReady, rejectReady) => {
      let sawReady = false;
      reader.on("line", (line: string) => {
        if (!sawReady) {
          sawReady = true;
          try {
            const message = JSON.parse(line) as { ready?: boolean };
            if (message.ready) {
              resolveReady();
              return;
            }
          } catch {
            // fall through
          }
          rejectReady(new Error(`remember server bad ready line: ${line}`));
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
      proc.on("error", rejectReady);
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
