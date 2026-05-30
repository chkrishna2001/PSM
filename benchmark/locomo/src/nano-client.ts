import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

export interface NanoClientOptions {
  python: string;
  script: string;
  config: string;
  checkpoint: string;
  device: string;
}

export interface NanoPrediction {
  action: string;
  memory: Record<string, unknown> | null;
  facts?: Record<string, unknown>[];
  indexables?: Record<string, unknown>[];
  updates?: Record<string, unknown>[];
  conflicts?: Record<string, unknown>[];
  reasoning?: string;
  confidence?: number;
  nano?: Record<string, unknown>;
  parse_error?: string;
}

export class NanoClient {
  private readonly child: ReturnType<typeof spawn>;
  private readonly lines: ReturnType<typeof createInterface>;
  private readonly pending: Array<{ resolve: (value: NanoPrediction) => void; reject: (error: Error) => void }> = [];

  constructor(options: NanoClientOptions) {
    this.child = spawn(options.python, [
      options.script,
      "--config", options.config,
      "--checkpoint", options.checkpoint,
      "--device", options.device
    ], {
      stdio: ["pipe", "pipe", "pipe"]
    });
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on("line", (line: string) => this.handleLine(line));
    this.child.stderr?.on("data", (chunk: unknown) => process.stderr.write(String(chunk)));
    this.child.on("error", (error: Error) => this.rejectAll(error));
    this.child.on("exit", (code: number | null, signal: string | null) => {
      if (this.pending.length > 0) {
        this.rejectAll(new Error(`Nano PSM process exited before replying: code=${code ?? "null"} signal=${signal ?? "null"}`));
      }
    });
  }

  predict(row: Record<string, unknown>): Promise<NanoPrediction> {
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      this.child.stdin?.write(`${JSON.stringify(row)}\n`, "utf8", (error?: Error | null) => {
        if (error) {
          const pending = this.pending.pop();
          pending?.reject(error);
        }
      });
    });
  }

  close(): void {
    this.lines.close();
    this.child.stdin?.end();
  }

  private handleLine(line: string): void {
    const pending = this.pending.shift();
    if (!pending) return;
    try {
      pending.resolve(JSON.parse(line) as NanoPrediction);
    } catch (error) {
      pending.reject(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private rejectAll(error: Error): void {
    while (this.pending.length > 0) {
      this.pending.shift()?.reject(error);
    }
  }
}
