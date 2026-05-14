import { createWriteStream, existsSync, mkdirSync, renameSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { homedir, platform } from "node:os";

export const defaultModel = {
  repo: "chkrishna2001/psm-memory-qwen-1.5b-gguf",
  file: "gguf/psm-memory-qwen-1.5b-q4_k_m.gguf",
  filename: "psm-memory-qwen-1.5b-q4_k_m.gguf",
  size: 986047808,
  sha256: "05a35ea07f27514e20db9f55e28d4e6f51a15c1684125067a0b65f9b483cc6e3"
} as const;

export function modelCacheDir(): string {
  const override = process.env.PSM_MEMORY_HOME;
  if (override?.trim()) return override;
  if (platform() === "win32") {
    return join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "psm-memory", "models");
  }
  return join(process.env.XDG_CACHE_HOME || join(homedir(), ".cache"), "psm-memory", "models");
}

export function defaultModelPath(): string {
  return join(modelCacheDir(), defaultModel.filename);
}

export function hasDefaultModel(): boolean {
  if (!existsSync(defaultModelPath())) return false;
  return statSync(defaultModelPath()).size === defaultModel.size;
}

export function resolveModelPath(): string {
  if (hasDefaultModel()) return defaultModelPath();
  throw new Error([
    "PSM Memory model is not installed.",
    `Run "psm-memory setup" to download ${defaultModel.filename}.`,
    "This usually means the npm install-time download was skipped, interrupted, or blocked by network settings."
  ].join(" "));
}

export async function setupModel(options: { force?: boolean; log?: (message: string) => void } = {}): Promise<string> {
  const target = defaultModelPath();
  if (!options.force && hasDefaultModel()) {
    options.log?.(`PSM model already installed: ${target}`);
    return target;
  }

  mkdirSync(modelCacheDir(), { recursive: true });
  const temp = `${target}.download`;
  if (existsSync(temp)) unlinkSync(temp);

  const url = `https://huggingface.co/${defaultModel.repo}/resolve/main/${defaultModel.file}`;
  options.log?.(`Downloading PSM Memory model from ${url}`);
  options.log?.(`Target: ${target}`);

  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Model download failed: ${response.status} ${response.statusText}`);
  }

  await writeResponseBody(response.body, temp, defaultModel.size, options.log);
  const actualSize = statSync(temp).size;
  if (actualSize !== defaultModel.size) {
    unlinkSync(temp);
    throw new Error(`Model download size mismatch: expected ${defaultModel.size}, got ${actualSize}`);
  }

  renameSync(temp, target);
  options.log?.(`Installed PSM model: ${target}`);
  return target;
}

async function writeResponseBody(body: unknown, target: string, expectedSize: number, log?: (message: string) => void): Promise<void> {
  const stream = createWriteStream(target);
  const reader = (body as { getReader(): { read(): Promise<{ done: boolean; value?: Uint8Array }> } }).getReader();
  let written = 0;
  let nextReport = 0;

  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      if (!chunk.value) continue;
      written += chunk.value.byteLength;
      if (!stream.write(chunk.value)) {
        await new Promise<void>((resolve) => stream.once("drain", () => resolve()));
      }
      const percent = Math.floor((written / expectedSize) * 100);
      if (percent >= nextReport) {
        log?.(`Downloaded ${percent}%`);
        nextReport += 10;
      }
    }
  } finally {
    stream.end();
  }

  await new Promise<void>((resolve, reject) => {
    stream.once("finish", () => resolve());
    stream.once("error", (error) => reject(error));
  });
}
