import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { defaultHfBinaryAdapter, defaultHfExtractAdapter } from "@psm-memory/sdk";

export const defaultModel = {
  kind: "hf-two-pass",
  modelKey: "qwen0.5b",
  binaryAdapter: defaultHfBinaryAdapter,
  extractAdapter: defaultHfExtractAdapter
} as const;

export function resolveRepoRoot(): string {
  const cwd = process.cwd();
  if (existsSync(resolve(cwd, "psm-model", "src", "psm_model"))) return cwd;
  return cwd;
}

export function defaultModelPath(): string {
  return `hf-two-pass:${defaultModel.binaryAdapter}+${defaultModel.extractAdapter}`;
}

export function hfAdapterPaths(repoRoot = resolveRepoRoot()): { binary: string; extract: string } {
  return {
    binary: resolve(repoRoot, defaultModel.binaryAdapter),
    extract: resolve(repoRoot, defaultModel.extractAdapter)
  };
}

export function hasDefaultModel(repoRoot = resolveRepoRoot()): boolean {
  const paths = hfAdapterPaths(repoRoot);
  return existsSync(join(paths.binary, "adapter_config.json")) && existsSync(join(paths.extract, "adapter_config.json"));
}

export function resolveModelPath(): string {
  if (hasDefaultModel()) return defaultModelPath();
  const paths = hfAdapterPaths();
  throw new Error([
    "PSM HF adapters are not available.",
    `Expected gate adapter at ${paths.binary}`,
    `Expected extract adapter at ${paths.extract}`,
    "Clone the PSM repo with psm-model checkpoints or run setup from the repo root."
  ].join(" "));
}

export async function setupModel(options: { force?: boolean; log?: (message: string) => void; repoRoot?: string } = {}): Promise<string> {
  const repoRoot = options.repoRoot ?? resolveRepoRoot();
  const paths = hfAdapterPaths(repoRoot);
  if (!options.force && hasDefaultModel(repoRoot)) {
    options.log?.(`PSM HF adapters ready: ${defaultModelPath()}`);
    return defaultModelPath();
  }
  if (!existsSync(join(paths.binary, "adapter_config.json"))) {
    throw new Error(`Missing gate adapter: ${paths.binary}`);
  }
  if (!existsSync(join(paths.extract, "adapter_config.json"))) {
    throw new Error(`Missing extract adapter: ${paths.extract}`);
  }
  options.log?.(`Verified PSM HF adapters: ${defaultModelPath()}`);
  return defaultModelPath();
}
