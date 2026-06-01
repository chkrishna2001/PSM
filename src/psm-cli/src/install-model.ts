import { appendFileSync, mkdirSync } from "node:fs";
import { platform } from "node:os";
import { dirname, join } from "node:path";
import { defaultPsmConfigPath } from "@psm-memory/sdk";
import { run } from "./index.js";

async function main(): Promise<number> {
  const logPath = resolveInstallLogPath();
  process.env.PSM_MEMORY_INSTALL_LOG = logPath;
  installLog(logPath, "postinstall_start", {
    node: process.execPath,
    platform: platform(),
    arch: process.env.PROCESSOR_ARCHITECTURE || process.env.npm_config_arch,
    npm_execpath: process.env.npm_execpath,
    npm_config_prefix: process.env.npm_config_prefix,
    npm_lifecycle_event: process.env.npm_lifecycle_event,
    https_proxy: proxySet("HTTPS_PROXY"),
    http_proxy: proxySet("HTTP_PROXY"),
    node_extra_ca_certs: Boolean(process.env.NODE_EXTRA_CA_CERTS)
  });

  try {
    const args = ["setup", "--yes", "--pretty"];
    const code = await run(args);
    if (code !== 0) {
      installLog(logPath, "postinstall_failed", { exit_code: code });
      process.stderr.write(`PSM Memory install-time setup failed. See install log: ${logPath}\n`);
      return code;
    }
    installLog(logPath, "postinstall_ok", {});
    return 0;
  } catch (error) {
    installLog(logPath, "postinstall_error", { error: serializeError(error) });
    process.stderr.write(`PSM Memory install-time setup failed. See install log: ${logPath}\n`);
    return 1;
  }
}

process.exitCode = await main();

function resolveInstallLogPath(): string {
  const candidates = [
    join(dirname(defaultPsmConfigPath()), "install.log"),
    join(tempDir(), "psm-memory-install.log")
  ];
  for (const path of candidates) {
    try {
      mkdirSync(dirname(path), { recursive: true });
      appendFileSync(path, "", "utf8");
      return path;
    } catch {
      // Try the next location.
    }
  }
  return join(tempDir(), "psm-memory-install.log");
}

function installLog(path: string, event: string, data: Record<string, unknown>): void {
  try {
    appendFileSync(path, `${JSON.stringify({ ts: new Date().toISOString(), event, ...data })}\n`, "utf8");
  } catch {
    // Install logging must not hide the original install failure.
  }
}

function proxySet(name: "HTTP_PROXY" | "HTTPS_PROXY"): boolean {
  return Boolean(process.env[name] || process.env[name.toLowerCase()]);
}

function serializeError(error: unknown): Record<string, unknown> {
  if (!(error instanceof Error)) return { message: String(error) };
  const detail = error as Error & { code?: string; syscall?: string; path?: string; dest?: string; errno?: number };
  return {
    name: error.name,
    message: error.message,
    code: detail.code,
    errno: detail.errno,
    syscall: detail.syscall,
    path: detail.path,
    dest: detail.dest,
    stack: error.stack
  };
}

function tempDir(): string {
  return process.env.TEMP || process.env.TMP || dirname(defaultPsmConfigPath());
}
