import { run } from "./index.js";

async function main(): Promise<number> {
  if (process.env.PSM_MEMORY_SKIP_SETUP === "1" || process.env.PSM_MEMORY_SKIP_SETUP === "true") {
    process.stdout.write("Skipping PSM Memory setup because PSM_MEMORY_SKIP_SETUP is set.\n");
    return 0;
  }

  try {
    const args = ["setup"];
    if (process.env.PSM_MEMORY_SKIP_MODEL_DOWNLOAD === "1" || process.env.PSM_MEMORY_SKIP_MODEL_DOWNLOAD === "true") {
      args.push("--skip-model", "--skip-embeddings", "--yes");
    }
    const code = await run(args);
    if (code !== 0) {
      process.stderr.write("PSM Memory setup did not complete during install. Run `psm-memory setup` after installation.\n");
    }
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`PSM Memory setup failed during install: ${message}\n`);
    process.stderr.write("Run `psm-memory setup` after installation, or set PSM_MEMORY_SKIP_SETUP=1 to skip intentionally.\n");
    return 0;
  }
}

process.exitCode = await main();
