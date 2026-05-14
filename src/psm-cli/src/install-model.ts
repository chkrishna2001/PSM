import { setupModel } from "./model.js";

async function main(): Promise<number> {
  if (process.env.PSM_MEMORY_SKIP_MODEL_DOWNLOAD === "1" || process.env.PSM_MEMORY_SKIP_MODEL_DOWNLOAD === "true") {
    process.stdout.write("Skipping PSM Memory model download because PSM_MEMORY_SKIP_MODEL_DOWNLOAD is set.\n");
    return 0;
  }

  try {
    await setupModel({
      log: (message) => process.stdout.write(`${message}\n`)
    });
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`PSM Memory model download failed: ${message}\n`);
    process.stderr.write("Run `psm-memory setup` after installation, or set PSM_MEMORY_SKIP_MODEL_DOWNLOAD=1 to skip intentionally.\n");
    return 0;
  }
}

process.exitCode = await main();
