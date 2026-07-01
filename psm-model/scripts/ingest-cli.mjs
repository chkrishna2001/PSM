#!/usr/bin/env node
import path from "node:path";
import { pathToFileURL } from "node:url";

async function run() {
  const repo = process.env.PSM_REPO_ROOT || process.cwd();
  const mod = await import(
    pathToFileURL(path.join(repo, "dist/benchmark/locomo/src/ingest-psm-model.js")).href
  );
  const code = await mod.main(process.argv.slice(2));
  process.exit(code ?? 0);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
