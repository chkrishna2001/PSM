#!/usr/bin/env node
// ponytail: CJS wrapper — dynamic import().then() keeps Node alive until async main() finishes
const { pathToFileURL } = require("node:url");
const path = require("node:path");

const repo = process.env.PSM_REPO_ROOT || process.cwd();
const ingest = path.join(repo, "dist/benchmark/locomo/src/ingest-psm-model.js");

import(pathToFileURL(ingest).href)
  .then((mod) => {
    console.error("run_ingest: calling main");
    return mod.main(process.argv.slice(2));
  })
  .then((code) => {
    console.error("run_ingest: done code=", code);
    process.exit(code ?? 0);
  })
  .catch((err) => {
    console.error("run_ingest: error", err);
    process.exit(1);
  });
