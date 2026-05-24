#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const args = parseArgs(process.argv.slice(2));
const dataDir = stringArg(args, "data-dir", "nano-psm/data-pipeline/data/generated");
const reportDir = stringArg(args, "report-dir", "nano-psm/data-pipeline/reports/gate-current");
const outDir = stringArg(args, "out", "hf-upload/nano-psm-training-data");
const name = stringArg(args, "name", "nano-psm-training-data");

requireFile(join(dataDir, "train.jsonl"));
requireFile(join(dataDir, "validation.jsonl"));
requireFile(join(dataDir, "metadata.json"));
requireFile(join(reportDir, "quality-audit.json"));
requireFile(join(reportDir, "dataset-summary.json"));

mkdirSync(outDir, { recursive: true });
mkdirSync(join(outDir, "reports"), { recursive: true });

copyFileSync(join(dataDir, "train.jsonl"), join(outDir, "train.jsonl"));
copyFileSync(join(dataDir, "validation.jsonl"), join(outDir, "validation.jsonl"));
copyFileSync(join(dataDir, "metadata.json"), join(outDir, "metadata.json"));

for (const file of readdirSync(reportDir)) {
  if (file.endsWith(".json") || file.endsWith(".jsonl")) {
    copyFileSync(join(reportDir, file), join(outDir, "reports", file));
  }
}

writeFileSync(join(outDir, "README.md"), datasetCard(name), "utf8");

console.log(JSON.stringify({
  status: "prepared",
  out_dir: outDir,
  files: ["train.jsonl", "validation.jsonl", "metadata.json", "README.md", "reports/"]
}, null, 2));

function datasetCard(datasetName) {
  return `---
pretty_name: ${datasetName}
task_categories:
- text-classification
- token-classification
- question-answering
language:
- en
tags:
- psm
- memory
- indexables
- recall
---

# ${datasetName}

This dataset contains quality-gated examples for training Nano PSM, a structured memory-operation model.

## Files

- \`train.jsonl\`
- \`validation.jsonl\`
- \`metadata.json\`
- \`reports/dataset-summary.json\`
- \`reports/action-mix.json\`
- \`reports/source-mix.json\`
- \`reports/quality-audit.json\`
- \`reports/review-sample.jsonl\`

## Quality Gate

This export is expected to be produced only after:

1. canonical schema validation
2. runtime compatibility validation
3. dataset quality audit

Do not train Nano PSM on ungated data.
`;
}

function requireFile(path) {
  if (!existsSync(path)) {
    throw new Error(`Required file missing: ${path}`);
  }
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) parsed[key] = "true";
    else parsed[key] = next, i++;
  }
  return parsed;
}

function stringArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

