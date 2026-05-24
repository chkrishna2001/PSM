#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { generateExamples } from "./adapters/local-psm.mjs";
import { writeJsonl } from "./lib/jsonl.mjs";

const args = parseArgs(process.argv.slice(2));
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/generated-local-psm");
const dbPath = stringArg(args, "db", "user_memory.db");
const docs = listArg(args, "docs", ["docs/indexables-conv.txt"]);
const maxDbRows = intArg(args, "max-db-rows", 250);
const maxDocExamples = intArg(args, "max-doc-examples", 40);
const maxMemoryChars = intArg(args, "max-memory-chars", 260);
const validationRatio = numberArg(args, "validation-ratio", 0.15);

const examples = generateExamples({ dbPath, docs, maxDbRows, maxDocExamples, maxMemoryChars });
const { train, validation } = splitDeterministically(examples, validationRatio);

mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), examples);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "local_psm",
  db: dbPath,
  docs,
  max_memory_chars: maxMemoryChars,
  total_examples: examples.length,
  train_examples: train.length,
  validation_examples: validation.length,
  validation_ratio: validationRatio
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  total: examples.length,
  train: train.length,
  validation: validation.length
}, null, 2));

function splitDeterministically(items, ratio) {
  const validation = [];
  const train = [];
  const interval = Math.max(2, Math.round(1 / ratio));
  items.forEach((item, index) => {
    if (index % interval === 0) validation.push(item);
    else train.push(item);
  });
  return { train, validation };
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) parsed[key] = "true";
    else parsed[key] = argv[++i];
  }
  return parsed;
}

function stringArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function listArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim()
    ? value.split(",").map((item) => item.trim()).filter(Boolean)
    : fallback;
}

function intArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

function numberArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isFinite(value) && value > 0 && value < 1 ? value : fallback;
}
