#!/usr/bin/env node
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { readJsonl, writeJsonl } from "./lib/jsonl.mjs";

const args = parseArgs(process.argv.slice(2));
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/retention-blend-7k");

const sources = [
  {
    name: "reviewed_5k",
    dir: "nano-psm/data-pipeline/data/fast-mixed-reviewed",
    trainLimit: intArg(args, "reviewed-train-limit", 2000),
    validationLimit: intArg(args, "reviewed-validation-limit", 175)
  },
  {
    name: "reviewed_v2",
    dir: "nano-psm/data-pipeline/data/fast-mixed-reviewed-v2",
    trainLimit: intArg(args, "reviewed-v2-train-limit", 2000),
    validationLimit: intArg(args, "reviewed-v2-validation-limit", 175)
  },
  {
    name: "incremental_5k",
    dir: "nano-psm/data-pipeline/data/fast-mixed-reviewed-incremental-5k",
    trainLimit: intArg(args, "incremental-train-limit", 1500),
    validationLimit: intArg(args, "incremental-validation-limit", 175)
  },
  {
    name: "retention_decay_5k",
    dir: "nano-psm/data-pipeline/data/retention-decay-5k",
    trainLimit: intArg(args, "retention-train-limit", 1500),
    validationLimit: intArg(args, "retention-validation-limit", 175)
  },
  {
    name: "codex_sessions_gpt41_mini_200",
    dir: "nano-psm/data-pipeline/data/codex-sessions-gpt41-mini-200",
    trainLimit: intArg(args, "codex-train-limit", 160),
    validationLimit: intArg(args, "codex-validation-limit", 40)
  }
];

const train = [];
const validation = [];
const loaded = {};

for (const source of sources) {
  const trainRows = requireRows(join(source.dir, "train.jsonl"));
  const validationRows = requireRows(join(source.dir, "validation.jsonl"));
  const selectedTrain = deterministicPick(trainRows, source.trainLimit, source.name);
  const selectedValidation = deterministicPick(validationRows, source.validationLimit, source.name);
  train.push(...selectedTrain.map((row) => tagSource(row, source.name)));
  validation.push(...selectedValidation.map((row) => tagSource(row, source.name)));
  loaded[source.name] = {
    train_available: trainRows.length,
    train_selected: selectedTrain.length,
    validation_available: validationRows.length,
    validation_selected: selectedValidation.length
  };
}

const all = [...train, ...validation];
mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), all);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "retention_blend",
  total_examples: all.length,
  train_examples: train.length,
  validation_examples: validation.length,
  loaded,
  action_mix: countBy(all, (row) => row.output?.action ?? "unknown"),
  train_action_mix: countBy(train, (row) => row.output?.action ?? "unknown"),
  validation_action_mix: countBy(validation, (row) => row.output?.action ?? "unknown"),
  source_mix: countBy(all, (row) => row.input?.blend_source ?? row.input?.source_kind ?? "unknown")
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  train: train.length,
  validation: validation.length,
  action_mix: countBy(all, (row) => row.output?.action ?? "unknown"),
  source_mix: countBy(all, (row) => row.input?.blend_source ?? row.input?.source_kind ?? "unknown")
}, null, 2));

function requireRows(path) {
  if (!existsSync(path)) throw new Error(`Missing required file: ${path}`);
  return readJsonl(path);
}

function deterministicPick(rows, limit, salt) {
  if (rows.length <= limit) return rows;
  return rows
    .map((row, index) => ({ row, key: hash(`${salt}:${row.id ?? index}:${index}`) }))
    .sort((left, right) => left.key - right.key)
    .slice(0, limit)
    .map((item) => item.row)
    .sort((left, right) => String(left.id ?? "").localeCompare(String(right.id ?? "")));
}

function tagSource(row, blendSource) {
  return {
    ...row,
    id: `${blendSource}:${row.id}`,
    input: {
      ...row.input,
      blend_source: blendSource
    }
  };
}

function countBy(rows, getKey) {
  const counts = {};
  for (const row of rows) {
    const key = getKey(row);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function hash(value) {
  let result = 2166136261;
  for (let index = 0; index < value.length; index++) {
    result ^= value.charCodeAt(index);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
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

function intArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}
