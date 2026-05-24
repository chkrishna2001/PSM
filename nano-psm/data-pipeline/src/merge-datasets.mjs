#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { readJsonl, writeJsonl } from "./lib/jsonl.mjs";

const args = parseArgs(process.argv.slice(2));
const inputs = listArg(args, "inputs", []);
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/merged");
const validationRatio = numberArg(args, "validation-ratio", 0.15);
const maxTotal = intArg(args, "max-total", 0);

if (inputs.length === 0) {
  console.error("Usage: node nano-psm/data-pipeline/src/merge-datasets.mjs --inputs <dir1,dir2,...> --out <out-dir>");
  process.exit(2);
}

const loaded = [];
for (const input of inputs) {
  loaded.push(...loadDatasetDir(input));
}

const deduped = dedupe(loaded);
const limited = maxTotal > 0 ? balancedLimit(deduped, maxTotal) : deduped;
const { train, validation } = splitDeterministically(limited, validationRatio);

mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), limited.map((item) => item.row));
writeJsonl(join(outDir, "train.jsonl"), train.map((item) => item.row));
writeJsonl(join(outDir, "validation.jsonl"), validation.map((item) => item.row));
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "merged",
  inputs,
  loaded_examples: loaded.length,
  duplicate_examples_removed: loaded.length - deduped.length,
  total_examples: limited.length,
  train_examples: train.length,
  validation_examples: validation.length,
  validation_ratio: validationRatio,
  max_total: maxTotal || null,
  action_mix: countBy(limited, (item) => item.row.output?.action ?? "unknown"),
  source_mix: countBy(limited, (item) => item.row.input?.source_kind ?? "unknown")
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  loaded: loaded.length,
  deduped: deduped.length,
  total: limited.length,
  train: train.length,
  validation: validation.length
}, null, 2));

function loadDatasetDir(dir) {
  const metadata = readMetadata(dir);
  const rows = [
    ...readJsonl(join(dir, "train.jsonl")).map((row) => ({ split: "train", row })),
    ...readJsonl(join(dir, "validation.jsonl")).map((row) => ({ split: "validation", row }))
  ];
  return rows.map((item, index) => ({
    ...item,
    source_dataset: metadata.source ?? metadata.name ?? dir,
    source_dataset_dir: dir,
    order: index
  }));
}

function readMetadata(dir) {
  try {
    return JSON.parse(readFileSync(join(dir, "metadata.json"), "utf8"));
  } catch {
    return { source: dir };
  }
}

function dedupe(items) {
  const seen = new Set();
  const result = [];
  for (const item of items) {
    const key = dedupeKey(item.row);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(item);
  }
  return result;
}

function dedupeKey(row) {
  const action = row.output?.action ?? "";
  const sourceKind = row.input?.source_kind ?? "";
  const sourceId = row.input?.source_id ?? "";
  const memory = normalize(row.output?.memory?.content ?? "");
  const query = normalize(row.input?.current_query?.question ?? "");
  const text = normalize(row.input?.current_turn?.text ?? "");
  return [action, sourceKind, sourceId, memory || query || text].join("|");
}

function balancedLimit(items, maxTotal) {
  if (items.length <= maxTotal) return items;
  const byAction = new Map();
  for (const item of items) {
    const action = item.row.output?.action ?? "unknown";
    const list = byAction.get(action) ?? [];
    list.push(item);
    byAction.set(action, list);
  }
  const actions = [...byAction.keys()].sort();
  const base = Math.max(1, Math.floor(maxTotal / actions.length));
  const selected = [];
  for (const action of actions) {
    selected.push(...byAction.get(action).slice(0, base));
  }
  let cursor = 0;
  while (selected.length < maxTotal && cursor < items.length) {
    const item = items[cursor++];
    if (!selected.includes(item)) selected.push(item);
  }
  return selected.slice(0, maxTotal);
}

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

function countBy(items, getKey) {
  const counts = {};
  for (const item of items) {
    const key = getKey(item);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function normalize(value) {
  return String(value).toLowerCase().replace(/\s+/g, " ").trim();
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

function listArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim()
    ? value.split(",").map((item) => item.trim()).filter(Boolean)
    : fallback;
}

function stringArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

function numberArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isFinite(value) && value > 0 && value < 1 ? value : fallback;
}

