#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { readJsonl, writeJsonl } from "./lib/jsonl.mjs";

const args = parseArgs(process.argv.slice(2));
const files = listArg(args, "files", args._);
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/reports/latest");
const sampleSize = intArg(args, "sample-size", 80);
const failOnWarnings = boolArg(args, "fail-on-warnings", false);

if (files.length === 0) {
  console.error("Usage: node nano-psm/data-pipeline/src/audit-quality.mjs --files <train.jsonl> <validation.jsonl> [--out reports/latest]");
  process.exit(2);
}

const rows = [];
for (const file of files) {
  readJsonl(file).forEach((row, index) => rows.push({ file, line: index + 1, row }));
}

const report = auditRows(rows);
mkdirSync(outDir, { recursive: true });
writeJson(join(outDir, "dataset-summary.json"), report.summary);
writeJson(join(outDir, "action-mix.json"), report.action_mix);
writeJson(join(outDir, "source-mix.json"), report.source_mix);
writeJson(join(outDir, "quality-audit.json"), {
  failures: report.failures,
  warnings: report.warnings,
  duplicate_memories: report.duplicate_memories,
  weak_indexables: report.weak_indexables
});
writeJsonl(join(outDir, "review-sample.jsonl"), reviewSample(rows, sampleSize));

console.log(JSON.stringify({
  checked: rows.length,
  failures: report.failures.length,
  warnings: report.warnings.length,
  duplicate_memory_groups: report.duplicate_memories.length,
  weak_indexables: report.weak_indexables.length,
  reports: outDir
}, null, 2));

if (report.failures.length > 0 || (failOnWarnings && report.warnings.length > 0)) {
  process.exit(1);
}

function auditRows(items) {
  const failures = [];
  const warnings = [];
  const weakIndexables = [];
  const actionMix = {};
  const sourceMix = {};
  const memoryContent = new Map();
  const indexableKeys = {};
  let recallRows = 0;
  let rowsWithFacts = 0;
  let rowsWithIndexables = 0;
  let totalIndexables = 0;

  for (const item of items) {
    const { row, file, line } = item;
    const where = `${file}:${line}`;
    const action = row?.output?.action ?? "unknown";
    const sourceKind = row?.input?.source_kind ?? "unknown";
    actionMix[action] = (actionMix[action] ?? 0) + 1;
    sourceMix[sourceKind] = (sourceMix[sourceKind] ?? 0) + 1;

    if (!row?.input?.source_kind) failures.push(issue(where, "missing_source_kind", "input.source_kind is required for source mix and leakage audits"));
    if (!row?.input?.source_id && action !== "recall_context") warnings.push(issue(where, "missing_source_id", "non-recall example should preserve source_id"));

    const memory = row?.output?.memory;
    if (memory?.content) {
      const normalizedContent = normalizeText(memory.content);
      const group = memoryContent.get(normalizedContent) ?? [];
      group.push(where);
      memoryContent.set(normalizedContent, group);
      auditMemory(where, row, memory, failures, warnings);
    }

    if (Array.isArray(row?.output?.facts) && row.output.facts.length > 0) {
      rowsWithFacts++;
      row.output.facts.forEach((fact, index) => auditFact(where, fact, index, failures, warnings));
    }

    if (Array.isArray(row?.output?.indexables) && row.output.indexables.length > 0) {
      rowsWithIndexables++;
      totalIndexables += row.output.indexables.length;
      row.output.indexables.forEach((indexable, index) => {
        const result = auditIndexable(where, indexable, index, failures, warnings);
        if (result) weakIndexables.push(result);
        if (indexable?.key) indexableKeys[indexable.key] = (indexableKeys[indexable.key] ?? 0) + 1;
      });
    }

    if (action === "recall_context") {
      recallRows++;
      auditRecall(where, row, failures, warnings);
    }
  }

  const duplicateMemories = [...memoryContent.entries()]
    .filter(([, locations]) => locations.length > 3)
    .map(([content, locations]) => ({ content, count: locations.length, locations: locations.slice(0, 12) }));

  for (const duplicate of duplicateMemories) {
    warnings.push(issue(duplicate.locations[0], "duplicate_memory_content", `memory content appears ${duplicate.count} times`));
  }

  const checked = items.length;
  const ignoreRatio = ratio(actionMix.ignore, checked);
  const recallRatio = ratio(actionMix.recall_context, checked);
  if (checked >= 100 && ignoreRatio > 0.45) warnings.push(issue("dataset", "ignore_overrepresented", `ignore ratio is ${ignoreRatio.toFixed(3)}`));
  if (checked >= 100 && recallRatio < 0.05) warnings.push(issue("dataset", "recall_underrepresented", `recall_context ratio is ${recallRatio.toFixed(3)}`));
  if (checked >= 100 && rowsWithIndexables / checked < 0.35) warnings.push(issue("dataset", "indexables_underrepresented", "fewer than 35% of rows include indexables"));

  return {
    summary: {
      checked,
      files: [...new Set(items.map((item) => item.file))],
      rows_with_facts: rowsWithFacts,
      rows_with_indexables: rowsWithIndexables,
      total_indexables: totalIndexables,
      recall_rows: recallRows,
      unique_indexable_keys: Object.keys(indexableKeys).length
    },
    action_mix: actionMix,
    source_mix: sourceMix,
    failures,
    warnings,
    duplicate_memories: duplicateMemories,
    weak_indexables: weakIndexables
  };
}

function auditMemory(where, row, memory, failures, warnings) {
  const content = String(memory.content);
  if (content.length > 260) warnings.push(issue(where, "memory_too_long", `memory.content is ${content.length} chars`));
  if (content.length < 16) warnings.push(issue(where, "memory_too_short", "memory.content is too short to be useful"));
  if (/Current utterance:|Previous context:|Return JSON|target schema/i.test(content)) {
    failures.push(issue(where, "wrapper_text_leak", "memory.content contains prompt/schema wrapper text"));
  }
  if (/^(ok|thanks|thank you|hello|hi)[.! ]*$/i.test(content)) {
    failures.push(issue(where, "trivial_memory", "stored memory is trivial chatter"));
  }
  const speaker = row?.input?.current_turn?.speaker;
  if (/^\s*User\b/.test(content) && speaker && speaker !== "User") {
    failures.push(issue(where, "generic_user_with_named_speaker", `memory uses User but speaker is ${speaker}`));
  }
}

function auditFact(where, fact, index, failures, warnings) {
  if (!fact?.evidence_text) failures.push(issue(where, "fact_missing_evidence", `facts[${index}] missing evidence_text`));
  if (fact?.evidence_text && String(fact.evidence_text).length < 8) warnings.push(issue(where, "fact_weak_evidence", `facts[${index}] evidence_text is very short`));
  if (fact?.confidence < 0.5) warnings.push(issue(where, "fact_low_confidence", `facts[${index}] confidence below 0.5`));
  if (fact?.inference_kind !== "explicit") failures.push(issue(where, "fact_not_explicit", `facts[${index}] inference_kind must remain explicit for training`));
}

function auditIndexable(where, indexable, index, failures, warnings) {
  const key = String(indexable?.key ?? "");
  if (!key) {
    failures.push(issue(where, "indexable_missing_key", `indexables[${index}] missing key`));
    return null;
  }
  const tokens = key.split("-");
  const weak = tokens.length < 2
    || tokens.length > 5
    || new Set(["memory", "user", "local", "tools", "project", "system", "thing"]).has(key)
    || tokens.some((token) => token.length < 3 && !/^\d+$/.test(token));
  if (weak) {
    const item = issue(where, "weak_indexable_key", `indexables[${index}] key may be too generic: ${key}`);
    warnings.push(item);
    return { ...item, key };
  }
  if (!indexable.evidence_text) failures.push(issue(where, "indexable_missing_evidence", `indexables[${index}] missing evidence_text`));
  if (!indexable.reconstructive_hint || String(indexable.reconstructive_hint).length < 20) {
    warnings.push(issue(where, "indexable_weak_hint", `indexables[${index}] reconstructive_hint is weak`));
  }
  return null;
}

function auditRecall(where, row, failures, warnings) {
  const recall = row?.output?.recall;
  const store = row?.input?.memory_store ?? [];
  if (!Array.isArray(store) || store.length === 0) failures.push(issue(where, "recall_missing_memory_store", "recall row needs candidate memory_store"));
  if (!Array.isArray(recall?.selected_memory_ids)) failures.push(issue(where, "recall_missing_selected_ids", "recall row missing selected_memory_ids"));
  if ((recall?.selected_memory_ids ?? []).length === 0 && store.length > 0) warnings.push(issue(where, "empty_recall_selection", "recall selected no memory ids despite candidates"));
}

function reviewSample(items, sampleSize) {
  const byAction = new Map();
  for (const item of items) {
    const action = item.row?.output?.action ?? "unknown";
    const list = byAction.get(action) ?? [];
    list.push(item);
    byAction.set(action, list);
  }
  const sample = [];
  const actions = [...byAction.keys()].sort();
  const perAction = Math.max(1, Math.floor(sampleSize / Math.max(1, actions.length)));
  for (const action of actions) {
    const list = byAction.get(action);
    deterministicPick(list, perAction).forEach((item) => sample.push({
      file: item.file,
      line: item.line,
      action,
      source_kind: item.row?.input?.source_kind,
      source_id: item.row?.input?.source_id,
      input_preview: preview(item.row?.input?.current_turn?.text ?? item.row?.input?.current_query?.question ?? ""),
      memory: item.row?.output?.memory?.content ?? null,
      facts: item.row?.output?.facts ?? [],
      indexables: item.row?.output?.indexables ?? [],
      recall: item.row?.output?.recall ?? null,
      reasoning: item.row?.output?.reasoning
    }));
  }
  return sample.slice(0, sampleSize);
}

function deterministicPick(items, count) {
  if (items.length <= count) return items;
  const result = [];
  const step = items.length / count;
  for (let i = 0; i < count; i++) result.push(items[Math.floor(i * step)]);
  return result;
}

function preview(value) {
  const text = String(value).replace(/\s+/g, " ").trim();
  return text.length <= 180 ? text : `${text.slice(0, 177)}...`;
}

function issue(location, code, message) {
  return { location, code, message };
}

function normalizeText(value) {
  return String(value).toLowerCase().replace(/\s+/g, " ").trim();
}

function ratio(value, total) {
  return total > 0 ? (value ?? 0) / total : 0;
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2), "utf8");
}

function parseArgs(argv) {
  const parsed = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      parsed._.push(arg);
      continue;
    }
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) parsed[key] = "true";
    else parsed[key] = next, i++;
  }
  return parsed;
}

function listArg(parsed, key, fallback) {
  const value = parsed[key];
  if (Array.isArray(value)) return value;
  if (typeof value === "string" && value.trim()) return [value, ...fallback];
  return fallback;
}

function stringArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

function boolArg(parsed, key, fallback) {
  const value = parsed[key];
  if (value == null) return fallback;
  if (value === "true") return true;
  if (value === "false") return false;
  return Boolean(value);
}

