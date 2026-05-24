#!/usr/bin/env node
import { readFileSync } from "node:fs";

const allowedActions = new Set(["ignore", "store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store", "recall_context"]);
const allowedMemoryTypes = new Set(["episodic", "semantic"]);
const allowedTopLevelOutput = new Set(["action", "memory", "facts", "indexables", "updates", "conflicts", "recall", "reasoning"]);
const files = process.argv.slice(2);

if (files.length === 0) {
  console.error("Usage: node nano-psm/data-pipeline/src/validate-examples.mjs <file.jsonl> [...]");
  process.exit(2);
}

let checked = 0;
let failures = 0;
const stats = {};
for (const file of files) {
  const lines = readFileSync(file, "utf8").split(/\r?\n/).filter((line) => line.trim());
  for (let i = 0; i < lines.length; i++) {
    checked++;
    let row;
    try {
      row = JSON.parse(lines[i]);
    } catch (error) {
      report(file, i + 1, `invalid JSONL row: ${error.message}`);
      continue;
    }
    const errors = validateExample(row);
    const action = row?.output?.action ?? "unknown";
    stats[action] = (stats[action] ?? 0) + 1;
    if (errors.length > 0) {
      for (const error of errors) report(file, i + 1, error);
    }
  }
}

console.log(JSON.stringify({ checked, failures, actions: stats }, null, 2));
if (failures > 0) process.exit(1);

function validateExample(row) {
  const errors = [];
  if (!isRecord(row)) return ["row must be an object"];
  if (typeof row.instruction !== "string" || !row.instruction.trim()) errors.push("instruction is required");
  if (!isRecord(row.input)) errors.push("input object is required");
  if (!isRecord(row.output)) {
    errors.push("output object is required");
    return errors;
  }

  for (const key of Object.keys(row.output)) {
    if (!allowedTopLevelOutput.has(key)) errors.push(`unexpected output key: ${key}`);
  }
  if ("operation" in row.output) errors.push("legacy output key operation is forbidden");
  if ("assistant_response" in row.output) errors.push("legacy output key assistant_response is forbidden");

  if (!allowedActions.has(row.output.action)) errors.push(`invalid action: ${row.output.action}`);
  if (!Array.isArray(row.output.facts)) errors.push("facts must be an array");
  if (!Array.isArray(row.output.indexables)) errors.push("indexables must be an array");
  if (!Array.isArray(row.output.updates)) errors.push("updates must be an array");
  if (!Array.isArray(row.output.conflicts)) errors.push("conflicts must be an array");
  if (typeof row.output.reasoning !== "string") errors.push("reasoning must be a string");

  const speaker = row.input?.current_turn?.speaker;
  const memory = row.output.memory;
  if ((row.output.action === "ignore" || row.output.action === "recall_context") && memory !== null) errors.push(`${row.output.action} action must use memory:null`);
  if (row.output.action !== "ignore" && row.output.action !== "recall_context") {
    if (!isRecord(memory)) errors.push("non-ignore action requires memory object");
    else validateMemory(memory, speaker, errors);
  }
  if (row.output.action === "recall_context") validateRecall(row.output.recall, errors);

  if (Array.isArray(row.output.facts)) {
    for (const [index, fact] of row.output.facts.entries()) validateFact(fact, index, errors);
  }
  if (Array.isArray(row.output.indexables)) {
    for (const [index, indexable] of row.output.indexables.entries()) validateIndexable(indexable, index, errors);
  }
  return errors;
}

function validateMemory(memory, speaker, errors) {
  if (typeof memory.content !== "string" || !memory.content.trim()) errors.push("memory.content is required");
  if (!allowedMemoryTypes.has(memory.type)) errors.push(`invalid memory.type: ${memory.type}`);
  for (const key of ["strength", "decay_rate", "emotional_weight", "confidence"]) {
    if (typeof memory[key] !== "number" || memory[key] < 0 || memory[key] > 1) errors.push(`memory.${key} must be 0..1`);
  }
  if (!Array.isArray(memory.tags)) errors.push("memory.tags must be an array");
  const content = String(memory.content ?? "");
  if (/^\s*User\b/.test(content) && typeof speaker === "string" && speaker && speaker !== "User") {
    errors.push("memory.content uses generic User even though speaker is known");
  }
  if (content.includes("Current utterance:") || content.includes("Previous context:")) {
    errors.push("memory.content leaks wrapper text");
  }
}

function validateFact(fact, index, errors) {
  if (!isRecord(fact)) {
    errors.push(`facts[${index}] must be an object`);
    return;
  }
  for (const key of ["subject", "predicate", "value", "evidence_text"]) {
    if (typeof fact[key] !== "string" || !fact[key].trim()) errors.push(`facts[${index}].${key} is required`);
  }
  if (!/^[a-z][a-z0-9_]*$/.test(String(fact.predicate ?? ""))) errors.push(`facts[${index}].predicate must be snake_case`);
  if (fact.inference_kind !== "explicit") errors.push(`facts[${index}].inference_kind must be explicit`);
  if (typeof fact.confidence !== "number" || fact.confidence < 0 || fact.confidence > 1) errors.push(`facts[${index}].confidence must be 0..1`);
}

function validateIndexable(indexable, index, errors) {
  if (!isRecord(indexable)) {
    errors.push(`indexables[${index}] must be an object`);
    return;
  }
  for (const key of ["kind", "key", "target_type", "reconstructive_hint", "evidence_text"]) {
    if (typeof indexable[key] !== "string" || !indexable[key].trim()) errors.push(`indexables[${index}].${key} is required`);
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(String(indexable.key ?? ""))) errors.push(`indexables[${index}].key must be lowercase hyphenated tokens`);
  if (typeof indexable.salience !== "number" || indexable.salience < 0 || indexable.salience > 1) errors.push(`indexables[${index}].salience must be 0..1`);
  if (!Array.isArray(indexable.tags)) errors.push(`indexables[${index}].tags must be an array`);
}

function validateRecall(recall, errors) {
  if (!isRecord(recall)) {
    errors.push("recall_context action requires recall object");
    return;
  }
  if (typeof recall.query_intent !== "string" || !recall.query_intent.trim()) errors.push("recall.query_intent is required");
  if (!Array.isArray(recall.selected_memory_ids)) errors.push("recall.selected_memory_ids must be an array");
  if (!Array.isArray(recall.selected_indexable_keys)) errors.push("recall.selected_indexable_keys must be an array");
  if (typeof recall.max_items !== "number" || recall.max_items < 0) errors.push("recall.max_items must be a non-negative number");
  if (typeof recall.reasoning !== "string" || !recall.reasoning.trim()) errors.push("recall.reasoning is required");
}

function report(file, line, message) {
  failures++;
  console.error(`${file}:${line}: ${message}`);
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
