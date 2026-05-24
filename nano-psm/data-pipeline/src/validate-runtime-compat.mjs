#!/usr/bin/env node
import { readJsonl } from "./lib/jsonl.mjs";
import { validateRuntimeCompatibility } from "./lib/runtime-compat.mjs";

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error("Usage: node nano-psm/data-pipeline/src/validate-runtime-compat.mjs <file.jsonl> [...]");
  process.exit(2);
}

let checked = 0;
let failures = 0;
const actions = {};

for (const file of files) {
  const rows = readJsonl(file);
  rows.forEach((row, index) => {
    checked++;
    const action = row?.output?.action ?? "unknown";
    actions[action] = (actions[action] ?? 0) + 1;
    const errors = validateRuntimeCompatibility(row);
    for (const error of errors) {
      failures++;
      console.error(`${file}:${index + 1}: ${error}`);
    }
  });
}

console.log(JSON.stringify({ checked, failures, actions }, null, 2));
if (failures > 0) process.exit(1);

