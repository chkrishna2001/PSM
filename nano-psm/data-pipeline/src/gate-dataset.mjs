#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const args = parseArgs(process.argv.slice(2));
const train = stringArg(args, "train", "nano-psm/data-pipeline/data/generated/train.jsonl");
const validation = stringArg(args, "validation", "nano-psm/data-pipeline/data/generated/validation.jsonl");
const out = stringArg(args, "out", "nano-psm/data-pipeline/reports/gate");

mkdirSync(out, { recursive: true });

const steps = [
  {
    name: "schema",
    command: ["node", "nano-psm/data-pipeline/src/validate-examples.mjs", train, validation]
  },
  {
    name: "runtime",
    command: ["node", "nano-psm/data-pipeline/src/validate-runtime-compat.mjs", train, validation]
  },
  {
    name: "quality",
    command: ["node", "nano-psm/data-pipeline/src/audit-quality.mjs", "--files", train, validation, "--out", out]
  }
];

const results = [];
for (const step of steps) {
  const [program, ...programArgs] = step.command;
  const result = spawnSync(program, programArgs, {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  results.push({
    name: step.name,
    status: result.status === 0 ? "pass" : "fail",
    exit_code: result.status
  });
  if (result.status !== 0) {
    console.error(JSON.stringify({ gate: "failed", failed_step: step.name, reports: out, results }, null, 2));
    process.exit(result.status ?? 1);
  }
}

console.log(JSON.stringify({ gate: "pass", reports: out, results }, null, 2));

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

