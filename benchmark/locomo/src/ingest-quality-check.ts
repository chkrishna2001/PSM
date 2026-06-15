import { writeFileSync } from "node:fs";
import path from "node:path";
import { MemoryStore, type MemoryRecord } from "@psm-memory/sdk";
import { flattenTurns, loadSamples, parseOptions, parseTags, tagValue } from "./common.js";

const BLEED_PATTERN = /checkpoint|powershell|gate datasets|nvidia-smi|direct probe|token budget|runpod/i;
const WRAPPER_PATTERN = /current utterance:|source id:|locomo benchmark|extraction guidance:/i;

interface QualityIssue {
  source: string;
  check: string;
  detail: string;
}

export function main(argv: string[]): number {
  const options = parseOptions(argv);
  const ingestedLimit = intOption(argv, "ingest-limit", 0);
  const out = getOption(argv, "out", options.db.replace(/\.db$/i, "-quality.json"));
  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  const issues: QualityIssue[] = [];

  const ingestedIds = collectIngestedDiaIds(samples, ingestedLimit);
  const memories = listMemories(store, samples);
  const factsCount = countFacts(store);

  for (const memory of memories) {
    const source = tagValue(parseTags(memory.tags), "locomo_dia_id") || memory.source_id || memory.id;
    const content = memory.content?.trim() ?? "";
    if (!content) {
      issues.push({ source, check: "empty_content", detail: "memory content is empty" });
      continue;
    }
    if (content.startsWith("{") || WRAPPER_PATTERN.test(content)) {
      issues.push({ source, check: "wrapper_or_json", detail: content.slice(0, 120) });
    }
    if (BLEED_PATTERN.test(content)) {
      issues.push({ source, check: "curriculum_bleed", detail: content.slice(0, 120) });
    }
    if (/^user prefers\b/i.test(content)) {
      issues.push({ source, check: "generic_user_pref", detail: content.slice(0, 120) });
    }
  }

  if (memories.length === 0) {
    issues.push({ source: "db", check: "no_memories", detail: "no episodic or semantic rows found" });
  }

  const goldChecks = [
    { diaId: "D1:3", needles: [/lgbtq/i, /support group/i] },
    { diaId: "D1:5", needles: [/transgender/i] },
    { diaId: "D1:12", needles: [/sunrise|paint/i] }
  ];
  for (const probe of goldChecks) {
    if (!ingestedIds.has(probe.diaId)) continue;
    const row = memories.find((memory) => tagValue(parseTags(memory.tags), "locomo_dia_id") === probe.diaId);
    if (!row) {
      issues.push({ source: probe.diaId, check: "missing_gold_memory", detail: "expected stored memory for ingested evidence turn" });
      continue;
    }
    const text = `${row.content ?? ""} ${row.tags ?? ""}`;
    if (!probe.needles.some((pattern) => pattern.test(text))) {
      issues.push({ source: probe.diaId, check: "gold_fact_missing", detail: `content missing expected facts for ${probe.diaId}` });
    }
  }

  const summary = {
    db: options.db,
    ingested_turns: ingestedIds.size,
    memory_rows: memories.length,
    memory_facts: factsCount,
    issues: issues.length,
    passed: issues.length === 0
  };
  writeFileSync(out, JSON.stringify({ summary, issues }, null, 2), "utf8");
  store.close();
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  for (const issue of issues.slice(0, 20)) {
    process.stdout.write(`ISSUE ${issue.source} ${issue.check}: ${issue.detail}\n`);
  }
  return issues.length > 0 ? 1 : 0;
}

function collectIngestedDiaIds(samples: ReturnType<typeof loadSamples>, limit: number): Set<string> {
  const ids = new Set<string>();
  let seen = 0;
  for (const sample of samples) {
    for (const turn of flattenTurns(sample)) {
      if (limit > 0 && seen >= limit) return ids;
      const diaId = String(turn.dia_id ?? "");
      if (diaId) ids.add(diaId);
      seen++;
    }
  }
  return ids;
}

function listMemories(store: MemoryStore, samples: ReturnType<typeof loadSamples>): MemoryRecord[] {
  const memories: MemoryRecord[] = [];
  for (const sample of samples) {
    const userId = `locomo-${String(sample.sample_id ?? "unknown")}`;
    memories.push(...store.selectMemories(userId, ["semantic", "episodic"], 10000));
  }
  return memories;
}

function countFacts(store: MemoryStore): number {
  const db = (store as unknown as { db?: { prepare: (sql: string) => { get: () => { count?: number } } } }).db;
  if (!db) return 0;
  try {
    return Number(db.prepare("SELECT COUNT(*) AS count FROM memory_facts").get()?.count ?? 0);
  } catch {
    return 0;
  }
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

function intOption(argv: string[], key: string, fallback: number): number {
  const index = argv.indexOf(`--${key}`);
  if (index < 0) return fallback;
  const value = Number(argv[index + 1]);
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

if (process.argv[1]?.endsWith("ingest-quality-check.js")) {
  process.exitCode = main(process.argv.slice(2));
}
