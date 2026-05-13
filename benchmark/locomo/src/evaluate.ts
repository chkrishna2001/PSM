import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, rankMemories, type MemoryRecord } from "psm-sdk";
import { loadSamples, parseOptions, parseTags, tagValue } from "./common.js";

export function main(argv: string[]): number {
  const options = parseOptions(argv);
  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  const records: Array<Record<string, unknown>> = [];

  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? "unknown");
    const userId = `locomo-${sampleId}`;
    const memories = store.selectMemories(userId, ["semantic", "episodic"], 10000);
    if (memories.length === 0) continue;
    for (const qa of sample.qa ?? []) {
      const evidence = (qa.evidence ?? []).map(String).filter(Boolean);
      if (evidence.length === 0) continue;
      const ranked = rankMemories(String(qa.question ?? ""), memories, options.topK);
      const selected = ranked.map(locomoDiaId).filter(Boolean);
      records.push({
        sample_id: sampleId,
        category: String(qa.category ?? "unknown"),
        question: String(qa.question ?? ""),
        answer: String(qa.answer ?? ""),
        evidence,
        selected_ids: selected,
        hit_at_1: hitAt(evidence, selected, 1),
        hit_at_k: hitAt(evidence, selected, options.topK)
      });
    }
  }

  const summary = summarize(records, options.topK);
  const output = { summary, records };
  mkdirSync(dirname(options.out), { recursive: true });
  writeFileSync(options.out, JSON.stringify(output, null, 2), "utf8");
  store.close();
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\nWrote ${options.out}\n`);
  return records.length === 0 ? 1 : 0;
}

function hitAt(evidence: string[], selected: string[], k: number): boolean {
  return evidence.some((id) => selected.slice(0, k).includes(id));
}

function locomoDiaId(memory: MemoryRecord): string {
  return tagValue(parseTags(memory.tags), "locomo_dia_id");
}

function summarize(records: Array<Record<string, unknown>>, topK: number): Record<string, unknown> {
  const denom = records.length || 1;
  return {
    questions: records.length,
    hit_at_1: records.filter((record) => record.hit_at_1 === true).length / denom,
    [`hit_at_${topK}`]: records.filter((record) => record.hit_at_k === true).length / denom
  };
}

if (process.argv[1]?.endsWith("evaluate.js")) {
  process.exitCode = main(process.argv.slice(2));
}
