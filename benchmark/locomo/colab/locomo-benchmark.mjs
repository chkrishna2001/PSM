import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  defaultEmbeddingModel,
  MemoryStore,
  NodeLlamaRuntime,
  parseStorageDecision,
  rankMemories,
  TransformersEmbeddingRuntime
} from "@psm-memory/sdk";

const command = process.argv[2] ?? "help";
const args = parseArgs(process.argv.slice(3));

if (command === "ingest") {
  process.exitCode = await ingest(args);
} else if (command === "evaluate") {
  process.exitCode = evaluate(args);
} else {
  console.log(`Usage:
  node locomo-benchmark.mjs ingest --data <locomo10.json> --db <db> --model <gguf> [--limit n]
  node locomo-benchmark.mjs evaluate --data <locomo10.json> --db <db> --out <results.json> [--top-k n]
`);
}

async function ingest(options) {
  const dataPath = stringOption(options, "data", "/content/PSM/benchmark/locomo/data/locomo10.json");
  const dbPath = stringOption(options, "db", "/content/locomo/results/locomo-psm-memory.db");
  const modelPath = stringOption(options, "model", "/content/psm-memory-cache/psm-memory-qwen-1.5b-q4_k_m.gguf");
  const limit = intOption(options, "limit", 100);
  const batchSize = intOption(options, "batch-size", 10);
  const contextSize = intOption(options, "context-size", 4096);
  const userPrefix = stringOption(options, "user-prefix", "locomo");
  const embeddingModel = stringOption(options, "embedding-model", defaultEmbeddingModel);
  const samples = loadSamples(dataPath);
  mkdirSync(dirname(dbPath), { recursive: true });

  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const runtime = new NodeLlamaRuntime({
    modelPath,
    contextSize,
    gpu: "auto",
    gpuLayers: "auto",
    log: (message) => console.error(message)
  });
  const embeddings = new TransformersEmbeddingRuntime({
    model: embeddingModel,
    cacheDir: "/content/psm-memory-cache/hf"
  });

  const stats = {
    data: dataPath,
    db: dbPath,
    model: modelPath,
    embedding_model: embeddingModel,
    limit,
    seen: 0,
    stored: 0,
    ignored: 0,
    failed: 0,
    started_at: new Date().toISOString(),
    ended_at: null,
    errors: []
  };

  try {
    for (const sample of samples) {
      const sampleId = String(sample.sample_id ?? "unknown");
      const userId = `${userPrefix}-${sampleId}`;
      for (const turn of flattenTurns(sample)) {
        if (limit > 0 && stats.seen >= limit) return finish(store, stats);
        const diaId = String(turn.dia_id ?? "");
        const source = `${sampleId}:${diaId || stats.seen}`;
        const text = `${turn.speaker ?? "speaker"}: ${turn.text ?? ""}`;
        stats.seen++;
        try {
          const raw = await runtime.generateJson(buildStoragePrompt(text), { temperature: 0, maxTokens: 128 });
          const decision = parseStorageDecision(raw, text, "store_episodic");
          const result = store.applyDecision(userId, source, decision, [
            `locomo_sample_id:${sampleId}`,
            `locomo_dia_id:${diaId}`,
            `locomo_speaker:${turn.speaker ?? ""}`
          ]);
          if (result.route === "ignore" || result.route === "recall_only") {
            stats.ignored++;
          } else {
            stats.stored++;
          }
          for (const ref of result.memory_refs) {
            const embedding = await embeddings.embed(ref.content);
            store.upsertMemoryEmbedding(ref, userId, embeddingModel, embedding);
          }
        } catch (error) {
          stats.failed++;
          const message = error instanceof Error ? error.message : String(error);
          stats.errors.push({ source, error: message });
          store.insertDecision(userId, source, "error", "error", message, JSON.stringify({ error: message }));
        }
        if (stats.seen % batchSize === 0) {
          console.log(`ingested=${stats.seen} stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}`);
        }
      }
    }
    return finish(store, stats);
  } finally {
    store.close();
  }
}

function evaluate(options) {
  const dataPath = stringOption(options, "data", "/content/PSM/benchmark/locomo/data/locomo10.json");
  const dbPath = stringOption(options, "db", "/content/locomo/results/locomo-psm-memory.db");
  const outPath = stringOption(options, "out", "/content/locomo/results/locomo-results.json");
  const topK = intOption(options, "top-k", 3);
  const userPrefix = stringOption(options, "user-prefix", "locomo");
  const samples = loadSamples(dataPath);
  const store = new MemoryStore(dbPath);
  const records = [];

  try {
    for (const sample of samples) {
      const sampleId = String(sample.sample_id ?? "unknown");
      const userId = `${userPrefix}-${sampleId}`;
      const memories = store.selectMemories(userId, ["semantic", "episodic"], 10000);
      if (memories.length === 0) continue;
      for (const qa of sample.qa ?? []) {
        const evidence = (qa.evidence ?? []).map(String).filter(Boolean);
        if (evidence.length === 0) continue;
        const ranked = rankMemories(String(qa.question ?? ""), memories, topK);
        const selectedIds = ranked.map(locomoDiaId).filter(Boolean);
        records.push({
          sample_id: sampleId,
          category: String(qa.category ?? "unknown"),
          question: String(qa.question ?? ""),
          answer: String(qa.answer ?? ""),
          evidence,
          selected_ids: selectedIds,
          hit_at_1: hitAt(evidence, selectedIds, 1),
          hit_at_k: hitAt(evidence, selectedIds, topK)
        });
      }
    }
  } finally {
    store.close();
  }

  const summary = summarize(records, topK);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify({ summary, records }, null, 2), "utf8");
  console.log(JSON.stringify(summary, null, 2));
  console.log(`Wrote ${outPath}`);
  return records.length === 0 ? 1 : 0;
}

function finish(store, stats) {
  stats.ended_at = new Date().toISOString();
  const outPath = "/content/locomo/results/ingest-summary.json";
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(stats, null, 2), "utf8");
  console.log(JSON.stringify(stats, null, 2));
  return stats.failed > 0 ? 1 : 0;
}

function loadSamples(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function flattenTurns(sample) {
  const conversation = sample.conversation ?? {};
  return Object.keys(conversation)
    .filter((key) => /^session_\d+$/.test(key))
    .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]))
    .flatMap((key) => (conversation[key] ?? []).map((turn) => ({ ...turn, session: key })));
}

function locomoDiaId(memory) {
  const tags = parseTags(memory.tags);
  const prefix = "locomo_dia_id:";
  return tags.find((tag) => tag.startsWith(prefix))?.slice(prefix.length) ?? "";
}

function parseTags(value) {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function hitAt(evidence, selected, k) {
  return evidence.some((id) => selected.slice(0, k).includes(id));
}

function summarize(records, topK) {
  const denom = records.length || 1;
  return {
    questions: records.length,
    hit_at_1: records.filter((record) => record.hit_at_1 === true).length / denom,
    [`hit_at_${topK}`]: records.filter((record) => record.hit_at_k === true).length / denom
  };
}

function buildStoragePrompt(text) {
  return `<|system|>
You are PSM, a memory-management model. Return JSON only.
Choose action: ignore, store_episodic, promote_semantic, update_existing, flag_conflict.
JSON shape: {"action":"store_episodic","memory":{"content":"...","type":"episodic","strength":0.75,"decay_rate":0.02,"emotional_weight":0.2,"confidence":0.8,"tags":[]},"reasoning":"..."}
<|user|>
Remember this conversation turn if useful:
${JSON.stringify(text)}
<|assistant|>
`;
}

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      result[key] = next;
      i++;
    } else {
      result[key] = true;
    }
  }
  return result;
}

function stringOption(options, key, fallback) {
  const value = options[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intOption(options, key, fallback) {
  const parsed = Number(options[key]);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}
