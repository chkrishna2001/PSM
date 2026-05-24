import { existsSync, readFileSync } from "node:fs";
import { basename } from "node:path";
import Database from "better-sqlite3";
import {
  buildIndexables,
  cleanText,
  createRecallExample,
  createRememberExample,
  parseJsonArray,
  storeMemoryOutput
} from "../lib/psm-example.mjs";

export function generateExamples(options = {}) {
  const dbPath = options.dbPath ?? "user_memory.db";
  const docs = options.docs ?? ["docs/indexables-conv.txt"];
  const maxDbRows = Number.isInteger(options.maxDbRows) ? options.maxDbRows : 250;
  const maxDocExamples = Number.isInteger(options.maxDocExamples) ? options.maxDocExamples : 40;
  const maxMemoryChars = Number.isInteger(options.maxMemoryChars) ? options.maxMemoryChars : 260;
  const examples = [];

  if (existsSync(dbPath)) {
    examples.push(...generateDbMemoryExamples(dbPath, maxDbRows, maxMemoryChars));
  }

  for (const docPath of docs) {
    if (existsSync(docPath)) {
      examples.push(...generateDocConceptExamples(docPath, maxDocExamples));
    }
  }

  examples.push(...generateRecallExamples(examples));
  return examples.map((example, index) => ({ ...example, id: example.id ?? `local-psm-${index + 1}` }));
}

function generateDbMemoryExamples(dbPath, maxRows, maxMemoryChars) {
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  try {
    const episodic = tableExists(db, "episodic") ? selectMemoryRows(db, "episodic", maxRows) : [];
    const semantic = tableExists(db, "semantic") ? selectMemoryRows(db, "semantic", maxRows) : [];
    const facts = tableExists(db, "memory_facts") ? selectFacts(db, maxRows * 4) : [];
    const factsBySource = groupFactsBySource(facts);

    return [...episodic, ...semantic]
      .filter((row) => isUsableMemoryRow(row, maxMemoryChars))
      .slice(0, maxRows)
      .map((row) => {
        const type = row.memory_table;
        const action = type === "semantic" ? "promote_semantic" : "store_episodic";
        const sourceKey = `${type}:${row.id}`;
        const linkedFacts = factsBySource.get(sourceKey) ?? [];
        const tags = parseJsonArray(row.tags);
        return createRememberExample(`local-psm-db-${type}-${row.id}`, {
          source_kind: "local_psm_db",
          source_id: sourceKey,
          current_turn: {
            speaker: speakerForMemory(row.content),
            text: row.content,
            timestamp: row.source_timestamp || row.created_at || ""
          },
          memory_store: []
        }, storeMemoryOutput(action, type, {
          content: row.content,
          strength: numberOr(row.strength, type === "semantic" ? 0.85 : 0.78),
          decay_rate: numberOr(row.decay_rate, type === "semantic" ? 0.02 : 0.04),
          emotional_weight: numberOr(row.emotional_weight, 0.35),
          confidence: numberOr(row.confidence, 0.86),
          tags: [...tags, "local_psm"],
          facts: linkedFacts.map(factToTrainingFact),
          target_id: sourceKey,
          temporal_expression: row.temporal_expression || undefined,
          resolved_time: row.resolved_time || undefined,
          resolved_time_confidence: row.resolved_time_confidence ?? undefined,
          reasoning: "Existing concise PSM memory row converted into a real local training example."
        }));
      });
  } finally {
    db.close();
  }
}

function generateDocConceptExamples(docPath, maxExamples) {
  const text = readFileSync(docPath, "utf8");
  const chunks = text
    .split(/\r?\n\s*\r?\n/)
    .map(cleanText)
    .filter((chunk) => chunk.length >= 80)
    .filter((chunk) => /indexable|mnemonic|earworm|recall|cue|compress|memory/i.test(chunk))
    .slice(0, maxExamples);

  return chunks.map((chunk, index) => {
    const content = conceptMemoryForChunk(chunk);
    const keySeed = mnemonicKeyForConcept(chunk);
    return createRememberExample(`local-psm-doc-${basename(docPath)}-${index + 1}`, {
      source_kind: "local_psm_doc",
      source_id: `${docPath}#chunk-${index + 1}`,
      current_turn: {
        speaker: "User",
        text: chunk,
        timestamp: ""
      },
      memory_store: []
    }, storeMemoryOutput("promote_semantic", "semantic", {
      content,
      strength: 0.9,
      decay_rate: 0.01,
      emotional_weight: 0.55,
      confidence: 0.88,
      tags: ["local_psm", "indexables", "mnemonic_recall"],
      indexables: [{
        kind: "mnemonic",
        key: keySeed,
        target_type: "semantic",
        target_id: `${docPath}#chunk-${index + 1}`,
        salience: 0.92,
        reconstructive_hint: content,
        evidence_text: chunk.slice(0, 240),
        tags: ["indexables", "mnemonic_recall"]
      }],
      facts: [{
        subject: "PSM indexables",
        predicate: "support",
        value: "compressed reconstructive recall cues",
        confidence: 0.86,
        inference_kind: "explicit",
        evidence_text: chunk.slice(0, 240)
      }],
      reasoning: "Curated indexables concept from local PSM design discussion."
    }));
  });
}

function generateRecallExamples(memoryExamples) {
  const stored = memoryExamples
    .filter((example) => example.output?.memory)
    .slice(0, 25)
    .map((example) => {
      const memory = example.output.memory;
      return {
        id: example.input.source_id,
        source_id: example.input.source_id,
        content: memory.content,
        tags: memory.tags,
        indexables: example.output.indexables
      };
    });

  const indexableRows = stored.filter((row) => row.indexables?.some((item) => /indexable|mnemonic|recall|cue/.test(item.key)));
  if (indexableRows.length === 0) return [];

  const selected = indexableRows.slice(0, 5);
  return [
    createRecallExample("local-psm-recall-indexables-1", {
      source_kind: "local_psm_recall",
      current_query: {
        question: "Which memories explain the PSM indexables and mnemonic recall direction?"
      },
      memory_store: stored
    }, {
      recall: {
        query_intent: "memory_recall",
        selected_memory_ids: selected.map((row) => row.id),
        selected_indexable_keys: selected.flatMap((row) => row.indexables.map((indexable) => indexable.key)).slice(0, 8),
        max_items: selected.length,
        reasoning: "Selected local PSM memories whose mnemonic/indexable cues explain compressed recall."
      },
      reasoning: "Recall should route through real local indexable memories and their compact cues."
    })
  ];
}

function selectMemoryRows(db, table, limit) {
  return db.prepare(`
SELECT *, '${table}' as memory_table
FROM ${table}
ORDER BY created_at DESC
LIMIT ?
`).all(limit);
}

function selectFacts(db, limit) {
  return db.prepare(`
SELECT *
FROM memory_facts
ORDER BY created_at DESC
LIMIT ?
`).all(limit);
}

function groupFactsBySource(facts) {
  const map = new Map();
  for (const fact of facts) {
    if (!fact.source_memory_table || !fact.source_memory_id) continue;
    const key = `${fact.source_memory_table}:${fact.source_memory_id}`;
    const list = map.get(key) ?? [];
    list.push(fact);
    map.set(key, list);
  }
  return map;
}

function tableExists(db, table) {
  const row = db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(table);
  return Boolean(row);
}

function factToTrainingFact(fact) {
  return {
    subject: fact.subject,
    predicate: fact.predicate,
    value: fact.value_text ?? fact.object,
    confidence: numberOr(fact.confidence, 0.8),
    inference_kind: "explicit",
    evidence_text: fact.evidence_text ?? fact.value_text ?? fact.object,
    temporal_expression: fact.temporal_expression || undefined,
    resolved_time: fact.resolved_time || undefined,
    resolved_time_confidence: fact.resolved_time_confidence ?? undefined
  };
}

function conceptMemoryForChunk(chunk) {
  const lower = chunk.toLowerCase();
  if (lower.includes("earworm")) {
    return "PSM should use earworm-style mnemonic cues as compact handles that can reactivate larger memories.";
  }
  if (lower.includes("indexable")) {
    return "PSM indexables are compressed reconstructive recall handles, not simple tags.";
  }
  if (lower.includes("reconstruct")) {
    return "PSM recall should reconstruct relevant memory context from compact cues while staying evidence-grounded.";
  }
  return "PSM should train compact mnemonic cues that improve memory recall quality and reduce context load.";
}

function mnemonicKeyForConcept(chunk) {
  const lower = chunk.toLowerCase();
  if (lower.includes("earworm")) return "earworm-sparse-recall";
  if (lower.includes("indexable")) return "compressed-recall-handles";
  if (lower.includes("mnemonic")) return "mnemonic-memory-cues";
  if (lower.includes("reconstruct")) return "reconstructive-recall-cues";
  return buildIndexables({ content: chunk, tags: ["indexables"], target_type: "semantic" })[0].key;
}

function numberOr(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function speakerForMemory(content) {
  return /^User\b/.test(cleanText(content)) ? "User" : "PSM";
}

function isUsableMemoryRow(row, maxMemoryChars) {
  const content = cleanText(row.content);
  if (content.length < 16 || content.length > maxMemoryChars) return false;
  if (/Current utterance:|Previous context:|Return JSON|target schema/i.test(content)) return false;
  if (/^(ok|thanks|thank you|hello|hi)[.! ]*$/i.test(content)) return false;
  return true;
}
