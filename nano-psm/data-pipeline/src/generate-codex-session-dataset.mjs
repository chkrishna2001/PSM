#!/usr/bin/env node
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import {
  buildIndexables,
  createRecallExample,
  createRememberExample,
  ignoreOutput,
  storeMemoryOutput
} from "./lib/psm-example.mjs";
import { writeJsonl } from "./lib/jsonl.mjs";

const args = parseArgs(process.argv.slice(2));
const sessionsDir = stringArg(args, "sessions-dir", "C:/Users/chkri/.codex/sessions/2026");
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/codex-sessions-2026");
const trainLimit = intArg(args, "train-limit", 1200);
const validationLimit = intArg(args, "validation-limit", 200);
const maxRows = trainLimit + validationLimit;

if (!existsSync(sessionsDir)) throw new Error(`Missing sessions directory: ${sessionsDir}`);

const messages = loadSessionMessages(sessionsDir);
const candidates = [];
const storedMemories = [];

for (const message of messages) {
  const row = classifyMessage(message, candidates.length);
  if (!row) continue;
  candidates.push(row);
  if (row.output?.memory) storedMemories.push(memoryStoreItem(row));
  if (candidates.length >= maxRows) break;
}

const recallRows = buildRecallRows(storedMemories, Math.min(180, Math.floor(maxRows * 0.14)));
const allRows = balancedRows([...candidates, ...recallRows], maxRows);
const { train, validation } = splitRows(allRows, validationLimit);

mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), allRows);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "codex_sessions_2026",
  source_dir: sessionsDir,
  total_messages_scanned: messages.length,
  total_examples: allRows.length,
  train_examples: train.length,
  validation_examples: validation.length,
  action_mix: countBy(allRows, (row) => row.output?.action ?? "unknown"),
  source_mix: countBy(allRows, (row) => row.input?.source_kind ?? "unknown"),
  notes: [
    "Raw session metadata, system/developer instructions, tool payloads, secrets, and long outputs are not emitted.",
    "Examples are heuristic labels from real Codex project sessions and must pass gate-dataset before training.",
    "Recall rows are grounded in stored examples generated from the same sanitized session evidence."
  ]
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  scanned_messages: messages.length,
  train: train.length,
  validation: validation.length,
  action_mix: countBy(allRows, (row) => row.output?.action ?? "unknown"),
  source_mix: countBy(allRows, (row) => row.input?.source_kind ?? "unknown")
}, null, 2));

function loadSessionMessages(root) {
  const files = listJsonl(root);
  const rows = [];
  for (const file of files) {
    const sessionId = basename(file, ".jsonl");
    const lines = readFileSync(file, "utf8").split(/\r?\n/).filter((line) => line.trim());
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
      let row;
      try {
        row = JSON.parse(lines[lineIndex]);
      } catch {
        continue;
      }
      const payload = row.payload ?? {};
      const payloadType = payload.type;
      if (!["user_message", "agent_message", "message", "task_complete"].includes(payloadType)) continue;
      const role = roleFor(payloadType, payload);
      const text = sanitizeText(extractText(payload));
      if (!isUsableText(text)) continue;
      rows.push({
        session_id: sessionId,
        source_id: `${sessionId}:${lineIndex + 1}`,
        timestamp: row.timestamp,
        role,
        text
      });
    }
  }
  const seen = new Set();
  return rows
    .sort((left, right) => String(left.timestamp).localeCompare(String(right.timestamp)))
    .filter((row) => {
      const key = `${row.role}:${row.text}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function classifyMessage(message, index) {
  const text = trimForMemory(message.text);
  const lower = text.toLowerCase();
  const sourceKind = "codex_session";
  const input = {
    source_kind: sourceKind,
    source_id: message.source_id,
    session_id: message.session_id,
    current_turn: {
      speaker: message.role === "assistant" ? "Assistant" : "User",
      text,
      timestamp: message.timestamp
    },
    memory_store: []
  };

  if (isTransient(lower)) {
    return createRememberExample(`codex-ignore-${index}`, input, ignoreOutput("Transient chat/control/tool text without durable PSM memory value."));
  }

  if (isCorrection(lower)) {
    const tags = ["codex_session", "correction", "training_feedback", ...topicTags(lower)];
    const facts = [fact("Codex session feedback", "captured_correction", text, text, 0.88)];
    const memory = storeMemoryOutput("flag_and_store", "episodic", {
      content: contentSentence("A Codex session captured a training or workflow correction", text),
      tags,
      strength: 0.82,
      decay_rate: 0.05,
      emotional_weight: emotionalWeight(lower),
      confidence: 0.9,
      facts,
      indexables: evidenceIndexables(text, tags, facts, "episodic"),
      conflicts: [{ target_id: "prior_training_assumption", reason: "User correction or failure report should not be overwritten silently." }],
      reasoning: "Explicit correction or failure report from the session should be stored and routed as a conflict/update signal."
    });
    return createRememberExample(`codex-correction-${index}`, input, memory);
  }

  if (isSemanticRule(lower)) {
    const tags = ["codex_session", "project_rule", ...topicTags(lower)];
    const facts = [fact("Codex/PSM project rule", "stated_rule", text, text, 0.9)];
    const memory = storeMemoryOutput("promote_semantic", "semantic", {
      content: contentSentence("Codex/PSM project rule", text),
      tags,
      strength: 0.9,
      decay_rate: 0.015,
      emotional_weight: emotionalWeight(lower),
      confidence: 0.92,
      facts,
      indexables: evidenceIndexables(text, tags, facts, "semantic"),
      reasoning: "The session states a durable workflow rule or preference that should remain available across future PSM work."
    });
    return createRememberExample(`codex-semantic-${index}`, input, memory);
  }

  if (isMilestone(lower)) {
    const tags = ["codex_session", "training_milestone", ...topicTags(lower)];
    const facts = [fact("PSM training session", "recorded_milestone", text, text, 0.86)];
    const memory = storeMemoryOutput("store_episodic", "episodic", {
      content: contentSentence("A Codex session recorded a PSM training milestone", text),
      tags,
      strength: 0.8,
      decay_rate: 0.045,
      emotional_weight: emotionalWeight(lower),
      confidence: 0.88,
      facts,
      indexables: evidenceIndexables(text, tags, facts, "episodic"),
      reasoning: "Concrete session event or artifact status is useful episodic context but should decay faster than a project rule."
    });
    return createRememberExample(`codex-episodic-${index}`, input, memory);
  }

  return null;
}

function buildRecallRows(memories, limit) {
  const topics = [
    { id: "hf-artifacts", query: "Which Hugging Face datasets or checkpoints should be used for Nano PSM training?", match: /huggingface|hf|checkpoint|dataset|upload/ },
    { id: "retention-quality", query: "What retention decay or validation issues were found in recent Nano PSM work?", match: /retention|decay|validation|mae|forgetting/ },
    { id: "project-rules", query: "What durable project rules or training preferences did the user state?", match: /rule|should|must|prefer|best|quality/ },
    { id: "failures", query: "What failures or corrections should guide the next PSM training run?", match: /failed|error|empty|404|wrong|disconnect|correction/ }
  ];
  const rows = [];
  for (const topic of topics) {
    const matches = deterministicPick(
      memories.filter((memory) => topic.match.test(`${memory.content} ${(memory.tags ?? []).join(" ")}`.toLowerCase())),
      Math.min(120, memories.length),
      topic.id
    );
    for (let offset = 0; offset < matches.length; offset += 3) {
      const selected = matches.slice(offset, offset + 3);
      const selectedIds = new Set(selected.map((item) => item.id));
      const distractors = deterministicPick(
        memories.filter((memory) => !selectedIds.has(memory.id)),
        3,
        `${topic.id}:${offset}`
      );
      if (selected.length === 0 || selected.length + distractors.length < 2) continue;
      const memoryStore = [...selected, ...distractors];
      rows.push(createRecallExample(`codex-recall-${topic.id}-${offset}`, {
        source_kind: "codex_session",
        source_id: `codex-recall-${topic.id}-${offset}`,
        current_query: {
          question: topic.query,
          timestamp: "2026-05-30T00:00:00Z"
        },
        memory_store: memoryStore
      }, {
        recall: {
          query_intent: "codex_session_project_recall",
          selected_memory_ids: selected.map((item) => item.id),
          selected_indexable_keys: selected.flatMap((item) => item.indexables.map((idx) => idx.key)).slice(0, 8),
          max_items: selected.length,
          reasoning: "Selected only sanitized Codex session memories matching the query topic."
        },
        reasoning: "Recall should activate grounded project/session memories instead of answering from general knowledge."
      }));
      if (rows.length >= limit) return rows;
    }
  }
  return rows;
}

function memoryStoreItem(row) {
  const memory = row.output.memory;
  const id = `mem-${row.id}`;
  return {
    id,
    content: memory.content,
    type: memory.type,
    tags: memory.tags,
    strength: memory.strength,
    decay_rate: memory.decay_rate,
    emotional_weight: memory.emotional_weight,
    confidence: memory.confidence,
    indexables: buildIndexables({
      content: memory.content,
      tags: memory.tags,
      facts: row.output.facts,
      target_type: memory.type,
      target_id: id
    })
  };
}

function splitRows(rows, validationCount) {
  const shuffled = deterministicPick(rows, rows.length, "codex-sessions-split");
  const validation = shuffled.slice(0, Math.min(validationCount, Math.floor(rows.length * 0.2)));
  const validationIds = new Set(validation.map((row) => row.id));
  const train = shuffled.filter((row) => !validationIds.has(row.id));
  return { train, validation };
}

function balancedRows(rows, limit) {
  const groups = new Map();
  for (const row of rows) {
    const action = row.output?.action ?? "unknown";
    const group = groups.get(action) ?? [];
    group.push(row);
    groups.set(action, group);
  }
  const targets = {
    promote_semantic: Math.floor(limit * 0.32),
    store_episodic: Math.floor(limit * 0.28),
    flag_and_store: Math.floor(limit * 0.14),
    recall_context: Math.floor(limit * 0.12),
    ignore: Math.floor(limit * 0.14)
  };
  const result = [];
  for (const [action, target] of Object.entries(targets)) {
    result.push(...deterministicPick(groups.get(action) ?? [], target, action));
  }
  return deterministicPick(result, Math.min(limit, result.length), "codex-sessions-final")
    .sort((left, right) => String(left.id).localeCompare(String(right.id)));
}

function extractText(payload) {
  if (typeof payload.message === "string") return payload.message;
  if (typeof payload.summary === "string") return payload.summary;
  if (Array.isArray(payload.content)) {
    return payload.content
      .map((item) => typeof item?.text === "string" ? item.text : "")
      .filter(Boolean)
      .join("\n");
  }
  if (payload.content && typeof payload.content === "object") return JSON.stringify(payload.content);
  return "";
}

function sanitizeText(value) {
  return String(value)
    .replace(/hf_[A-Za-z0-9_=-]{16,}/g, "[REDACTED_HF_TOKEN]")
    .replace(/sk-[A-Za-z0-9_-]{8,}/gi, "[REDACTED_API_KEY]")
    .replace(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, "[REDACTED_EMAIL]")
    .replace(/C:\\Users\\chkri\\[^ \n\r\t"'`]+/gi, "[LOCAL_PATH]")
    .replace(/\/content\/[^ \n\r\t"'`]+/gi, "[COLAB_PATH]")
    .replace(/\s+/g, " ")
    .trim();
}

function trimForMemory(text) {
  const trimmed = text.replace(/\s+/g, " ").trim();
  return trimmed.length <= 420 ? trimmed : `${trimmed.slice(0, 417).trim()}...`;
}

function isUsableText(text) {
  if (text.length < 16 || text.length > 5000) return false;
  if (/^<permissions instructions>|^<environment_context>|# Tools|base_instructions|developer instructions/i.test(text)) return false;
  if (/^\{ ?"status": ?"in_progress"/i.test(text)) return false;
  return true;
}

function isTransient(lower) {
  return /^(okay|ok|yes|no|proceed|thanks|thank you|what's going on|whats going on)[.! ]*$/.test(lower)
    || /^i('|’)m checking|^i found|^next i('|’)m|^i('|’)ll/.test(lower)
    || /token_count|rate_limits|sandboxing defines/.test(lower);
}

function isCorrection(lower) {
  return /wrong|strange|failed|failure|error|empty result|404|not found|disconnect|forgot|forgetting|regression|miss another|didn'?t upload|no error/.test(lower);
}

function isSemanticRule(lower) {
  return /should|must|need to|make sure|best practice|prefer|always|never|do not|don't|quality|keep training|full dataset|everything should/.test(lower);
}

function isMilestone(lower) {
  return /psm|nano|retention|decay|checkpoint|validation|dataset|colab|huggingface| hf |training|mae|accuracy|uploaded|generated|notebook|duckdb/.test(lower);
}

function topicTags(lower) {
  const tags = [];
  if (/huggingface| hf |upload/.test(lower)) tags.push("huggingface");
  if (/checkpoint/.test(lower)) tags.push("checkpoint");
  if (/dataset|data/.test(lower)) tags.push("dataset");
  if (/retention|decay/.test(lower)) tags.push("retention_decay");
  if (/validation|mae|accuracy/.test(lower)) tags.push("validation");
  if (/colab|notebook/.test(lower)) tags.push("colab");
  if (/duckdb/.test(lower)) tags.push("duckdb");
  return tags;
}

function contentSentence(prefix, text) {
  const budget = Math.max(80, 252 - prefix.length);
  const cleaned = trimForMemory(text).replace(/^["']|["']$/g, "").slice(0, budget).trim();
  return `${prefix}: ${cleaned}`;
}

function evidenceIndexables(text, tags, facts, targetType) {
  return buildIndexables({
    content: trimForMemory(text),
    tags: topicTags(text.toLowerCase()).length > 0 ? topicTags(text.toLowerCase()) : tags.filter((tag) => !["codex_session", "project_rule", "training_milestone", "correction"].includes(tag)),
    facts,
    target_type: targetType
  });
}

function fact(subject, predicate, value, evidenceText, confidence) {
  return {
    subject,
    predicate,
    value: trimForMemory(value),
    confidence,
    evidence_text: trimForMemory(evidenceText)
  };
}

function emotionalWeight(lower) {
  if (/urgent|today|failed|error|wrong|404|disconnect|quality|best/.test(lower)) return 0.58;
  if (/prefer|should|must|need/.test(lower)) return 0.42;
  return 0.28;
}

function roleFor(payloadType, payload) {
  if (payloadType === "agent_message" || payload.role === "assistant") return "assistant";
  return "user";
}

function listJsonl(root) {
  const result = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) result.push(...listJsonl(path));
    else if (entry.isFile() && entry.name.endsWith(".jsonl")) result.push(path);
  }
  return result.sort();
}

function deterministicPick(rows, limit, salt) {
  if (rows.length <= limit) return [...rows];
  return rows
    .map((row, index) => ({ row, key: hash(`${salt}:${row.id ?? row.source_id ?? ""}:${index}`) }))
    .sort((left, right) => left.key - right.key)
    .slice(0, limit)
    .map((item) => item.row);
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
