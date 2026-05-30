#!/usr/bin/env node
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { buildIndexables, createRememberExample, ignoreOutput, instruction, normalizeTrainingExample } from "./lib/psm-example.mjs";
import { writeJsonl } from "./lib/jsonl.mjs";

const defaultModel = "nvidia/nemotron-3-super-120b-a12b:free";
const args = parseArgs(process.argv.slice(2));
const sessionsDir = stringArg(args, "sessions-dir", "C:/Users/chkri/.codex/sessions/2026");
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/codex-sessions-nemotron-pilot");
const limit = intArg(args, "limit", 20);
const model = stringArg(args, "model", process.env.CODEX_SESSION_LABEL_MODEL || process.env.LOCOMO_ANSWER_MODEL || defaultModel);
const baseUrl = stringArg(args, "base-url", process.env.OPENROUTER_BASE_URL || process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1");
const apiKey = stringArg(args, "api-key", process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || "");
const requestDelayMs = intArg(args, "request-delay-ms", 1600);
const requestMaxRetries = intArg(args, "request-max-retries", 5);
const requestTimeoutMs = intArg(args, "request-timeout-ms", 45000);

if (!existsSync(sessionsDir)) throw new Error(`Missing sessions directory: ${sessionsDir}`);
if (!apiKey) throw new Error("OPENROUTER_API_KEY is required, or pass --api-key.");

const candidates = selectCandidates(loadMessages(sessionsDir), limit);
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, "candidates.json"), JSON.stringify(candidates, null, 2), "utf8");

const rows = [];
const raw = [];
for (const [index, candidate] of candidates.entries()) {
  const response = await labelCandidate(candidate);
  const parsed = parseTeacherJson(response);
  const row = teacherOutputToRow(candidate, parsed, index);
  rows.push(row);
  raw.push({
    id: row.id,
    source_id: candidate.source_id,
    model,
    raw_response: response,
    parsed_output: parsed,
    row
  });
  writeJsonl(join(outDir, "all.jsonl"), rows);
  writeFileSync(join(outDir, "raw-responses.json"), JSON.stringify(raw, null, 2), "utf8");
  process.stdout.write(JSON.stringify({ labeled: rows.length, id: row.id, action: row.output.action }) + "\n");
}

const validationCount = Math.min(4, Math.max(1, Math.floor(rows.length * 0.2)));
const validation = rows.slice(0, validationCount);
const train = rows.slice(validationCount);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "codex_sessions_nemotron_pilot",
  source_dir: sessionsDir,
  model,
  total_candidates: candidates.length,
  total_examples: rows.length,
  train_examples: train.length,
  validation_examples: validation.length,
  action_mix: countBy(rows, (row) => row.output.action),
  notes: [
    "Pilot dataset for evaluating Nemotron as a teacher labeler over sanitized Codex session windows.",
    "Only write-time actions are requested in this pilot; recall_context should be generated after reviewing storage-label quality.",
    "Every row should pass gate-dataset before it is mixed into a training corpus."
  ]
}, null, 2), "utf8");

console.log(JSON.stringify({
  status: "complete",
  out_dir: outDir,
  rows: rows.length,
  action_mix: countBy(rows, (row) => row.output.action)
}, null, 2));
process.exit(0);

async function labelCandidate(candidate) {
  return chatCompletion([
    {
      role: "system",
      content: [
        "You are a strict PSM training-data labeler.",
        "Return one complete JSON object only. No markdown. No commentary. No analysis. Start with { and end with }.",
        "Your task is to label one sanitized Codex session window for a compact memory-management model.",
        "Allowed actions: ignore, store_episodic, promote_semantic, update_existing, flag_conflict, flag_and_store.",
        "Use ignore for transient progress updates, commands, raw logs, greetings, or messages without durable memory value.",
        "Use promote_semantic for durable project rules, user preferences, product constraints, or reusable training policy.",
        "Use store_episodic for concrete session milestones, validation results, uploads, checkpoint outcomes, or debugging findings.",
        "Use flag_and_store when the current text corrects, contradicts, or invalidates an earlier assumption and the new fact should be kept.",
        "If candidate_bucket is transient, choose ignore unless the text states a reusable rule or completed milestone.",
        "If candidate_bucket is correction, prefer flag_and_store when the text rejects a previous approach, reports a regression, or says something failed and should change.",
        "Do not store one-off commands, notebook cells, or step-by-step instructions unless they record a completed durable workflow decision.",
        "Do not invent facts. Every fact and indexable must quote or paraphrase only evidence in the session window.",
        "Redacted placeholders such as [LOCAL_PATH] and [REDACTED_API_KEY] are safe; do not reconstruct the hidden values.",
        "Memory content must be concise, under 240 chars, and not just a raw transcript.",
        "Return at most 2 facts. You may return indexables: [] because the pipeline can build indexables locally.",
        "Prefer a short complete JSON object over a rich but incomplete one.",
        "Return exactly this shape:",
        "{\"action\":\"ignore|store_episodic|promote_semantic|update_existing|flag_conflict|flag_and_store\",\"memory\":null|{\"content\":\"...\",\"type\":\"episodic|semantic\",\"strength\":0.8,\"decay_rate\":0.02,\"emotional_weight\":0.3,\"confidence\":0.9,\"tags\":[\"...\"]},\"facts\":[],\"indexables\":[],\"updates\":[],\"conflicts\":[],\"reasoning\":\"short reason\"}"
      ].join(" ")
    },
    {
      role: "user",
      content: JSON.stringify({
        operation: "label_codex_session_window",
        source_id: candidate.source_id,
        candidate_bucket: candidate.candidate_bucket,
        current_turn: candidate.current_turn,
        nearby_turns: candidate.nearby_turns
      }, null, 2)
    }
  ], 1000, 0);
}

function teacherOutputToRow(candidate, output, index) {
  if (!output || output.__parse_error) {
    const baseInput = {
      source_kind: "codex_session_nemotron",
      source_id: candidate.source_id,
      session_id: candidate.session_id,
      current_turn: candidate.current_turn,
      nearby_turns: candidate.nearby_turns,
      memory_store: []
    };
    return createRememberExample(`codex-nemotron-${index}`, baseInput, ignoreOutput(output?.reasoning || "Teacher response could not be parsed as complete JSON."));
  }
  const action = normalizeAction(output.action);
  const baseInput = {
    source_kind: "codex_session_nemotron",
    source_id: candidate.source_id,
    session_id: candidate.session_id,
    candidate_bucket: candidate.candidate_bucket,
    current_turn: candidate.current_turn,
    nearby_turns: candidate.nearby_turns,
    memory_store: []
  };
  if (action === "ignore") {
    return createRememberExample(`codex-nemotron-${index}`, baseInput, ignoreOutput(output.reasoning || "Teacher labeled this session window as transient or not durable."));
  }
  const memory = normalizeMemory(output.memory, action, candidate.current_turn?.speaker);
  const facts = normalizeFacts(output.facts);
  let row = createRememberExample(`codex-nemotron-${index}`, baseInput, {
    action,
    memory,
    facts,
    indexables: normalizeIndexables(output.indexables, memory, facts),
    updates: Array.isArray(output.updates) ? output.updates : [],
    conflicts: Array.isArray(output.conflicts) ? output.conflicts : [],
    reasoning: typeof output.reasoning === "string" && output.reasoning.trim()
      ? output.reasoning.trim()
      : "Teacher generated a grounded write-time PSM label from the sanitized session window."
  });
  row = normalizeTrainingExample(row);
  return row;
}

function normalizeAction(value) {
  const action = String(value ?? "").trim();
  const allowed = new Set(["ignore", "store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"]);
  return allowed.has(action) ? action : "ignore";
}

function normalizeMemory(memory, action, speaker) {
  const type = action === "promote_semantic" ? "semantic" : (memory?.type === "semantic" ? "semantic" : "episodic");
  let evidence = cleanText(memory?.content).slice(0, 238);
  if (/^User\b/.test(evidence) && typeof speaker === "string" && speaker && speaker !== "User") {
    evidence = evidence.replace(/^User\b/, "The session");
  }
  return {
    content: evidence || "Codex session captured a durable PSM project memory.",
    type,
    strength: clamp01(memory?.strength ?? (type === "semantic" ? 0.86 : 0.78)),
    decay_rate: clamp01(memory?.decay_rate ?? (type === "semantic" ? 0.02 : 0.045)),
    emotional_weight: clamp01(memory?.emotional_weight ?? 0.35),
    confidence: clamp01(memory?.confidence ?? 0.86),
    tags: normalizeTags(memory?.tags ?? ["codex_session", type])
  };
}

function normalizeFacts(facts) {
  return (Array.isArray(facts) ? facts : [])
    .map((fact) => ({
      subject: cleanText(fact?.subject),
      predicate: snake(cleanText(fact?.predicate || "has_value")),
      value: cleanText(fact?.value),
      confidence: clamp01(fact?.confidence ?? 0.8),
      inference_kind: "explicit",
      evidence_text: cleanText(fact?.evidence_text || fact?.value)
    }))
    .filter((fact) => fact.subject && fact.predicate && fact.value && fact.evidence_text)
    .slice(0, 4);
}

function normalizeIndexables(indexables, memory, facts) {
  const valid = (Array.isArray(indexables) ? indexables : [])
    .map((item) => ({
      kind: ["mnemonic", "fact_anchor"].includes(item?.kind) ? item.kind : "mnemonic",
      key: hyphenKey(item?.key || memory.content),
      target_type: item?.target_type === "semantic" || item?.target_type === "episodic" ? item.target_type : memory.type,
      target_id: typeof item?.target_id === "string" ? item.target_id : "",
      salience: clamp01(item?.salience ?? 0.78),
      reconstructive_hint: cleanText(item?.reconstructive_hint || memory.content),
      evidence_text: cleanText(item?.evidence_text || memory.content),
      tags: normalizeTags(item?.tags ?? memory.tags)
    }))
    .filter((item) => item.key && item.reconstructive_hint && item.evidence_text)
    .slice(0, 3);
  return valid.length > 0 ? valid : buildIndexables({ content: memory.content, tags: memory.tags, facts, target_type: memory.type });
}

function loadMessages(root) {
  const rows = [];
  for (const file of listJsonl(root)) {
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
      if (!["user_message", "agent_message"].includes(payload.type)) continue;
      const text = sanitizeText(extractText(payload));
      if (!isUsableText(text)) continue;
      rows.push({
        session_id: sessionId,
        source_id: `${sessionId}:${lineIndex + 1}`,
        timestamp: row.timestamp,
        role: payload.type === "agent_message" ? "Assistant" : "User",
        text
      });
    }
  }
  return dedupe(rows.sort((left, right) => String(left.timestamp).localeCompare(String(right.timestamp))));
}

function selectCandidates(messages, count) {
  const scored = messages
    .map((message, index) => ({ message, index, bucket: candidateBucket(message.text, message.role), score: candidateScore(message.text) }))
    .filter((item) => item.bucket !== "skip")
    .sort((left, right) => right.score - left.score || left.index - right.index);
  const buckets = {
    correction: scored.filter((item) => item.bucket === "correction"),
    rule: scored.filter((item) => item.bucket === "rule"),
    milestone: scored.filter((item) => item.bucket === "milestone"),
    transient: scored.filter((item) => item.bucket === "transient")
  };
  const targets = {
    correction: Math.max(1, Math.floor(count * 0.22)),
    rule: Math.max(1, Math.floor(count * 0.28)),
    milestone: Math.max(1, Math.floor(count * 0.34)),
    transient: Math.max(1, Math.floor(count * 0.16))
  };
  const ordered = [
    ...buckets.correction.slice(0, targets.correction),
    ...buckets.rule.slice(0, targets.rule),
    ...buckets.milestone.slice(0, targets.milestone),
    ...buckets.transient.slice(0, targets.transient)
  ];
  for (const item of scored) {
    if (ordered.length >= count) break;
    if (!ordered.some((picked) => picked.index === item.index)) ordered.push(item);
  }
  const picked = [];
  const seen = new Set();
  for (const item of ordered.sort((left, right) => left.index - right.index)) {
    const key = item.message.text.toLowerCase().slice(0, 160);
    if (seen.has(key)) continue;
    seen.add(key);
    const nearby = messages.slice(Math.max(0, item.index - 2), Math.min(messages.length, item.index + 3))
      .filter((message) => message.source_id !== item.message.source_id)
      .map((message) => ({
        speaker: message.role,
        text: trim(compactForTeacher(message.text), 260),
        timestamp: message.timestamp
      }));
    picked.push({
      session_id: item.message.session_id,
      source_id: item.message.source_id,
      candidate_bucket: item.bucket,
      current_turn: {
        speaker: item.message.role,
        text: trim(compactForTeacher(item.message.text), 520),
        timestamp: item.message.timestamp
      },
      nearby_turns: nearby
    });
    if (picked.length >= count) break;
  }
  return picked;
}

function candidateScore(text) {
  const lower = text.toLowerCase();
  let score = 0;
  if (/psm|nano|retention|decay|checkpoint|validation|dataset|colab|huggingface|duckdb|memory/.test(lower)) score += 3;
  if (/should|must|need|make sure|quality|best|prefer|full dataset|training/.test(lower)) score += 2;
  if (/failed|wrong|empty|404|not found|disconnect|regression|forgot|miss/.test(lower)) score += 3;
  if (/accuracy|mae|score|cross-eval|results/.test(lower)) score += 2;
  if (/^i('|’)m |^i found |^next i('|’)m /.test(lower)) score -= 3;
  if (text.length < 30) score -= 2;
  return score;
}

function candidateBucket(text, role) {
  const lower = text.toLowerCase();
  if (/failed|wrong|empty|404|not found|disconnect|regression|forgot|miss|failure|error|should not|don't like|do not like|we went wrong/.test(lower)) {
    return "correction";
  }
  if (/should|must|need to|make sure|best practice|prefer|always|never|do not|don't|quality|full dataset|today|works well/.test(lower)) {
    return "rule";
  }
  if (/psm|nano|retention|decay|checkpoint|validation|dataset|colab|huggingface|duckdb|memory|accuracy|mae|cross-eval|results|training/.test(lower)) {
    return "milestone";
  }
  if (role === "Assistant" && (/^i('|â€™)m |^i found |^next i('|â€™)m |run this|use this|checking|inspecting/.test(lower) || lower.includes("[code_block]"))) {
    return "transient";
  }
  if (/^(okay|ok|yes|no|proceed|thanks|thank you|what'?s going on|whats going on)[.! ]*$/.test(lower)) {
    return "transient";
  }
  return "skip";
}

async function chatCompletion(messages, maxTokens, temperature) {
  let lastError = "";
  for (let attempt = 0; attempt <= requestMaxRetries; attempt++) {
    if (requestDelayMs > 0) await sleep(requestDelayMs);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);
    let response;
    try {
      response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: {
          "authorization": `Bearer ${apiKey}`,
          "content-type": "application/json"
        },
        signal: controller.signal,
        body: JSON.stringify({
          model,
          messages,
          temperature,
          max_tokens: maxTokens,
          response_format: { type: "json_object" }
        })
      });
    } catch (error) {
      clearTimeout(timeout);
      lastError = `Chat completion request failed: ${error.message}`;
      if (attempt >= requestMaxRetries) break;
      await sleep(Math.min(120000, 3000 * 2 ** attempt));
      continue;
    }
    clearTimeout(timeout);
    if (response.ok) {
      const data = await response.json();
      return data.choices?.[0]?.message?.content ?? "";
    }
    const body = await response.text();
    lastError = `Chat completion failed ${response.status}: ${body}`;
    if (response.status !== 429 || attempt >= requestMaxRetries) break;
    await sleep(rateLimitWaitMs(response, body, attempt));
  }
  throw new Error(lastError);
}

function rateLimitWaitMs(response, body, attempt) {
  const retryAfter = Number(response.headers?.get("retry-after"));
  if (Number.isFinite(retryAfter) && retryAfter > 0) return clampMs(retryAfter * 1000 + 1000);
  try {
    const parsed = JSON.parse(body);
    const seconds = Number(parsed?.error?.metadata?.retry_after_seconds ?? parsed?.error?.metadata?.retry_after_seconds_raw);
    if (Number.isFinite(seconds) && seconds > 0) return clampMs(seconds * 1000 + 1000);
    const headerSeconds = Number(parsed?.error?.metadata?.headers?.["Retry-After"]);
    if (Number.isFinite(headerSeconds) && headerSeconds > 0) return clampMs(headerSeconds * 1000 + 1000);
  } catch {
    // Fall through to exponential backoff.
  }
  return clampMs(5000 * 2 ** attempt);
}

function clampMs(value) {
  return Math.max(3000, Math.min(180000, value));
}

function parseTeacherJson(value) {
  const trimmed = String(value ?? "").trim();
  const candidates = completeJsonObjects(trimmed);
  for (const json of candidates.reverse()) {
    try {
      const parsed = JSON.parse(json);
      if (typeof parsed?.action === "string") return parsed;
    } catch {
      // Try earlier complete objects.
    }
  }
  return { __parse_error: true, action: "ignore", memory: null, facts: [], indexables: [], updates: [], conflicts: [], reasoning: "Teacher returned no complete JSON object with an action." };
}

function completeJsonObjects(text) {
  const objects = [];
  let start = -1;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let index = 0; index < text.length; index++) {
    const ch = text[index];
    if (inString) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === "\"") inString = false;
      continue;
    }
    if (ch === "\"") {
      inString = true;
      continue;
    }
    if (ch === "{") {
      if (depth === 0) start = index;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        objects.push(text.slice(start, index + 1));
        start = -1;
      }
    }
  }
  return objects;
}

function extractText(payload) {
  if (typeof payload.message === "string") return payload.message;
  if (Array.isArray(payload.content)) {
    return payload.content.map((item) => typeof item?.text === "string" ? item.text : "").filter(Boolean).join("\n");
  }
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

function compactForTeacher(value) {
  return sanitizeText(value)
    .replace(/```[\s\S]*?```/g, "[CODE_BLOCK]")
    .replace(/`[^`]{80,}`/g, "[INLINE_CODE]")
    .replace(/\b[A-Za-z]:\\[^ ]+/g, "[LOCAL_PATH]")
    .replace(/\s+/g, " ")
    .trim();
}

function isUsableText(text) {
  return text.length >= 16
    && text.length <= 6000
    && !/^<permissions instructions>|^<environment_context>|# Tools|base_instructions|developer instructions/i.test(text);
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

function dedupe(rows) {
  const seen = new Set();
  return rows.filter((row) => {
    const key = `${row.role}:${row.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function cleanText(value) {
  return typeof value === "string" ? value.trim().replace(/\s+/g, " ") : "";
}

function trim(value, max) {
  const text = cleanText(value);
  return text.length <= max ? text : `${text.slice(0, max - 3).trim()}...`;
}

function normalizeTags(tags) {
  return [...new Set((Array.isArray(tags) ? tags : [])
    .map((tag) => String(tag).trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""))
    .filter(Boolean))]
    .slice(0, 8);
}

function snake(value) {
  const key = value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return /^[a-z]/.test(key) ? key : "has_value";
}

function hyphenKey(value) {
  const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "into", "user", "memory", "codex", "session"]);
  const tokens = cleanText(value).toLowerCase().match(/[a-z0-9]+/g)?.filter((token) => token.length > 2 && !stop.has(token)).slice(0, 5) ?? [];
  return tokens.length > 0 ? tokens.join("-") : "session-memory";
}

function clamp01(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
}

function countBy(rows, getKey) {
  const counts = {};
  for (const row of rows) {
    const key = getKey(row);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
