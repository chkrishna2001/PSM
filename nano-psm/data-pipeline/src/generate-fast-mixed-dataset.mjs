#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { readJsonl, writeJsonl } from "./lib/jsonl.mjs";

const instruction = [
  "Perform the PSM memory operation for the current input.",
  "Return JSON only using the target schema.",
  "Do not use legacy keys such as operation or assistant_response.",
  "Do not write generic User when a speaker name is available.",
  "Only extract facts that are explicitly supported by evidence_text.",
  "Create compact indexables for stored memories so later recall can use mnemonic cues.",
  "For recall inputs, select grounded memory ids and indexable keys; do not answer from general knowledge."
].join(" ");

const args = parseArgs(process.argv.slice(2));
const baseDirs = listArg(args, "base", [
  "nano-psm/data-pipeline/data/generated",
  "nano-psm/data-pipeline/data/generated-local-psm"
]);
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/fast-mixed");
const validationRatio = numberArg(args, "validation-ratio", 0.15);
const maxTotal = intArg(args, "max-total", 10000);

const limits = {
  base: intArg(args, "base-limit", 2000),
  userPreference: intArg(args, "user-preference-limit", 2200),
  personaMem: intArg(args, "personamem-limit", 2200),
  personaRecall: intArg(args, "personamem-recall-limit", 900),
  longMemEval: intArg(args, "longmemeval-limit", 1800),
  realTalk: intArg(args, "realtalk-limit", 900),
  synthetic: intArg(args, "synthetic-limit", 900)
};

const examples = [
  ...loadBaseExamples(baseDirs, limits.base),
  ...generateUserPreferenceExamples("nano-psm/data-pipeline/data/raw/user-preference-564k/preference_extractor_564k.jsonl", limits.userPreference),
  ...generatePersonaMemExamples("nano-psm/data-pipeline/data/raw/personamem/shared_contexts_32k.jsonl", limits.personaMem),
  ...generatePersonaRecallExamples("nano-psm/data-pipeline/data/raw/personamem/questions_32k.csv", limits.personaRecall),
  ...generateLongMemEvalExamples("nano-psm/data-pipeline/data/raw/longmemeval/longmemeval_oracle.json", limits.longMemEval),
  ...generateRealTalkExamples("nano-psm/data-pipeline/data/raw/realtalk-mteb/realtalk-training.jsonl", limits.realTalk),
  ...generateSyntheticCoverageExamples(limits.synthetic)
].map(normalizeTrainingExample);

const deduped = dedupe(examples);
const limited = balancedLimit(deduped, maxTotal);
const { train, validation } = splitDeterministically(limited, validationRatio);

mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), limited);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "fast_mixed",
  base_dirs: baseDirs,
  limits,
  loaded_examples: examples.length,
  duplicate_examples_removed: examples.length - deduped.length,
  total_examples: limited.length,
  train_examples: train.length,
  validation_examples: validation.length,
  validation_ratio: validationRatio,
  action_mix: countBy(limited, (row) => row.output?.action ?? "unknown"),
  source_mix: countBy(limited, (row) => row.input?.source_kind ?? "unknown"),
  unavailable_sources: {
    perltqa: "No official/public HF dataset path was found by name during this run."
  }
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  loaded: examples.length,
  deduped: deduped.length,
  total: limited.length,
  train: train.length,
  validation: validation.length,
  action_mix: countBy(limited, (row) => row.output?.action ?? "unknown"),
  source_mix: countBy(limited, (row) => row.input?.source_kind ?? "unknown")
}, null, 2));

function loadBaseExamples(dirs, limit) {
  const rows = [];
  for (const dir of dirs) {
    const all = join(dir, "all.jsonl");
    const train = join(dir, "train.jsonl");
    const validation = join(dir, "validation.jsonl");
    if (existsSync(all)) rows.push(...readJsonl(all));
    else {
      if (existsSync(train)) rows.push(...readJsonl(train));
      if (existsSync(validation)) rows.push(...readJsonl(validation));
    }
  }
  return rows.slice(0, limit);
}

function generateUserPreferenceExamples(path, limit) {
  if (!existsSync(path)) return [];
  const rows = readJsonl(path);
  const examples = [];
  for (const [index, row] of rows.entries()) {
    if (examples.length >= limit) break;
    const inputText = cleanText(row.input);
    const parsed = safeJson(row.output);
    const preferences = Array.isArray(parsed?.preferences) ? parsed.preferences : [];
    if (preferences.length === 0) {
      examples.push(example(`user-pref-ignore-${index + 1}`, "user_preference_564k", {
        current_turn: { speaker: "User", text: inputText, timestamp: "" }
      }, ignore("Preference extractor produced no durable preference from this request.")));
      continue;
    }
    for (const [prefIndex, pref] of preferences.entries()) {
      if (examples.length >= limit) break;
      const condition = cleanText(pref.condition || "general");
      const action = sanitizePreferenceAction(cleanText(pref.action));
      if (!action) continue;
      const content = `User prefers to ${action}${condition && condition !== "general" ? ` when ${condition}` : ""}.`;
      examples.push(example(`user-pref-${index + 1}-${prefIndex + 1}`, "user_preference_564k", {
        current_turn: { speaker: "User", text: inputText, timestamp: "" }
      }, storeSemantic({
        content,
        tags: ["preference", keyToken(condition), keyToken(action)].filter(Boolean),
        confidence: clamp(Number(pref.confidence) || 0.86, 0.55, 0.98),
        facts: [fact("User", "prefers", action, inputText || action, Number(pref.confidence) || 0.86)],
        reasoning: "Preference rule is explicit in the source extractor output."
      })));
    }
  }
  return examples;
}

function generatePersonaMemExamples(path, limit) {
  if (!existsSync(path)) return [];
  const lines = readJsonl(path);
  const examples = [];
  for (const object of lines) {
    for (const [contextId, messages] of Object.entries(object)) {
      if (!Array.isArray(messages)) continue;
      let userTurn = 0;
      for (const message of messages) {
        if (examples.length >= limit) return examples;
        if (message?.role !== "user") continue;
        userTurn++;
        const text = stripSpeaker(cleanText(message.content));
        if (!text || text.length < 24) continue;
        const output = outputForPersonaText(text, "User");
        examples.push(example(`personamem-${contextId}-${userTurn}`, "personamem", {
          source_id: `${contextId}:${userTurn}`,
          current_turn: { speaker: "User", text, timestamp: "" }
        }, output));
      }
    }
  }
  return examples;
}

function generatePersonaRecallExamples(path, limit) {
  if (!existsSync(path)) return [];
  const records = parseCsv(readFileSync(path, "utf8"));
  const examples = [];
  for (const [index, record] of records.entries()) {
    if (examples.length >= limit) break;
    const question = cleanText(record.user_question_or_message);
    const topic = cleanText(record.topic);
    if (!question) continue;
    const memory = memoryStoreItem({
      id: `personamem-pref-${index + 1}`,
      source_id: String(record.shared_context_id || `personamem:${index + 1}`),
      content: `User has a durable preference or personal context related to ${topic || "the current request"}.`,
      tags: ["personamem", keyToken(topic), "preference"].filter(Boolean),
      target_type: "semantic"
    });
    examples.push({
      id: `personamem-recall-${index + 1}`,
      instruction,
      input: {
        operation: "recall",
        source_kind: "personamem_recall",
        current_query: { question, category: cleanText(record.question_type) },
        memory_store: [
          memory,
          memoryStoreItem({
            id: `personamem-distractor-${index + 1}`,
            source_id: `personamem:distractor:${index + 1}`,
            content: "User asked a generic one-off question with no durable preference signal.",
            tags: ["distractor"],
            target_type: "semantic"
          })
        ]
      },
      output: recallOutput([memory], "Selected the memory row carrying the PersonaMem preference context for this query.")
    });
  }
  return examples;
}

function generateLongMemEvalExamples(path, limit) {
  if (!existsSync(path)) return [];
  const data = JSON.parse(readFileSync(path, "utf8"));
  const rows = Array.isArray(data) ? data : Object.values(data).flat();
  const examples = [];
  for (const [index, row] of rows.entries()) {
    if (examples.length >= limit) break;
    const question = cleanText(row.question);
    const answer = cleanText(row.answer);
    const sessions = Array.isArray(row.haystack_sessions) ? row.haystack_sessions : [];
    const memories = [];
    for (const [sessionIndex, session] of sessions.entries()) {
      if (!Array.isArray(session)) continue;
      for (const [messageIndex, message] of session.entries()) {
        if (message?.role !== "user" && !message?.has_answer) continue;
        const text = cleanText(message.content);
        if (!text) continue;
        memories.push(memoryStoreItem({
          id: `longmem-${index + 1}-${sessionIndex + 1}-${messageIndex + 1}`,
          source_id: cleanText(row.haystack_session_ids?.[sessionIndex]) || `longmem:${index + 1}:${sessionIndex + 1}`,
          source_timestamp: cleanText(row.haystack_dates?.[sessionIndex]),
          content: summarizeMemoryText(text),
          tags: ["longmemeval", keyToken(row.question_type), "evidence"].filter(Boolean),
          target_type: "episodic"
        }));
      }
    }
    const selected = memories.slice(0, Math.max(1, Math.min(5, memories.length)));
    if (question && selected.length > 0) {
      examples.push({
        id: `longmemeval-recall-${index + 1}`,
        instruction,
        input: {
          operation: "recall",
          source_kind: "longmemeval",
          current_query: { question, category: cleanText(row.question_type) },
          memory_store: memories.slice(0, 12)
        },
        output: recallOutput(selected, answer ? `Selected LongMemEval evidence memories for answer: ${answer}` : "Selected LongMemEval evidence memories.")
      });
    }
    if (examples.length >= limit) break;
  }
  return examples;
}

function generateRealTalkExamples(path, limit) {
  if (!existsSync(path)) return [];
  const rows = readJsonl(path);
  const examples = [];
  for (const [index, row] of rows.entries()) {
    if (examples.length >= limit) break;
    const query = cleanText(row.query);
    const positive = cleanText(row.positive);
    const negative = cleanText(row.negative);
    if (!query || !positive) continue;
    const selected = memoryStoreItem({
      id: `realtalk-pos-${index + 1}`,
      source_id: cleanText(row.positive_id) || `realtalk:pos:${index + 1}`,
      content: summarizeMemoryText(positive),
      tags: ["realtalk", keyToken(row.subset), "dialogue"].filter(Boolean),
      target_type: "episodic"
    });
    const store = [selected];
    if (negative) {
      store.push(memoryStoreItem({
        id: `realtalk-neg-${index + 1}`,
        source_id: cleanText(row.negative_id) || `realtalk:neg:${index + 1}`,
        content: summarizeMemoryText(negative),
        tags: ["realtalk", "distractor"],
        target_type: "episodic"
      }));
    }
    examples.push({
      id: `realtalk-recall-${index + 1}`,
      instruction,
      input: {
        operation: "recall",
        source_kind: "realtalk",
        current_query: { question: query, category: cleanText(row.subset) },
        memory_store: store
      },
      output: recallOutput([selected], "Selected the REALTALK retrieval-positive dialogue memory.")
    });
  }
  return examples;
}

function generateSyntheticCoverageExamples(limit) {
  const examples = [];
  for (let i = 1; i <= limit; i++) {
    const lane = i % 6;
    if (lane === 0) {
      examples.push(example(`synthetic-hard-ignore-${i}`, "synthetic_hard_negative", {
        current_turn: { speaker: "Assistant", text: `Build completed in ${(i % 9) + 1}.${i % 10}s with no warnings.`, timestamp: "2026-05-27T10:00:00Z" }
      }, ignore("Transient command output is not durable memory.")));
    } else if (lane === 1) {
      examples.push(example(`synthetic-flag-store-${i}`, "synthetic", {
        current_turn: { speaker: "User", text: `Correction: evaluation lane ${i} must use checkpoint-best.pt, not checkpoint-last.pt.`, timestamp: "2026-05-27T10:00:00Z" },
        memory_store: [{ id: `ckpt-${i}`, content: `Evaluation lane ${i} uses checkpoint-last.pt.` }]
      }, {
        action: "flag_and_store",
        memory: memory(`Evaluation lane ${i} must use checkpoint-best.pt instead of checkpoint-last.pt.`, "semantic", ["checkpoint", "evaluation"], 0.93),
        facts: [fact(`evaluation lane ${i}`, "uses_checkpoint", "checkpoint-best.pt", "must use checkpoint-best.pt", 0.96)],
        indexables: buildIndexables(`Evaluation lane ${i} must use checkpoint-best.pt instead of checkpoint-last.pt.`, ["checkpoint", "evaluation"]),
        updates: [],
        conflicts: [{ target_id: `ckpt-${i}`, reason: "New correction conflicts with prior checkpoint-last setting." }],
        reasoning: "Explicit correction both stores new state and flags the old memory."
      }));
    } else if (lane === 2) {
      examples.push(example(`synthetic-episodic-${i}`, "synthetic", {
        current_turn: { speaker: "User", text: `On May ${padDay(i)}, the memory smoke found ${i % 7} recall-count misses in validation.`, timestamp: `2026-05-${padDay(i)}T12:00:00Z` }
      }, storeEpisodic({
        content: `On 2026-05-${padDay(i)}, the memory smoke found ${i % 7} recall-count misses in validation.`,
        tags: ["validation", "recall_count", "smoke"],
        facts: [fact("memory smoke", "recall_count_misses", String(i % 7), "recall-count misses in validation", 0.95)],
        reasoning: "Dated validation result with an explicit count."
      })));
    } else if (lane === 3) {
      examples.push(example(`synthetic-pref-${i}`, "synthetic", {
        current_turn: { speaker: "Rina", text: `I prefer short implementation answers with exact file paths when debugging build failures.`, timestamp: "" }
      }, storeSemantic({
        content: "Rina prefers short implementation answers with exact file paths when debugging build failures.",
        tags: ["preference", "debugging", "file_paths"],
        facts: [fact("Rina", "prefers", "short implementation answers with exact file paths", "I prefer short implementation answers with exact file paths", 0.97)]
      })));
    } else if (lane === 4) {
      examples.push(example(`synthetic-update-${i}`, "synthetic", {
        current_turn: { speaker: "User", text: `Update: dataset run ${i} now targets 10k gated rows before primary training.`, timestamp: "2026-05-27T10:00:00Z" },
        memory_store: [{ id: `dataset-run-${i}`, content: `Dataset run ${i} targets 1k compatibility rows.` }]
      }, {
        action: "update_existing",
        memory: memory(`Dataset run ${i} now targets 10k gated rows before primary training.`, "semantic", ["dataset", "training"], 0.88),
        facts: [fact(`dataset run ${i}`, "targets_rows", "10000 gated rows", "targets 10k gated rows", 0.96)],
        indexables: buildIndexables(`Dataset run ${i} now targets 10k gated rows before primary training.`, ["dataset", "training"]),
        updates: [{ target_id: `dataset-run-${i}`, relationship: "replaces", reason: "New target supersedes compatibility-row target." }],
        conflicts: [],
        reasoning: "Explicit project-state update."
      }));
    } else {
      const selected = memoryStoreItem({
        id: `synthetic-recall-memory-${i}`,
        source_id: `synthetic:recall:${i}`,
        content: `Training run ${i} should inspect recall-count mistakes before increasing model steps.`,
        tags: ["training", "recall_count", "validation"],
        target_type: "semantic"
      });
      examples.push({
        id: `synthetic-recall-${i}`,
        instruction,
        input: {
          operation: "recall",
          source_kind: "synthetic_recall",
          current_query: { question: `What should training run ${i} inspect before increasing steps?`, category: "training" },
          memory_store: [selected]
        },
        output: recallOutput([selected], "Selected the memory containing the exact training-run inspection rule.")
      });
    }
  }
  return examples;
}

function outputForPersonaText(text, speaker) {
  const lower = text.toLowerCase();
  const first = (text.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? text).toLowerCase();
  if (isQuestionOrChatter(text)) return ignore("Question, acknowledgement, or conversational bridge without durable memory.");
  if (isContextlessReaction(text)) return ignore("Positive reaction points to prior context but does not name a durable preference.");
  if (!isPersonaSelfStatement(first)) return ignore("PersonaMem turn depends on unresolved prior context.");
  if (hasPersonaPreferenceSignal(first)) {
    return storeSemantic({
      content: personaContent(text, speaker),
      tags: ["personamem", "preference", keyToken(text)].filter(Boolean),
      facts: [fact(speaker, "has_preference_context", summarizeMemoryText(text), text, 0.84)],
      reasoning: "PersonaMem user turn states a stable preference, interest, goal, or dislike."
    });
  }
  if (/\b(20\d{2}|19\d{2}|attended|launched|created|hosted|joined|moved|visited|collaborated|volunteering|acquired)\b/.test(first)) {
    return storeEpisodic({
      content: personaContent(text, speaker),
      tags: ["personamem", "episodic", keyToken(text)].filter(Boolean),
      facts: [fact(speaker, "experienced", summarizeMemoryText(text), text, 0.82)],
      reasoning: "PersonaMem user turn describes a specific personal event or milestone."
    });
  }
  return ignore("No durable memory can be extracted without over-inference.");
}

function semanticPredicatePhrase(text) {
  const summary = summarizeMemoryText(text).replace(/^I\b/i, "prefers or identifies with");
  return summary.endsWith(".") ? summary.slice(0, -1) : summary;
}

function personaContent(text, speaker) {
  const summary = summarizeMemoryText(text).replace(/[.!?]$/, "");
  let converted = summary
    .replace(/^One exciting development is that I\b/i, `${speaker}`)
    .replace(/^(A few days later|A couple days later|This week|Last week|Recently|More recently),?\s+I['â€™]m\b/i, `$1, ${speaker} is`)
    .replace(/^(A few days later|A couple days later|This week|Last week|Recently|More recently),?\s+I am\b/i, `$1, ${speaker} is`)
    .replace(/^(A few days later|A couple days later|This week|Last week|Recently|More recently),?\s+I['â€™]ve\b/i, `$1, ${speaker} has`)
    .replace(/^(A few days later|A couple days later|This week|Last week|Recently|More recently),?\s+I\b/i, `$1, ${speaker}`)
    .replace(/^I['’]m\b/i, `${speaker} is`)
    .replace(/^I am\b/i, `${speaker} is`)
    .replace(/^I['’]ve\b/i, `${speaker} has`)
    .replace(/^I have\b/i, `${speaker} has`)
    .replace(/^We\b/i, `${speaker} and others`)
    .replace(/^I\b/i, speaker)
    .replace(/\bmy\b/gi, `${speaker}'s`)
    .replace(/\bme\b/gi, speaker);
  if (!new RegExp(`^(?:${escapeRegExp(speaker)}|A few days later|A couple days later|This week|Last week|Recently|More recently)\\b`, "i").test(converted)) {
    converted = `${speaker} ${semanticPredicatePhrase(text)}`;
  }
  return `${converted.replace(/\s+/g, " ").trim()}.`;
}

function hasPersonaPreferenceSignal(firstSentence) {
  return /\b(prefer|passion|interested|goal|favorite|dislike|looking for|want|need|hoping to)\b/.test(firstSentence)
    || /\bi\s+(really\s+)?(like|love|enjoy)\b/.test(firstSentence);
}

function isPersonaSelfStatement(firstSentence) {
  return /^(i|i'm|i’ve|i've|i am|i have|my|we|recently|more recently|this week|last week|a few days later|a couple days later|one exciting development)\b/i.test(firstSentence);
}

function example(id, sourceKind, input, output) {
  return {
    id,
    instruction,
    input: {
      operation: "remember",
      source_kind: sourceKind,
      source_id: input.source_id ?? id,
      prior_context: input.prior_context ?? [],
      memory_store: input.memory_store ?? [],
      ...input
    },
    output
  };
}

function ignore(reasoning) {
  return { action: "ignore", memory: null, facts: [], indexables: [], updates: [], conflicts: [], reasoning };
}

function storeSemantic(options) {
  return storeMemory("promote_semantic", "semantic", options);
}

function storeEpisodic(options) {
  return storeMemory("store_episodic", "episodic", options);
}

function storeMemory(action, type, options) {
  const content = compactMemoryContent(options.content);
  return {
    action,
    memory: memory(content, type, options.tags ?? [], options.confidence ?? 0.88),
    facts: options.facts ?? [],
    indexables: options.indexables ?? buildIndexables(content, options.tags ?? [], type),
    updates: options.updates ?? [],
    conflicts: options.conflicts ?? [],
    reasoning: options.reasoning ?? "Durable memory directly supported by source evidence."
  };
}

function memory(content, type, tags, confidence) {
  return {
    content,
    type,
    strength: type === "semantic" ? 0.86 : 0.82,
    decay_rate: type === "semantic" ? 0.02 : 0.04,
    emotional_weight: 0.35,
    confidence: clamp(confidence, 0.5, 0.99),
    tags: tags.map(keyToken).filter(Boolean).slice(0, 8)
  };
}

function fact(subject, predicate, value, evidenceText, confidence = 0.88) {
  return {
    subject: cleanText(subject),
    predicate: snake(predicate),
    value: cleanText(value),
    confidence: clamp(confidence, 0.5, 0.99),
    inference_kind: "explicit",
    evidence_text: cleanText(evidenceText)
  };
}

function recallOutput(selected, reasoning) {
  return {
    action: "recall_context",
    memory: null,
    facts: [],
    indexables: [],
    updates: [],
    conflicts: [],
    recall: {
      query_intent: "memory_recall",
      selected_memory_ids: selected.map((item) => item.id),
      selected_indexable_keys: selected.flatMap((item) => item.indexables ?? []).map((item) => item.key).slice(0, 8),
      max_items: Math.min(5, selected.length),
      reasoning
    },
    reasoning: "Recall should select grounded memory rows and indexable keys."
  };
}

function memoryStoreItem(input) {
  const content = compactMemoryContent(input.content);
  const tags = (input.tags ?? []).map(keyToken).filter(Boolean);
  return {
    id: input.id,
    source_id: input.source_id,
    source_timestamp: input.source_timestamp ?? "",
    speaker: input.speaker ?? "User",
    content,
    tags,
    indexables: buildIndexables(content, tags, input.target_type ?? "semantic").map((item) => ({ ...item, target_id: input.id }))
  };
}

function buildIndexables(content, tags = [], targetType = "semantic") {
  const key = unique([...tags.flatMap(tokenize), ...tokenize(content)]).slice(0, 4).join("-") || "memory-anchor";
  return [{
    kind: "mnemonic",
    key,
    target_type: targetType,
    target_id: "",
    salience: 0.84,
    reconstructive_hint: summarizeMemoryText(content, 150),
    evidence_text: content,
    tags: tags.slice(0, 6)
  }];
}

function normalizeTrainingExample(row) {
  const output = row.output ?? {};
  const normalized = {
    ...output,
    facts: Array.isArray(output.facts) ? output.facts : [],
    indexables: Array.isArray(output.indexables) ? output.indexables : [],
    updates: Array.isArray(output.updates) ? output.updates : [],
    conflicts: Array.isArray(output.conflicts) ? output.conflicts : [],
    reasoning: typeof output.reasoning === "string" ? output.reasoning : ""
  };
  if (normalized.memory && normalized.indexables.length === 0) {
    normalized.indexables = buildIndexables(normalized.memory.content, normalized.memory.tags ?? [], normalized.memory.type);
  }
  if (normalized.memory?.content) {
    normalized.memory = {
      ...normalized.memory,
      content: compactMemoryContent(normalized.memory.content)
    };
    normalized.indexables = buildIndexables(normalized.memory.content, normalized.memory.tags ?? [], normalized.memory.type);
  }
  if (normalized.action === "recall_context" && !normalized.recall) normalized.recall = recallOutput([], "No grounded memory selected.").recall;
  return {
    ...row,
    instruction: row.instruction ?? instruction,
    input: {
      prior_context: [],
      memory_store: [],
      ...row.input
    },
    output: normalized
  };
}

function dedupe(rows) {
  const seen = new Set();
  const result = [];
  for (const row of rows) {
    const key = [
      row.output?.action,
      row.input?.source_kind,
      normalize(row.output?.memory?.content || row.input?.current_turn?.text || row.input?.current_query?.question || "")
    ].join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(row);
  }
  return result;
}

function balancedLimit(rows, maxTotal) {
  if (!maxTotal || rows.length <= maxTotal) return rows;
  const target = {
    ignore: 0.18,
    store_episodic: 0.18,
    promote_semantic: 0.24,
    recall_context: 0.22,
    update_existing: 0.08,
    flag_conflict: 0.06,
    flag_and_store: 0.04
  };
  const byAction = groupBy(rows, (row) => row.output?.action ?? "unknown");
  const selected = [];
  for (const [action, ratio] of Object.entries(target)) {
    selected.push(...(byAction.get(action) ?? []).slice(0, Math.round(maxTotal * ratio)));
  }
  for (const row of rows.filter((item) => item.output?.action !== "ignore")) {
    if (selected.length >= maxTotal) break;
    if (!selected.includes(row)) selected.push(row);
  }
  for (const row of rows) {
    if (selected.length >= maxTotal) break;
    if (!selected.includes(row)) selected.push(row);
  }
  return selected.slice(0, maxTotal);
}

function splitDeterministically(items, ratio) {
  const validation = [];
  const train = [];
  const interval = Math.max(2, Math.round(1 / ratio));
  items.forEach((item, index) => {
    if (index % interval === 0) validation.push(item);
    else train.push(item);
  });
  return { train, validation };
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];
    if (char === "\"" && inQuotes && next === "\"") {
      field += "\"";
      i++;
    } else if (char === "\"") {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i++;
      row.push(field);
      if (row.some((item) => item.length > 0)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  const [header, ...records] = rows;
  return records.map((values) => Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""])));
}

function safeJson(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function isQuestionOrChatter(text) {
  const lower = text.toLowerCase().trim();
  const first = lower.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? lower;
  if (lower.endsWith("?")) return true;
  if (/^(hi|hey|hello|thanks|thank you)\b/.test(lower)) return true;
  if (/^(yes|yeah|yep|exactly|absolutely|definitely|for sure|oh for sure|right|true)[.! ]*$/.test(first)) return text.length < 160;
  if (/^(that sounds|sounds good|great|awesome|cool|nice)\b/.test(lower) && text.length < 160) return true;
  return false;
}

function isContextlessReaction(text) {
  const first = cleanText(text).match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? cleanText(text);
  return /^I\s+(really\s+)?(like|love|enjoy)\s+(that|this|it)\b/i.test(first) && cleanText(text).length < 180;
}

function compactMemoryContent(text, max = 200) {
  const cleaned = cleanText(text);
  if (cleaned.length <= max) return cleaned;
  const sentence = cleaned.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? cleaned;
  const source = sentence.length < 80 ? cleaned : sentence;
  return source.length <= max ? source : `${source.slice(0, max - 3).trim()}...`;
}

function summarizeMemoryText(text, max = 220) {
  const cleaned = stripSpeaker(cleanText(text));
  const sentence = cleaned.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? cleaned;
  return sentence.length <= max ? sentence : `${sentence.slice(0, max - 3).trim()}...`;
}

function stripSpeaker(text) {
  return text.replace(/^(User|Assistant|System):\s*/i, "").replace(/\s*\]\)?\s*$/, "").trim();
}

function sanitizePreferenceAction(text) {
  return text
    .replace(/\breturn JSON\b/gi, "return structured data")
    .replace(/\bJSON-formatted\b/gi, "structured")
    .replace(/\bJSON format\b/gi, "structured format")
    .replace(/\bschema\b/gi, "format")
    .replace(/\btarget format\b/gi, "requested format")
    .trim();
}

function cleanText(value) {
  return typeof value === "string"
    ? value
      .replace(/\uFFFD/g, "'")
      .replace(/â€™/g, "'")
      .replace(/â€˜/g, "'")
      .replace(/â€œ|â€/g, "\"")
      .replace(/â€”|â€“|â€‘/g, "-")
      .replace(/\s+/g, " ")
      .trim()
    : "";
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalize(value) {
  return cleanText(value).toLowerCase();
}

function keyToken(value) {
  return tokenize(value).slice(0, 3).join("_");
}

function tokenize(value) {
  const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "into", "user", "assistant", "memory", "current", "answer"]);
  return cleanText(value).toLowerCase().match(/[a-z0-9]+/g)?.filter((token) => token.length > 2 && !stop.has(token)) ?? [];
}

function snake(value) {
  return cleanText(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "has_value";
}

function unique(items) {
  return [...new Set(items)];
}

function groupBy(rows, getKey) {
  const grouped = new Map();
  for (const row of rows) {
    const key = getKey(row);
    const list = grouped.get(key) ?? [];
    list.push(row);
    grouped.set(key, list);
  }
  return grouped;
}

function countBy(rows, getKey) {
  const counts = {};
  for (const row of rows) {
    const key = getKey(row);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function padDay(value) {
  return String(((value - 1) % 28) + 1).padStart(2, "0");
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

function numberArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isFinite(value) && value > 0 && value < 1 ? value : fallback;
}

function listArg(parsed, key, fallback) {
  const value = parsed[key];
  return typeof value === "string" && value.trim()
    ? value.split(",").map((item) => item.trim()).filter(Boolean)
    : fallback;
}
