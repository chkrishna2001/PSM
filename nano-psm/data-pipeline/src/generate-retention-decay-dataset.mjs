#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { writeJsonl } from "./lib/jsonl.mjs";

const instruction = [
  "Perform the PSM memory operation for the current input.",
  "Return JSON only using the target schema.",
  "Do not use legacy keys such as operation or assistant_response.",
  "Only extract facts that are explicitly supported by evidence_text.",
  "Choose memory scores that reflect retention: temporary memories should decay quickly; durable and critical memories should decay slowly.",
  "Use update_existing only when new information clearly replaces an older memory.",
  "Use flag_conflict for uncertain contradictions and flag_and_store for clear durable corrections."
].join(" ");

const args = parseArgs(process.argv.slice(2));
const outDir = stringArg(args, "out", "nano-psm/data-pipeline/data/retention-decay-5k");
const total = intArg(args, "total", 5000);
const validationRatio = numberArg(args, "validation-ratio", 1 / 7);

const plannedCounts = scaleCounts({
  ignore: 1000,
  short_lived: 750,
  normal_episodic: 750,
  durable_semantic: 900,
  critical_low_decay: 400,
  update_existing: 500,
  flag_conflict: 350,
  flag_and_store: 350
}, total);

const rows = [
  ...generateIgnoreRetentionRows(plannedCounts.ignore),
  ...generateShortLivedRows(plannedCounts.short_lived),
  ...generateNormalEpisodicRows(plannedCounts.normal_episodic),
  ...generateDurableSemanticRows(plannedCounts.durable_semantic),
  ...generateCriticalLowDecayRows(plannedCounts.critical_low_decay),
  ...generateUpdateExistingRows(plannedCounts.update_existing),
  ...generateFlagConflictRows(plannedCounts.flag_conflict),
  ...generateFlagAndStoreRows(plannedCounts.flag_and_store)
].map((row, index) => ({ ...row, id: `${row.id}-${String(index + 1).padStart(5, "0")}` }));

const { train, validation } = splitDeterministically(rows, validationRatio);

mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), rows);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  source: "retention_decay_curriculum",
  total_examples: rows.length,
  train_examples: train.length,
  validation_examples: validation.length,
  validation_ratio: validationRatio,
  planned_counts: plannedCounts,
  action_mix: countBy(rows, (row) => row.output.action),
  source_mix: countBy(rows, (row) => row.input.source_kind),
  retention_mix: countBy(rows, (row) => row.output.memory?.tags?.find((tag) => tag.startsWith("retention_")) ?? "retention_ignore")
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  total: rows.length,
  train: train.length,
  validation: validation.length,
  action_mix: countBy(rows, (row) => row.output.action),
  source_mix: countBy(rows, (row) => row.input.source_kind)
}, null, 2));

function generateIgnoreRetentionRows(count) {
  const templates = [
    "Thanks, that works for now.",
    "Write the next response in pirate style.",
    "Make this paragraph sound more dramatic.",
    "Give me three punchier title ideas for this article.",
    "Okay, continue.",
    "The command printed no output.",
    "Use bullet points in this one reply.",
    "Can you make the wording warmer?",
    "Hello, are you there?",
    "Summarize this pasted article in two sentences."
  ];
  return Array.from({ length: count }, (_, index) => {
    const text = variant(templates, index, ` Ignore sample ${index + 1}.`);
    return example(`retention-ignore-${index + 1}`, sourceKind(index), {
      current_turn: turn("User", text, timestamp(index))
    }, ignore("The current turn is a greeting, one-off formatting request, acknowledgement, or transient command observation with no durable memory."));
  });
}

function generateShortLivedRows(count) {
  const templates = [
    (n) => `This run is using checkpoint-temp-${n}.pt just for tonight.`,
    (n) => `Use the throwaway API key only for smoke test ${n}.`,
    (n) => `The current Colab runtime for lane ${n} has 12GB RAM available.`,
    (n) => `For today's debugging lane ${n}, keep the temporary log path at C:/tmp/psm-${n}.log.`,
    (n) => `Training run ${n} is waiting on validation upload before the next command.`
  ];
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const text = templates[index % templates.length](n);
    const content = shortLivedContent(text, n);
    return example(`retention-short-${n}`, sourceKind(index), {
      current_turn: turn("User", text, timestamp(index))
    }, storeMemory({
      action: "store_episodic",
      type: "episodic",
      content,
      tags: ["retention_short_lived", "temporary", "debug", "session_state", `lane_${n % 17}`],
      scores: score(index, [0.45, 0.65], [0.12, 0.30], shortLivedEmotion(index), [0.75, 0.95]),
      fact: fact(`temporary lane ${n}`, "has_state", content, text, 0.88),
      reasoning: "Temporary task state is useful briefly and should decay quickly."
    }));
  });
}

function generateNormalEpisodicRows(count) {
  const archiveEvery = Math.max(1, Math.floor(count / Math.min(150, count)));
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const day = padDay(n);
    const archive = n % archiveEvery === 0;
    const text = archive
      ? `The obsolete migration branch retention-${n} was deleted after merge on May ${day}. Keep it only for historical audit if needed.`
      : `On May ${day}, validation run ${n} found ${n % 7} recall-count misses in the memory smoke.`;
    const content = archive
      ? `On 2026-05-${day}, obsolete migration branch retention-${n} was deleted after merge and is only historically useful.`
      : `On 2026-05-${day}, validation run ${n} found ${n % 7} recall-count misses in the memory smoke.`;
    return example(`retention-normal-episodic-${n}`, sourceKind(index + 1000), {
      current_turn: turn("User", text, `2026-05-${day}T12:00:00Z`)
    }, storeMemory({
      action: "store_episodic",
      type: "episodic",
      content,
      tags: archive
        ? ["retention_normal_episodic", "archive_candidate", "historical", "merged_branch", `run_${n}`]
        : ["retention_normal_episodic", "validation", "benchmark", "recall_count", `run_${n}`],
      scores: archive
        ? score(index, [0.62, 0.78], [0.07, 0.11], 0.16, [0.84, 0.96])
        : score(index, [0.70, 0.85], [0.04, 0.08], 0.28, [0.85, 0.98]),
      fact: archive
        ? fact(`retention branch ${n}`, "was_deleted_after_merge", "true", text, 0.91)
        : fact(`validation run ${n}`, "recall_count_misses", String(n % 7), text, 0.94),
      reasoning: archive
        ? "Old project event is still useful historically but should rank down over time."
        : "Dated validation result is a normal episodic memory with moderate decay."
    }));
  });
}

function generateDurableSemanticRows(count) {
  const templates = [
    (n) => [`Always use the hf CLI directly for dataset sync lane ${n}; do not use deprecated huggingface-cli.`, `Dataset sync lane ${n} should use the hf CLI directly instead of deprecated huggingface-cli.`, ["project_rule", "hf_cli"]],
    (n) => [`The office VM for PSM lane ${n} is CPU-only, so CUDA cannot be assumed.`, `PSM lane ${n} runs on a CPU-only office VM, so CUDA cannot be assumed.`, ["environment", "cpu_only"]],
    (n) => [`Reviewed datasets for lane ${n} must be gated and sampled before training.`, `Reviewed datasets for lane ${n} must be gated and sampled before training.`, ["dataset_gate", "training"]],
    (n) => [`For training reports in lane ${n}, show action mix and failure buckets first.`, `Training reports in lane ${n} should show action mix and failure buckets first.`, ["preference", "reporting"]],
    (n) => [`From now on, keep release notes concise for package lane ${n}.`, `Release notes for package lane ${n} should stay concise.`, ["preference", "release_notes"]]
  ];
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const [text, content, extraTags] = templates[index % templates.length](n);
    return example(`retention-durable-${n}`, sourceKind(index + 2000), {
      current_turn: turn("User", text, timestamp(index))
    }, storeMemory({
      action: "promote_semantic",
      type: "semantic",
      content,
      tags: ["retention_durable_semantic", "durable", ...extraTags, `lane_${n % 23}`],
      scores: score(index, [0.80, 0.92], [0.01, 0.03], durableEmotion(index), [0.85, 0.98]),
      fact: fact(subjectFor(content), "has_rule", content, text, 0.94),
      reasoning: "The turn states a durable rule, preference, or environment constraint."
    }));
  });
}

function generateCriticalLowDecayRows(count) {
  const templates = [
    (n) => [`Never write real access tokens into retention dataset row ${n}; use placeholders only.`, `Retention dataset row ${n} must use placeholders instead of real access tokens.`, ["security", "access_token"]],
    (n) => [`Do not assume PSM deployment lane ${n} has GPU access; many office VMs are CPU-only.`, `PSM deployment lane ${n} must not assume GPU access because many office VMs are CPU-only.`, ["environment", "gpu_assumption"]],
    (n) => [`Before uploading retention dataset ${n}, the standard gate must pass locally.`, `Retention dataset ${n} must pass the standard local gate before upload.`, ["quality_gate", "upload"]],
    (n) => [`Do not delete archived memory for lane ${n} unless the user or policy explicitly requests deletion.`, `Archived memory for lane ${n} must not be deleted without an explicit user or policy request.`, ["archive_policy", "deletion"]]
  ];
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const [text, content, extraTags] = templates[index % templates.length](n);
    return example(`retention-critical-${n}`, sourceKind(index + 3000), {
      current_turn: turn("User", text, timestamp(index))
    }, storeMemory({
      action: "promote_semantic",
      type: "semantic",
      content,
      tags: ["retention_critical_low_decay", "critical", "durable", ...extraTags, `lane_${n % 19}`],
      scores: score(index, [0.90, 0.98], [0.001, 0.01], criticalEmotion(index), [0.90, 0.99]),
      fact: fact(subjectFor(content), "has_critical_rule", content, text, 0.96),
      reasoning: "Critical project, safety, access, or retention policy should persist with very low decay."
    }));
  });
}

function generateUpdateExistingRows(count) {
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const targetId = `dataset-run-${n}`;
    const oldContent = `Dataset run ${n} targets 1k compatibility rows before primary training.`;
    const text = `Actually, dataset run ${n} now targets 10k gated rows before primary training, not 1k.`;
    const content = `Dataset run ${n} now targets 10k gated rows before primary training.`;
    return example(`retention-update-${n}`, sourceKind(index + 4000), {
      current_turn: turn("User", text, timestamp(index)),
      memory_store: [memoryStoreItem(targetId, oldContent, ["dataset", "training"], "semantic")]
    }, storeMemory({
      action: "update_existing",
      type: "semantic",
      content,
      tags: ["retention_update_existing", "dataset", "training", "replacement", `run_${n}`],
      scores: score(index, [0.82, 0.95], [0.01, 0.05], 0.28, [0.90, 0.99]),
      fact: fact(`dataset run ${n}`, "targets_rows", "10000 gated rows", text, 0.96),
      updates: [{ target_id: targetId, relationship: "replaces", reason: "New target supersedes the prior 1k compatibility-row target." }],
      reasoning: "Clear replacement should update the older memory."
    }));
  });
}

function generateFlagConflictRows(count) {
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const targetId = `deploy-lane-${n}`;
    const oldContent = `Deployment lane ${n} does not require VPN access.`;
    const text = `I may have been wrong; deployment lane ${n} might still require VPN access.`;
    const content = `Deployment lane ${n} may still require VPN access.`;
    return example(`retention-conflict-${n}`, sourceKind(index + 5000), {
      current_turn: turn("User", text, timestamp(index)),
      memory_store: [memoryStoreItem(targetId, oldContent, ["deployment", "vpn"], "semantic")]
    }, storeMemory({
      action: "flag_conflict",
      type: "semantic",
      content,
      tags: ["retention_flag_conflict", "possible_conflict", "vpn", "deployment", `lane_${n}`],
      scores: score(index, [0.55, 0.75], [0.04, 0.12], 0.22, [0.55, 0.75]),
      fact: fact(`deployment lane ${n}`, "may_require", "VPN access", text, 0.67),
      conflicts: [{ target_id: targetId, reason: "New statement is uncertain and conflicts with prior no-VPN memory." }],
      reasoning: "Uncertain contradiction should be flagged without overwriting the prior memory."
    }));
  });
}

function generateFlagAndStoreRows(count) {
  return Array.from({ length: count }, (_, index) => {
    const n = index + 1;
    const targetId = `checkpoint-lane-${n}`;
    const oldContent = `Evaluation lane ${n} uses checkpoint-last.pt.`;
    const text = `No, evaluation lane ${n} must use checkpoint-best.pt, not checkpoint-last.pt.`;
    const content = `Evaluation lane ${n} must use checkpoint-best.pt instead of checkpoint-last.pt.`;
    return example(`retention-flag-store-${n}`, sourceKind(index + 6000), {
      current_turn: turn("User", text, timestamp(index)),
      memory_store: [memoryStoreItem(targetId, oldContent, ["checkpoint", "evaluation"], "semantic")]
    }, storeMemory({
      action: "flag_and_store",
      type: "semantic",
      content,
      tags: ["retention_flag_and_store", "correction", "checkpoint", "durable", `lane_${n}`],
      scores: score(index, [0.85, 0.98], [0.005, 0.03], 0.36, [0.92, 0.99]),
      fact: fact(`evaluation lane ${n}`, "uses_checkpoint", "checkpoint-best.pt", text, 0.97),
      conflicts: [{ target_id: targetId, reason: "Explicit correction conflicts with the prior checkpoint-last memory." }],
      reasoning: "Clear durable correction should be stored and the old memory should be marked as conflicting."
    }));
  });
}

function generateArchiveCandidateRows(count) {
  return generateNormalEpisodicRows(count).map((row) => ({
    ...row,
    id: row.id.replace("normal-episodic", "archive-candidate"),
    output: {
      ...row.output,
      memory: {
        ...row.output.memory,
        tags: unique([...row.output.memory.tags, "archive_candidate", "historical"])
      }
    }
  }));
}

function example(id, sourceKindValue, input, output) {
  return {
    id,
    instruction,
    input: {
      operation: "remember",
      source_kind: sourceKindValue,
      source_id: id,
      prior_context: [],
      memory_store: [],
      ...input
    },
    output
  };
}

function ignore(reasoning) {
  return { action: "ignore", memory: null, facts: [], indexables: [], updates: [], conflicts: [], reasoning };
}

function storeMemory({ action, type, content, tags, scores, fact: factPayload, updates = [], conflicts = [], reasoning }) {
  const memory = {
    content: compact(content),
    type,
    strength: scores.strength,
    decay_rate: scores.decay_rate,
    emotional_weight: scores.emotional_weight,
    confidence: scores.confidence,
    tags: unique(tags.map(keyTag).filter(Boolean)).slice(0, 10)
  };
  return {
    action,
    memory,
    facts: factPayload ? [factPayload] : [],
    indexables: buildIndexables(memory.content, memory.tags, type),
    updates,
    conflicts,
    reasoning
  };
}

function fact(subject, predicate, value, evidenceText, confidence) {
  return {
    subject: compact(subject, 120),
    predicate: snake(predicate),
    value: compact(value, 160),
    confidence: round(clamp(confidence, 0.5, 0.99)),
    inference_kind: "explicit",
    evidence_text: compact(evidenceText, 220)
  };
}

function memoryStoreItem(id, content, tags, targetType) {
  const cleanContent = compact(content);
  return {
    id,
    source_id: id,
    source_timestamp: "",
    speaker: "User",
    content: cleanContent,
    tags: tags.map(keyTag),
    indexables: buildIndexables(cleanContent, tags, targetType).map((item) => ({ ...item, target_id: id }))
  };
}

function buildIndexables(content, tags = [], targetType = "semantic") {
  const tagTokens = tags
    .filter((tag) => !tag.startsWith("retention_"))
    .flatMap(tokenize)
    .filter((token) => !["durable", "temporary", "correction", "replacement"].includes(token));
  const contentTokens = tokenize(content);
  const numericTokens = contentTokens.filter((token) => /^\d+$/.test(token)).reverse();
  const descriptiveTokens = contentTokens.filter((token) => !/^\d+$/.test(token) && !["lane", "run", "row"].includes(token));
  const tokens = unique([...tagTokens.slice(0, 2), ...descriptiveTokens.slice(0, 2), ...numericTokens, ...descriptiveTokens.slice(2)]).slice(0, 5);
  const key = tokens.join("-") || "retention-anchor";
  return [{
    kind: "mnemonic",
    key,
    target_type: targetType,
    target_id: "",
    salience: 0.86,
    reconstructive_hint: compact(content, 180),
    evidence_text: compact(content, 220),
    tags: tags.slice(0, 8)
  }];
}

function score(index, strengthRange, decayRange, emotionalRange, confidenceRange) {
  return {
    strength: ranged(index, strengthRange),
    decay_rate: ranged(index + 3, decayRange),
    emotional_weight: Array.isArray(emotionalRange) ? ranged(index + 5, emotionalRange) : round(emotionalRange),
    confidence: ranged(index + 7, confidenceRange)
  };
}

function shortLivedEmotion(index) {
  return [0.08, 0.20, 0.06, 0.08, 0.12][index % 5];
}

function durableEmotion(index) {
  return [0.24, 0.38, 0.32, 0.22, 0.16][index % 5];
}

function criticalEmotion(index) {
  return [0.65, 0.55, 0.45, 0.58][index % 4];
}

function ranged(index, [min, max]) {
  const step = (index % 17) / 16;
  return round(min + (max - min) * step);
}

function turn(speaker, text, timestampValue) {
  return { speaker, text, timestamp: timestampValue };
}

function shortLivedContent(text, n) {
  if (text.includes("checkpoint-temp")) return `Temporary run ${n} uses checkpoint-temp-${n}.pt just for tonight.`;
  if (text.includes("throwaway API key")) return `Smoke test ${n} uses a throwaway API key only for the current test.`;
  if (text.includes("Colab runtime")) return `Current Colab runtime for lane ${n} has 12GB RAM available.`;
  if (text.includes("temporary log path")) return `Today's debugging lane ${n} uses temporary log path C:/tmp/psm-${n}.log.`;
  return `Training run ${n} is waiting on validation upload before the next command.`;
}

function sourceKind(index) {
  const bucket = index % 20;
  if (bucket < 9) return "synthetic_retention";
  if (bucket < 13) return "local_psm_project_state";
  if (bucket < 16) return "user_preference_564k_reviewed";
  if (bucket < 18) return "realtalk_recall_noise";
  return "prior_reviewed_incremental_patterns";
}

function subjectFor(content) {
  const match = content.match(/^(.*?)(?: should| must| runs| targets| uses| is)/i);
  return match?.[1] ?? content.split(" ").slice(0, 5).join(" ");
}

function scaleCounts(baseCounts, desiredTotal) {
  const baseTotal = Object.values(baseCounts).reduce((sum, value) => sum + value, 0);
  if (baseTotal === desiredTotal) return baseCounts;
  const entries = Object.entries(baseCounts).map(([key, value]) => [key, Math.floor((value / baseTotal) * desiredTotal)]);
  let remaining = desiredTotal - entries.reduce((sum, [, value]) => sum + value, 0);
  for (let i = 0; remaining > 0; i++, remaining--) entries[i % entries.length][1]++;
  return Object.fromEntries(entries);
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

function countBy(rows, getKey) {
  const counts = {};
  for (const row of rows) {
    const key = getKey(row);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function variant(templates, index, suffix) {
  return `${templates[index % templates.length]}${suffix}`;
}

function timestamp(index) {
  return `2026-05-${padDay(index + 1)}T10:00:00Z`;
}

function padDay(value) {
  return String(((value - 1) % 28) + 1).padStart(2, "0");
}

function compact(value, max = 220) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length <= max ? text : `${text.slice(0, max - 3).trim()}...`;
}

function keyTag(value) {
  return snake(value).replace(/^_+|_+$/g, "");
}

function snake(value) {
  return String(value ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "value";
}

function tokenize(value) {
  const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "into", "user", "memory", "current", "should", "must"]);
  return String(value ?? "").toLowerCase().match(/[a-z0-9]+/g)?.filter((token) => (/^\d+$/.test(token) || token.length > 2) && !stop.has(token)) ?? [];
}

function unique(items) {
  return [...new Set(items)];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round(value) {
  return Number(value.toFixed(3));
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
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function numberArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isFinite(value) && value > 0 && value < 1 ? value : fallback;
}
