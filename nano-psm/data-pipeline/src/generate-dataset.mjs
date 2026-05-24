#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";

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
const positional = args._;
const locomoPath = stringArg(args, "locomo", positional[0] ?? "benchmark/locomo/data/locomo10.json");
const outDir = stringArg(args, "out", positional[1] ?? "nano-psm/data-pipeline/data/generated");
const limit = intArg(args, "limit", intValue(positional[2], 500));
const recallLimit = intArg(args, "recall-limit", Math.max(50, Math.floor(limit / 2)));
const syntheticCount = intArg(args, "synthetic-count", 200);
const validationRatio = numberArg(args, "validation-ratio", 0.15);

const locomoExamples = generateLocomoExamples(locomoPath, limit);
const locomoRecallExamples = generateLocomoRecallExamples(locomoPath, recallLimit);
const developerExamples = generateDeveloperExamples();
const syntheticIndexableExamples = generateSyntheticIndexableExamples(syntheticCount);
const hardNegativeExamples = generateHardNegativeExamples();
const examples = [...locomoExamples, ...locomoRecallExamples, ...developerExamples, ...syntheticIndexableExamples, ...hardNegativeExamples]
  .map(normalizeTrainingExample)
  .map((example, index) => ({ ...example, id: example.id ?? `example-${index + 1}` }));

const { train, validation } = splitDeterministically(examples, validationRatio);
mkdirSync(outDir, { recursive: true });
writeJsonl(join(outDir, "all.jsonl"), examples);
writeJsonl(join(outDir, "train.jsonl"), train);
writeJsonl(join(outDir, "validation.jsonl"), validation);
writeFileSync(join(outDir, "metadata.json"), JSON.stringify({
  generated_at: new Date().toISOString(),
  locomo: locomoPath,
  locomo_examples: locomoExamples.length,
  locomo_recall_examples: locomoRecallExamples.length,
  developer_examples: developerExamples.length,
  synthetic_indexable_examples: syntheticIndexableExamples.length,
  hard_negative_examples: hardNegativeExamples.length,
  total_examples: examples.length,
  train_examples: train.length,
  validation_examples: validation.length,
  validation_ratio: validationRatio
}, null, 2), "utf8");

console.log(JSON.stringify({
  out_dir: outDir,
  total: examples.length,
  train: train.length,
  validation: validation.length
}, null, 2));

function generateLocomoExamples(path, maxExamples) {
  const samples = JSON.parse(readFileSync(path, "utf8"));
  const examples = [];
  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? basename(path, ".json"));
    const turns = flattenTurns(sample);
    for (let index = 0; index < turns.length && examples.length < maxExamples; index++) {
      const turn = turns[index];
      const priorContext = turns.slice(Math.max(0, index - 2), index).map((prior) => ({
        speaker: String(prior.speaker ?? "Unknown"),
        text: String(prior.text ?? ""),
        dia_id: String(prior.dia_id ?? ""),
        session: String(prior.session ?? "")
      }));
      const sessionTimestamp = sessionTimestampFor(sample, turn.session);
      const sourceId = `${sampleId}:${turn.dia_id ?? `turn-${index + 1}`}`;
      examples.push({
        id: `locomo-${sourceId}`,
        instruction,
        input: {
          operation: "remember",
          source_kind: "locomo",
          source_id: sourceId,
          current_turn: {
            speaker: String(turn.speaker ?? "Unknown"),
            text: String(turn.text ?? ""),
            dia_id: String(turn.dia_id ?? ""),
            session: String(turn.session ?? ""),
            timestamp: sessionTimestamp,
            image_query: turn.query ? String(turn.query) : undefined,
            image_caption: turn.blip_caption ? String(turn.blip_caption) : undefined
          },
          prior_context: priorContext,
          memory_store: []
        },
        output: outputForLocomoTurn(turn, sessionTimestamp)
      });
    }
    if (examples.length >= maxExamples) break;
  }
  return examples;
}

function generateLocomoRecallExamples(path, maxExamples) {
  const samples = JSON.parse(readFileSync(path, "utf8"));
  const examples = [];
  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? basename(path, ".json"));
    const turns = flattenTurns(sample);
    const byDiaId = new Map(turns.map((turn) => [String(turn.dia_id ?? ""), turn]));
    const qaItems = Array.isArray(sample.qa) ? sample.qa : [];
    for (const qa of qaItems) {
      if (examples.length >= maxExamples) break;
      const question = typeof qa.question === "string" ? qa.question.trim() : "";
      const evidenceIds = Array.isArray(qa.evidence) ? qa.evidence.map(String).filter(Boolean) : [];
      const evidenceTurns = evidenceIds.map((id) => byDiaId.get(id)).filter(Boolean);
      if (!question || evidenceTurns.length === 0) continue;
      const sourceIds = new Set(evidenceIds.map((id) => `${sampleId}:${id}`));
      const distractors = turns
        .filter((turn) => !sourceIds.has(`${sampleId}:${turn.dia_id ?? ""}`))
        .filter((turn) => !isIgnorableTurn(cleanText(turn.text)))
        .slice(0, Math.max(4, evidenceTurns.length * 2));
      const memoryStore = [...evidenceTurns, ...distractors]
        .map((turn) => locomoMemoryStoreItem(sample, sampleId, turn))
        .filter(Boolean);
      const selected = memoryStore.filter((memory) => sourceIds.has(memory.source_id));
      if (selected.length === 0) continue;
      examples.push({
        id: `locomo-recall-${sampleId}-${examples.length + 1}`,
        instruction,
        input: {
          operation: "recall",
          source_kind: "locomo_qa",
          current_query: {
            question,
            category: qa.category != null ? String(qa.category) : ""
          },
          memory_store: memoryStore
        },
        output: {
          action: "recall_context",
          memory: null,
          facts: [],
          indexables: [],
          updates: [],
          conflicts: [],
          recall: {
            query_intent: recallIntent(question),
            selected_memory_ids: selected.map((memory) => memory.id),
            selected_indexable_keys: selected.flatMap((memory) => memory.indexables.map((indexable) => indexable.key)).slice(0, 6),
            max_items: Math.min(5, selected.length),
            reasoning: "Selected only memory rows that are grounded in the LOCOMO evidence ids."
          },
          reasoning: "Recall should route through evidence-backed memory rows and their mnemonic indexables."
        }
      });
    }
    if (examples.length >= maxExamples) break;
  }
  return examples;
}

function locomoMemoryStoreItem(sample, sampleId, turn) {
  const sourceId = `${sampleId}:${turn.dia_id ?? ""}`;
  const timestamp = sessionTimestampFor(sample, turn.session);
  const output = outputForLocomoTurn(turn, timestamp);
  const content = output.memory?.content ?? `${cleanSpeaker(turn.speaker)} said: ${cleanText(turn.text)}`;
  if (!content.trim()) return null;
  const tags = output.memory?.tags ?? [cleanSpeaker(turn.speaker).toLowerCase()];
  return {
    id: sourceId,
    source_id: sourceId,
    source_timestamp: timestamp,
    speaker: cleanSpeaker(turn.speaker),
    content,
    tags,
    indexables: buildIndexables({
      content,
      tags,
      facts: output.facts ?? [],
      source_id: sourceId,
      target_type: output.memory?.type ?? "episodic"
    })
  };
}

function outputForLocomoTurn(turn, sessionTimestamp) {
  const speaker = cleanSpeaker(turn.speaker);
  const text = cleanText(turn.text);
  const lower = text.toLowerCase();
  if (isIgnorableTurn(text)) {
    return ignore("Greeting, acknowledgement, question, or logistics only; no durable memory.");
  }

  if (lower.includes("support group")) {
    const temporal = lower.includes("yesterday") ? resolveRelativeDay("yesterday", sessionTimestamp) : {};
    return storeEpisodic({
      content: `${speaker} attended an LGBTQ support group${temporal.resolved_time ? ` on ${temporal.resolved_time}` : ""} and found it powerful.`,
      tags: ["support_group", "lgbtq", "wellbeing"],
      emotional_weight: 0.75,
      temporal_expression: temporal.temporal_expression,
      resolved_time: temporal.resolved_time,
      resolved_time_confidence: temporal.resolved_time_confidence,
      reasoning: "Specific personal event with explicit emotional significance."
    });
  }

  if (lower.includes("transgender stories")) {
    return storeEpisodic({
      content: `${speaker} felt happy and thankful after hearing inspiring transgender stories and receiving support.`,
      tags: ["support_group", "transgender", "gratitude"],
      emotional_weight: 0.85,
      reasoning: "Explicit emotional reaction to a specific support-group experience."
    });
  }

  if (lower.includes("accepted") && lower.includes("courage")) {
    return storeEpisodic({
      content: `${speaker} said the support group helped them feel accepted and gave them courage to embrace themself.`,
      tags: ["support_group", "self_acceptance", "confidence"],
      emotional_weight: 0.9,
      reasoning: "Directly stated personal impact from a concrete event."
    });
  }

  if (lower.includes("career options") || lower.includes("continue my edu")) {
    return storeEpisodic({
      content: `${speaker} plans to continue education and explore career options.`,
      tags: ["education", "career_planning"],
      emotional_weight: 0.45,
      reasoning: "Specific stated plan about education and career direction."
    });
  }

  if (lower.includes("counseling") || lower.includes("mental health")) {
    const facts = [{
      subject: speaker,
      predicate: "interested_in",
      value: "counseling or mental health work",
      confidence: 0.94,
      inference_kind: "explicit",
      evidence_text: text
    }];
    return storeSemantic({
      content: `${speaker} is interested in counseling or mental health work to support people with similar issues.`,
      tags: ["career_interest", "counseling", "mental_health"],
      emotional_weight: 0.55,
      facts,
      reasoning: "Stable career interest explicitly stated by the speaker."
    });
  }

  if (lower.includes("painted") && (lower.includes("lake") || lower.includes("sunrise"))) {
    const temporal = lower.includes("last year") ? { temporal_expression: "last year" } : {};
    return storeEpisodic({
      content: `${speaker} painted a lake sunrise scene last year and considers it special.`,
      tags: ["painting", "artwork", "lake_sunrise"],
      emotional_weight: 0.6,
      temporal_expression: temporal.temporal_expression,
      reasoning: "Specific artwork with explicit personal significance."
    });
  }

  if (lower.includes("painting") && lower.includes("express")) {
    return storeSemantic({
      content: `${speaker} sees painting as a way to express feelings, be creative, and relax after a long day.`,
      tags: ["painting", "creative_expression", "relaxation"],
      emotional_weight: 0.55,
      facts: [{
        subject: speaker,
        predicate: "uses_painting_for",
        value: "expressing feelings, creativity, and relaxation",
        confidence: 0.94,
        inference_kind: "explicit",
        evidence_text: text
      }],
      reasoning: "Stable hobby preference directly stated by the speaker."
    });
  }

  if (lower.includes("swimming with the kids")) {
    return storeEpisodic({
      content: `${speaker} planned to go swimming with the kids after the conversation.`,
      tags: ["family", "swimming", "self_care"],
      emotional_weight: 0.35,
      reasoning: "Concrete personal plan involving family and self-care."
    });
  }

  if (lower.includes("charity race")) {
    const temporal = lower.includes("last saturday") ? { temporal_expression: "last Saturday" } : {};
    return storeEpisodic({
      content: `${speaker} ran a charity race for mental health and found it rewarding.`,
      tags: ["charity_race", "mental_health", "recent_activity"],
      emotional_weight: 0.7,
      temporal_expression: temporal.temporal_expression,
      facts: [{
        subject: speaker,
        predicate: "participated_in",
        value: "charity race for mental health",
        confidence: 0.95,
        inference_kind: "explicit",
        evidence_text: text,
        temporal_expression: temporal.temporal_expression
      }],
      reasoning: "Specific event with explicit topic and emotional impact."
    });
  }

  return ignore("No durable memory can be extracted without over-inference.");
}

function generateDeveloperExamples() {
  return [
    example("dev-ignore-1", {
      current_turn: { speaker: "User", text: "ok thanks, let's do that after lunch", timestamp: "2026-05-20T10:00:00Z" },
      prior_context: [],
      memory_store: []
    }, ignore("Short logistics message; no durable project or preference memory.")),
    example("dev-episodic-1", {
      current_turn: { speaker: "User", text: "The LOCOMO Q4 smoke stored 8, ignored 0, and failed 12 out of 20 because JSON kept breaking.", timestamp: "2026-05-20T12:13:41Z" },
      prior_context: [],
      memory_store: []
    }, storeEpisodic({
      content: "On 2026-05-20, the LOCOMO Q4 smoke stored 8 turns, ignored 0, and failed 12 of 20 due to JSON failures.",
      tags: ["locomo", "q4_k_m", "json_failure", "benchmark_result"],
      temporal_expression: "2026-05-20",
      resolved_time: "2026-05-20",
      resolved_time_confidence: 1,
      facts: [{
        subject: "LOCOMO Q4 smoke",
        predicate: "failed_count",
        value: "12 of 20",
        confidence: 0.99,
        inference_kind: "explicit",
        evidence_text: "stored 8, ignored 0, and failed 12 out of 20"
      }],
      reasoning: "Concrete benchmark result with counts and cause."
    })),
    example("dev-episodic-2", {
      current_turn: { speaker: "User", text: "F16 had zero JSON failures on the same 20-turn smoke, but it still produced generic User and unsupported facts.", timestamp: "2026-05-20T14:41:39Z" },
      prior_context: [],
      memory_store: []
    }, storeEpisodic({
      content: "On the same 20-turn LOCOMO smoke, F16 had zero JSON failures but still produced generic User references and unsupported facts.",
      tags: ["locomo", "f16", "json_validity", "quality_issue"],
      temporal_expression: "same 20-turn smoke",
      facts: [{
        subject: "F16 LOCOMO smoke",
        predicate: "json_failures",
        value: "0",
        confidence: 0.99,
        inference_kind: "explicit",
        evidence_text: "F16 had zero JSON failures"
      }],
      reasoning: "Concrete comparison result and remaining quality issues."
    })),
    example("dev-semantic-1", {
      current_turn: { speaker: "User", text: "I prefer incremental version upgrades: version packages, build, then commit.", timestamp: "2026-05-17T13:10:52Z" },
      prior_context: [],
      memory_store: []
    }, storeSemantic({
      content: "User prefers incremental version upgrades with package versioning, build verification, and then commit.",
      tags: ["workflow_preference", "release_process", "versioning"],
      facts: [{
        subject: "User",
        predicate: "prefers_release_flow",
        value: "version packages, build, then commit",
        confidence: 0.98,
        inference_kind: "explicit",
        evidence_text: "I prefer incremental version upgrades: version packages, build, then commit."
      }],
      reasoning: "Explicit stable workflow preference."
    })),
    example("dev-update-1", {
      current_turn: { speaker: "User", text: "Actually don't use huggingface-cli anymore; use hf directly in the notebook.", timestamp: "2026-05-20T11:20:00Z" },
      prior_context: [],
      memory_store: [{ id: "m1", content: "LOCOMO notebook uploads artifacts with huggingface-cli." }]
    }, {
      action: "update_existing",
      memory: {
        content: "LOCOMO notebooks should use the `hf` CLI directly instead of deprecated `huggingface-cli`.",
        type: "semantic",
        strength: 0.86,
        decay_rate: 0.02,
        emotional_weight: 0.2,
        confidence: 0.96,
        tags: ["hugging_face", "colab", "cli_update"]
      },
      facts: [{
        subject: "LOCOMO notebook",
        predicate: "uses_cli",
        value: "hf",
        confidence: 0.96,
        inference_kind: "explicit",
        evidence_text: "use hf directly in the notebook"
      }],
      updates: [{ target_id: "m1", relationship: "replaces", reason: "huggingface-cli is deprecated" }],
      conflicts: [],
      reasoning: "New instruction explicitly replaces an older CLI command."
    }),
    example("dev-conflict-1", {
      current_turn: { speaker: "User", text: "We cannot assume user machines have GPU; many office VMs are CPU-only.", timestamp: "2026-05-20T10:30:00Z" },
      prior_context: [],
      memory_store: [{ id: "m2", content: "PSM can rely on local GPU acceleration for ingestion." }]
    }, {
      action: "flag_and_store",
      memory: {
        content: "Do not assume PSM users have GPU access; many office VMs are CPU-only.",
        type: "semantic",
        strength: 0.9,
        decay_rate: 0.01,
        emotional_weight: 0.45,
        confidence: 0.95,
        tags: ["hardware_constraint", "cpu_only", "office_vm"]
      },
      facts: [{
        subject: "office VMs",
        predicate: "may_be",
        value: "CPU-only",
        confidence: 0.92,
        inference_kind: "explicit",
        evidence_text: "many office VMs are CPU-only"
      }],
      updates: [],
      conflicts: [{ target_id: "m2", reason: "New information says GPU access cannot be assumed." }],
      reasoning: "Important product constraint that conflicts with prior GPU assumption."
    })
  ];
}

function generateSyntheticIndexableExamples(count) {
  const templates = [
    (i) => example(`synthetic-project-decision-${i}`, {
      current_turn: { speaker: "User", text: `For release ${i}, keep the CLI installer local-first and avoid admin-only setup steps.`, timestamp: `2026-05-${padDay(i)}T10:00:00Z` },
      prior_context: [],
      memory_store: []
    }, storeSemantic({
      content: `For release ${i}, the CLI installer should stay local-first and avoid admin-only setup steps.`,
      tags: ["release_process", "cli_installer", "local_first"],
      facts: [{
        subject: `release ${i} CLI installer`,
        predicate: "should_avoid",
        value: "admin-only setup steps",
        confidence: 0.96,
        inference_kind: "explicit",
        evidence_text: "avoid admin-only setup steps"
      }],
      reasoning: "Explicit stable product constraint for installer design."
    })),
    (i) => example(`synthetic-temporal-benchmark-${i}`, {
      current_turn: { speaker: "User", text: `Yesterday's memory smoke ${i} stored ${10 + (i % 7)}, ignored ${i % 3}, and failed ${2 + (i % 5)} because temporal facts were missing.`, timestamp: `2026-05-${padDay(i)}T15:00:00Z` },
      prior_context: [],
      memory_store: []
    }, storeEpisodic({
      content: `Memory smoke ${i} had missing temporal facts yesterday, with ${10 + (i % 7)} stored, ${i % 3} ignored, and ${2 + (i % 5)} failed.`,
      tags: ["benchmark_result", "temporal_fact", "smoke_test"],
      temporal_expression: "yesterday",
      resolved_time: previousDay(`2026-05-${padDay(i)}T15:00:00Z`),
      resolved_time_confidence: 0.9,
      facts: [{
        subject: `memory smoke ${i}`,
        predicate: "failure_cause",
        value: "missing temporal facts",
        confidence: 0.95,
        inference_kind: "explicit",
        evidence_text: "failed because temporal facts were missing",
        temporal_expression: "yesterday",
        resolved_time: previousDay(`2026-05-${padDay(i)}T15:00:00Z`)
      }],
      emotional_weight: 0.35,
      reasoning: "Concrete dated benchmark result with explicit counts and cause."
    })),
    (i) => example(`synthetic-user-preference-${i}`, {
      current_turn: { speaker: "Asha", text: `I remember things best with short cue words like migration-ladder-${i}, not long transcripts.`, timestamp: `2026-06-${padDay(i)}T09:30:00Z` },
      prior_context: [],
      memory_store: []
    }, storeSemantic({
      content: "Asha prefers short mnemonic cue words over long transcripts for memory recall.",
      tags: ["memory_preference", "mnemonic_cues", "indexables"],
      facts: [{
        subject: "Asha",
        predicate: "prefers_memory_cues",
        value: "short mnemonic cue words",
        confidence: 0.97,
        inference_kind: "explicit",
        evidence_text: "I remember things best with short cue words"
      }],
      indexables: [{
        kind: "mnemonic",
        key: `migration-ladder-${i}`,
        target_type: "semantic",
        target_id: "",
        salience: 0.92,
        reconstructive_hint: "Asha prefers compact cue words as recall handles instead of long transcripts.",
        evidence_text: `short cue words like migration-ladder-${i}`,
        tags: ["memory_preference", "mnemonic_cues", "indexables"]
      }],
      reasoning: "Explicit preference about mnemonic memory organization."
    })),
    (i) => example(`synthetic-update-${i}`, {
      current_turn: { speaker: "User", text: `Correction: project ${i} now uses Postgres for shared memory, not SQLite.`, timestamp: `2026-07-${padDay(i)}T11:00:00Z` },
      prior_context: [],
      memory_store: [{ id: `old-db-${i}`, content: `Project ${i} uses SQLite for shared memory.` }]
    }, {
      action: "update_existing",
      memory: {
        content: `Project ${i} now uses Postgres for shared memory instead of SQLite.`,
        type: "semantic",
        strength: 0.88,
        decay_rate: 0.02,
        emotional_weight: 0.25,
        confidence: 0.97,
        tags: ["database_change", "postgres", "shared_memory"]
      },
      facts: [{
        subject: `project ${i}`,
        predicate: "uses_database",
        value: "Postgres",
        confidence: 0.97,
        inference_kind: "explicit",
        evidence_text: "now uses Postgres for shared memory"
      }],
      indexables: buildIndexables({
        content: `Project ${i} now uses Postgres for shared memory instead of SQLite.`,
        tags: ["database_change", "postgres", "shared_memory"],
        facts: [],
        target_type: "semantic"
      }),
      updates: [{ target_id: `old-db-${i}`, relationship: "replaces", reason: "User corrected the database choice." }],
      conflicts: [],
      reasoning: "Explicit correction replaces the prior SQLite memory."
    }),
    (i) => example(`synthetic-conflict-${i}`, {
      current_turn: { speaker: "User", text: `I may have been wrong: deployment lane ${i} might still require VPN access.`, timestamp: `2026-08-${padDay(i)}T13:00:00Z` },
      prior_context: [],
      memory_store: [{ id: `deploy-${i}`, content: `Deployment lane ${i} does not require VPN access.` }]
    }, {
      action: "flag_conflict",
      memory: {
        content: `Deployment lane ${i} may still require VPN access.`,
        type: "semantic",
        strength: 0.72,
        decay_rate: 0.04,
        emotional_weight: 0.3,
        confidence: 0.68,
        tags: ["deployment", "vpn", "possible_conflict"]
      },
      facts: [{
        subject: `deployment lane ${i}`,
        predicate: "may_require",
        value: "VPN access",
        confidence: 0.68,
        inference_kind: "explicit",
        evidence_text: "might still require VPN access"
      }],
      indexables: buildIndexables({
        content: `Deployment lane ${i} may still require VPN access.`,
        tags: ["deployment", "vpn", "possible_conflict"],
        facts: [],
        target_type: "semantic"
      }),
      updates: [],
      conflicts: [{ target_id: `deploy-${i}`, reason: "New statement weakly contradicts prior no-VPN memory." }],
      reasoning: "The correction is uncertain, so flag conflict rather than fully replacing the old memory."
    })
  ];
  const examples = [];
  for (let i = 1; i <= count; i++) {
    examples.push(templates[(i - 1) % templates.length](i));
  }
  return examples;
}

function generateHardNegativeExamples() {
  return [
    example("hard-negative-legacy-schema", {
      current_turn: { speaker: "Caroline", text: "Thanks, Melanie! That's really sweet. Is this your own painting?", timestamp: "2023-05-08T13:56:00" },
      prior_context: [],
      memory_store: []
    }, ignore("Compliment and question only; no durable memory. Do not emit operation or assistant_response.")),
    example("hard-negative-generic-user", {
      current_turn: { speaker: "Caroline", text: "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.", timestamp: "2023-05-08T13:56:00" },
      prior_context: [],
      memory_store: []
    }, storeSemantic({
      content: "Caroline is interested in counseling or mental health work to support people with similar issues.",
      tags: ["career_interest", "mental_health", "speaker_grounding"],
      facts: [{
        subject: "Caroline",
        predicate: "interested_in",
        value: "counseling or mental health work",
        confidence: 0.95,
        inference_kind: "explicit",
        evidence_text: "I'm keen on counseling or working in mental health"
      }],
      reasoning: "Speaker is known, so the memory must use Caroline rather than generic User."
    })),
    example("hard-negative-unsupported-facts", {
      current_turn: { speaker: "Melanie", text: "Painting's a fun way to express my feelings and get creative. It's a great way to relax after a long day.", timestamp: "2023-05-08T13:56:00" },
      prior_context: [],
      memory_store: []
    }, storeSemantic({
      content: "Melanie uses painting to express feelings, be creative, and relax after a long day.",
      tags: ["painting", "creative_expression", "relaxation"],
      facts: [{
        subject: "Melanie",
        predicate: "uses_painting_for",
        value: "expressing feelings and relaxing",
        confidence: 0.95,
        inference_kind: "explicit",
        evidence_text: "Painting's a fun way to express my feelings... a great way to relax after a long day."
      }],
      reasoning: "Only painting is supported; do not invent photography, reading, or other hobbies."
    }))
  ];
}

function example(id, input, output) {
  return {
    id,
    instruction,
    input: {
      operation: "remember",
      source_kind: input.source_kind ?? sourceKindForExampleId(id),
      source_id: input.source_id ?? id,
      ...input
    },
    output
  };
}

function sourceKindForExampleId(id) {
  if (id.startsWith("dev-")) return "local_psm_developer";
  if (id.startsWith("synthetic-")) return "synthetic";
  if (id.startsWith("hard-negative-")) return "synthetic_hard_negative";
  return "generated";
}

function ignore(reasoning) {
  return { action: "ignore", memory: null, facts: [], updates: [], conflicts: [], reasoning };
}

function storeEpisodic(options) {
  return storeMemory("store_episodic", "episodic", options);
}

function storeSemantic(options) {
  return storeMemory("promote_semantic", "semantic", options);
}

function storeMemory(action, type, options) {
  const memory = {
    content: options.content,
    type,
    strength: options.strength ?? 0.85,
    decay_rate: options.decay_rate ?? (type === "semantic" ? 0.02 : 0.04),
    emotional_weight: options.emotional_weight ?? 0.4,
    confidence: options.confidence ?? 0.9,
    tags: options.tags ?? []
  };
  for (const key of ["temporal_expression", "resolved_time", "resolved_time_confidence"]) {
    if (options[key] != null) memory[key] = options[key];
  }
  return {
    action,
    memory,
    facts: options.facts ?? [],
    indexables: options.indexables ?? buildIndexables({
      content: options.content,
      tags: options.tags ?? [],
      facts: options.facts ?? [],
      target_type: type
    }),
    updates: options.updates ?? [],
    conflicts: options.conflicts ?? [],
    reasoning: options.reasoning ?? "Durable memory directly supported by the current input."
  };
}

function normalizeTrainingExample(example) {
  const output = example.output ?? {};
  const normalized = {
    ...output,
    facts: Array.isArray(output.facts) ? output.facts : [],
    indexables: Array.isArray(output.indexables) ? output.indexables : [],
    updates: Array.isArray(output.updates) ? output.updates : [],
    conflicts: Array.isArray(output.conflicts) ? output.conflicts : [],
    reasoning: typeof output.reasoning === "string" ? output.reasoning : ""
  };
  if (normalized.memory && normalized.indexables.length === 0) {
    normalized.indexables = buildIndexables({
      content: normalized.memory.content,
      tags: normalized.memory.tags ?? [],
      facts: normalized.facts,
      target_type: normalized.memory.type
    });
  }
  if (normalized.action === "recall_context" && !normalized.recall) {
    normalized.recall = {
      query_intent: "recall",
      selected_memory_ids: [],
      selected_indexable_keys: [],
      max_items: 0,
      reasoning: "No grounded recall context selected."
    };
  }
  return { ...example, output: normalized };
}

function buildIndexables(input) {
  const content = cleanText(input.content);
  if (!content) return [];
  const key = mnemonicKey(content, input.tags ?? []);
  const secondary = factKey(input.facts ?? []);
  const base = {
    kind: "mnemonic",
    key,
    target_type: input.target_type ?? "memory",
    target_id: input.source_id ?? "",
    salience: salienceFor(content, input.tags ?? []),
    reconstructive_hint: reconstructiveHint(content),
    evidence_text: content,
    tags: (input.tags ?? []).slice(0, 6)
  };
  return secondary && secondary !== key
    ? [base, { ...base, kind: "fact_anchor", key: secondary, salience: Math.max(base.salience, 0.82) }]
    : [base];
}

function mnemonicKey(content, tags) {
  const tagTokens = tags.flatMap((tag) => meaningfulTokens(String(tag).replace(/_/g, " ")));
  const contentTokens = meaningfulTokens(content);
  const tokens = unique([...tagTokens, ...contentTokens]).slice(0, 4);
  return tokens.length > 0 ? tokens.join("-") : "memory-anchor";
}

function factKey(facts) {
  const fact = facts.find((item) => item && typeof item.subject === "string" && typeof item.predicate === "string" && typeof item.value === "string");
  if (!fact) return "";
  return unique([...meaningfulTokens(fact.subject), ...meaningfulTokens(fact.predicate), ...meaningfulTokens(fact.value)]).slice(0, 4).join("-");
}

function recallIntent(question) {
  const lower = question.toLowerCase();
  if (/\bwhen\b|\bdate\b|\byear\b|\bmonth\b|\btime\b/.test(lower)) return "temporal_recall";
  if (/\bwhat\b/.test(lower)) return "fact_recall";
  if (/\bwould\b|\blikely\b/.test(lower)) return "inference_supported_recall";
  return "memory_recall";
}

function salienceFor(content, tags) {
  const lower = content.toLowerCase();
  let score = 0.68;
  if (/\b\d{4}\b|yesterday|last week|last year|next month|june|july|may/.test(lower)) score += 0.08;
  if (/support|accepted|courage|powerful|rewarding|identity|career|mental health/.test(lower)) score += 0.12;
  if ((tags ?? []).length > 0) score += 0.04;
  return Number(Math.min(score, 0.98).toFixed(2));
}

function reconstructiveHint(content) {
  const sentence = content.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? content;
  return sentence.length <= 160 ? sentence : `${sentence.slice(0, 157).trim()}...`;
}

function meaningfulTokens(text) {
  const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "into", "said", "them", "they", "their", "memory", "tools", "local", "after"]);
  return cleanText(text).toLowerCase().match(/[a-z0-9]+/g)?.filter((token) => token.length > 2 && !stop.has(token)) ?? [];
}

function unique(items) {
  const seen = new Set();
  const result = [];
  for (const item of items) {
    if (seen.has(item)) continue;
    seen.add(item);
    result.push(item);
  }
  return result;
}

function flattenTurns(sample) {
  const conversation = sample.conversation ?? {};
  return Object.keys(conversation)
    .filter((key) => /^session_\d+$/.test(key))
    .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]))
    .flatMap((session) => {
      const turns = conversation[session];
      return Array.isArray(turns) ? turns.map((turn) => ({ ...turn, session })) : [];
    });
}

function sessionTimestampFor(sample, session) {
  const conversation = sample.conversation ?? {};
  const raw = conversation[`${session}_date_time`];
  return typeof raw === "string" ? raw : "";
}

function isIgnorableTurn(text) {
  const lower = text.toLowerCase().trim();
  if (!lower) return true;
  if (/^(hey|hi|hello)\b/.test(lower) && lower.length < 90) return true;
  if (lower.endsWith("?") && !lower.includes("what happened to me")) return true;
  if (lower.includes("thanks") && lower.length < 110 && !lower.includes("support group")) return true;
  if (lower.includes("proud of you") && lower.length < 140) return true;
  return false;
}

function cleanSpeaker(value) {
  return typeof value === "string" && value.trim() ? value.trim() : "Unknown";
}

function cleanText(value) {
  return typeof value === "string" ? value.trim().replace(/\s+/g, " ") : "";
}

function resolveRelativeDay(expression, timestamp) {
  const parsed = parseLooseDate(timestamp);
  if (!parsed) return { temporal_expression: expression };
  const date = new Date(parsed.getTime());
  if (expression.toLowerCase() === "yesterday") date.setUTCDate(date.getUTCDate() - 1);
  return {
    temporal_expression: expression,
    resolved_time: date.toISOString().slice(0, 10),
    resolved_time_confidence: 0.9
  };
}

function parseLooseDate(value) {
  if (!value) return null;
  const direct = Date.parse(value);
  if (!Number.isNaN(direct)) return new Date(direct);
  const timeOnDay = value.match(/(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2})\s+([A-Z][a-z]+),?\s+(\d{4})/i);
  if (timeOnDay) {
    const [, time, day, month, year] = timeOnDay;
    const parsed = Date.parse(`${month} ${day}, ${year} ${time}`);
    if (!Number.isNaN(parsed)) return new Date(parsed);
  }
  const normalized = value.replace(/(\d{1,2}) ([A-Z][a-z]+),? (\d{4})/g, "$2 $1, $3");
  const parsed = Date.parse(normalized);
  return Number.isNaN(parsed) ? null : new Date(parsed);
}

function padDay(value) {
  return String(((value - 1) % 28) + 1).padStart(2, "0");
}

function previousDay(timestamp) {
  const parsed = new Date(timestamp);
  parsed.setUTCDate(parsed.getUTCDate() - 1);
  return parsed.toISOString().slice(0, 10);
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

function writeJsonl(path, rows) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
}

function parseArgs(argv) {
  const parsed = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      parsed._.push(arg);
      continue;
    }
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) parsed[key] = "true";
    else parsed[key] = argv[++i];
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

function intValue(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

function numberArg(parsed, key, fallback) {
  const value = Number(parsed[key]);
  return Number.isFinite(value) && value > 0 && value < 1 ? value : fallback;
}
