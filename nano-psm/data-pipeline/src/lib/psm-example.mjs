export const instruction = [
  "Perform the PSM memory operation for the current input.",
  "Return JSON only using the target schema.",
  "Do not use legacy keys such as operation or assistant_response.",
  "Do not write generic User when a speaker name is available.",
  "Only extract facts that are explicitly supported by evidence_text.",
  "Create compact indexables for stored memories so later recall can use mnemonic cues.",
  "For recall inputs, select grounded memory ids and indexable keys; do not answer from general knowledge."
].join(" ");

export function createRememberExample(id, input, output) {
  return normalizeTrainingExample({
    id,
    instruction,
    input: {
      operation: "remember",
      prior_context: [],
      memory_store: [],
      ...input
    },
    output
  });
}

export function createRecallExample(id, input, output) {
  return normalizeTrainingExample({
    id,
    instruction,
    input: {
      operation: "recall",
      memory_store: [],
      ...input
    },
    output: {
      action: "recall_context",
      memory: null,
      facts: [],
      indexables: [],
      updates: [],
      conflicts: [],
      ...output
    }
  });
}

export function storeMemoryOutput(action, type, options) {
  const memory = {
    content: cleanText(options.content),
    type,
    strength: clamp01(options.strength ?? (type === "semantic" ? 0.85 : 0.78)),
    decay_rate: clamp01(options.decay_rate ?? (type === "semantic" ? 0.02 : 0.04)),
    emotional_weight: clamp01(options.emotional_weight ?? 0.35),
    confidence: clamp01(options.confidence ?? 0.88),
    tags: normalizeTags(options.tags ?? [])
  };
  for (const key of ["temporal_expression", "resolved_time", "resolved_time_confidence"]) {
    if (options[key] != null) memory[key] = options[key];
  }
  const facts = normalizeFacts(options.facts ?? []);
  return {
    action,
    memory,
    facts,
    indexables: options.indexables ?? buildIndexables({
      content: memory.content,
      tags: memory.tags,
      facts,
      target_type: type,
      target_id: options.target_id
    }),
    updates: options.updates ?? [],
    conflicts: options.conflicts ?? [],
    reasoning: options.reasoning ?? "Durable memory directly supported by source evidence."
  };
}

export function ignoreOutput(reasoning) {
  return {
    action: "ignore",
    memory: null,
    facts: [],
    indexables: [],
    updates: [],
    conflicts: [],
    reasoning
  };
}

export function normalizeTrainingExample(example) {
  const output = example.output ?? {};
  const normalized = {
    ...output,
    facts: normalizeFacts(Array.isArray(output.facts) ? output.facts : []),
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
      query_intent: "memory_recall",
      selected_memory_ids: [],
      selected_indexable_keys: [],
      max_items: 0,
      reasoning: "No grounded recall context selected."
    };
  }
  return { ...example, output: normalized };
}

export function buildIndexables(input) {
  const content = cleanText(input.content);
  if (!content) return [];
  const key = mnemonicKey(content, input.tags ?? []);
  const secondary = factKey(input.facts ?? []);
  const base = {
    kind: "mnemonic",
    key,
    target_type: input.target_type ?? "memory",
    target_id: input.target_id ?? "",
    salience: salienceFor(content, input.tags ?? []),
    reconstructive_hint: reconstructiveHint(content),
    evidence_text: content,
    tags: normalizeTags(input.tags ?? []).slice(0, 6)
  };
  return secondary && secondary !== key
    ? [base, { ...base, kind: "fact_anchor", key: secondary, salience: Math.max(base.salience, 0.82) }]
    : [base];
}

export function normalizeTags(tags) {
  return [...new Set((tags ?? [])
    .map((tag) => String(tag).trim())
    .filter(Boolean)
    .map((tag) => tag.replace(/\s+/g, "_")))];
}

export function parseJsonArray(value) {
  if (Array.isArray(value)) return value;
  if (typeof value !== "string" || !value.trim()) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function cleanText(value) {
  return typeof value === "string" ? value.trim().replace(/\s+/g, " ") : "";
}

function normalizeFacts(facts) {
  return facts
    .filter((fact) => fact && typeof fact === "object")
    .map((fact) => ({
      subject: cleanText(fact.subject),
      predicate: normalizePredicate(fact.predicate),
      value: cleanText(fact.value ?? fact.value_text ?? fact.object),
      confidence: clamp01(fact.confidence ?? 0.8),
      inference_kind: "explicit",
      evidence_text: cleanText(fact.evidence_text ?? fact.value_text ?? fact.value ?? fact.object),
      ...(fact.temporal_expression ? { temporal_expression: cleanText(fact.temporal_expression) } : {}),
      ...(fact.resolved_time ? { resolved_time: cleanText(fact.resolved_time) } : {}),
      ...(fact.resolved_time_confidence != null ? { resolved_time_confidence: clamp01(fact.resolved_time_confidence) } : {})
    }))
    .filter((fact) => fact.subject && fact.predicate && fact.value && fact.evidence_text);
}

function normalizePredicate(value) {
  const normalized = cleanText(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return /^[a-z]/.test(normalized) ? normalized : "has_value";
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

function salienceFor(content, tags) {
  const lower = content.toLowerCase();
  let score = 0.68;
  if (/\b\d{4}\b|yesterday|last week|last year|next month|june|july|may/.test(lower)) score += 0.08;
  if (/decision|prefer|constraint|failed|failure|benchmark|indexable|mnemonic|recall|temporal/.test(lower)) score += 0.12;
  if ((tags ?? []).length > 0) score += 0.04;
  return Number(Math.min(score, 0.98).toFixed(2));
}

function reconstructiveHint(content) {
  const sentence = content.match(/^(.+?[.!?])(?:\s|$)/)?.[1] ?? content;
  return sentence.length <= 160 ? sentence : `${sentence.slice(0, 157).trim()}...`;
}

function meaningfulTokens(text) {
  const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "into", "said", "them", "they", "their", "memory", "tools", "local", "after", "user", "psm"]);
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

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number));
}

