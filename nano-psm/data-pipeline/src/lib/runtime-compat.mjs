const runtimeStorageActions = new Set([
  "ignore",
  "store_episodic",
  "promote_semantic",
  "update_existing",
  "flag_conflict",
  "flag_and_store"
]);

const runtimeRecallAction = "recall_context";

export function validateRuntimeCompatibility(row) {
  const errors = [];
  if (!row || typeof row !== "object") return ["row must be an object"];
  const output = row.output;
  if (!output || typeof output !== "object" || Array.isArray(output)) return ["output must be an object"];
  const action = output.action;

  if (action === runtimeRecallAction) {
    validateRecallCompatibility(row, errors);
    return errors;
  }

  if (!runtimeStorageActions.has(action)) {
    errors.push(`action cannot be bridged to current runtime: ${action}`);
    return errors;
  }

  if (action === "ignore") {
    if (output.memory !== null) errors.push("ignore must bridge to StorageDecision with memory:null");
  } else {
    validateMemoryPayload(output.memory, errors);
  }

  if (!Array.isArray(output.facts)) errors.push("facts must be an array for StorageDecision bridge");
  else output.facts.forEach((fact, index) => validateFactPayload(fact, index, errors));

  if (!Array.isArray(output.indexables)) errors.push("indexables must be an array for sidecar bridge");
  else output.indexables.forEach((indexable, index) => validateIndexableSidecar(indexable, index, errors));

  if (typeof output.reasoning !== "string") errors.push("reasoning must be a string for StorageDecision bridge");
  return errors;
}

export function toRuntimeBridge(row) {
  const errors = validateRuntimeCompatibility(row);
  if (errors.length > 0) {
    throw new Error(errors.join("; "));
  }
  const output = row.output;
  if (output.action === runtimeRecallAction) {
    return {
      kind: "recall_bridge",
      recall_plan: {
        intent: output.recall.query_intent,
        target_tables: ["semantic", "episodic"],
        filters: {},
        ranking_hints: output.recall.selected_indexable_keys,
        top_k: output.recall.max_items,
        raw_json: JSON.stringify(output.recall)
      },
      context_render: {
        context_items: [],
        selected_ids: output.recall.selected_memory_ids,
        reasoning: output.recall.reasoning,
        raw_json: JSON.stringify(output)
      }
    };
  }
  return {
    kind: "storage_bridge",
    storage_decision: {
      action: output.action,
      memory: output.memory,
      facts: output.facts,
      reasoning: output.reasoning,
      confidence: output.memory?.confidence,
      emotional_weight: output.memory?.emotional_weight,
      raw_json: JSON.stringify(output)
    },
    indexables: output.indexables
  };
}

function validateMemoryPayload(memory, errors) {
  if (!memory || typeof memory !== "object" || Array.isArray(memory)) {
    errors.push("storage action requires memory object");
    return;
  }
  if (typeof memory.content !== "string" || !memory.content.trim()) errors.push("memory.content is required");
  if (!["episodic", "semantic"].includes(memory.type)) errors.push(`memory.type cannot be bridged: ${memory.type}`);
  for (const key of ["strength", "decay_rate", "emotional_weight", "confidence"]) {
    if (typeof memory[key] !== "number" || memory[key] < 0 || memory[key] > 1) errors.push(`memory.${key} must be 0..1`);
  }
  if (!Array.isArray(memory.tags)) errors.push("memory.tags must be an array");
}

function validateFactPayload(fact, index, errors) {
  if (!fact || typeof fact !== "object" || Array.isArray(fact)) {
    errors.push(`facts[${index}] must be an object`);
    return;
  }
  for (const key of ["subject", "predicate", "value", "evidence_text"]) {
    if (typeof fact[key] !== "string" || !fact[key].trim()) errors.push(`facts[${index}].${key} is required`);
  }
}

function validateIndexableSidecar(indexable, index, errors) {
  if (!indexable || typeof indexable !== "object" || Array.isArray(indexable)) {
    errors.push(`indexables[${index}] must be an object`);
    return;
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(String(indexable.key ?? ""))) {
    errors.push(`indexables[${index}].key must be lowercase hyphenated`);
  }
  for (const key of ["kind", "target_type", "reconstructive_hint", "evidence_text"]) {
    if (typeof indexable[key] !== "string" || !indexable[key].trim()) errors.push(`indexables[${index}].${key} is required`);
  }
}

function validateRecallCompatibility(row, errors) {
  const recall = row.output.recall;
  if (!recall || typeof recall !== "object" || Array.isArray(recall)) {
    errors.push("recall_context requires recall object");
    return;
  }
  if (row.output.memory !== null) errors.push("recall_context must use memory:null");
  if (!Array.isArray(recall.selected_memory_ids)) errors.push("recall.selected_memory_ids must be an array");
  if (!Array.isArray(recall.selected_indexable_keys)) errors.push("recall.selected_indexable_keys must be an array");
  if (typeof recall.query_intent !== "string" || !recall.query_intent.trim()) errors.push("recall.query_intent is required");
  if (typeof recall.max_items !== "number" || recall.max_items < 0) errors.push("recall.max_items must be non-negative");
  if (typeof recall.reasoning !== "string" || !recall.reasoning.trim()) errors.push("recall.reasoning is required");

  const ids = new Set((row.input?.memory_store ?? []).map((item) => item?.id).filter(Boolean));
  for (const id of recall.selected_memory_ids ?? []) {
    if (ids.size > 0 && !ids.has(id)) errors.push(`recall selected id is not present in memory_store: ${id}`);
  }
}

