import type { MemoryRecord, MemorySourceMetadata } from "./types.js";

export const psmSystemPrompt = `You are the Personal Small Model (PSM), a specialized AI trained exclusively to perform memory management operations for LLM agents.

Your job is NOT to answer user questions. Your job is to:
1. Analyze conversations and decide what is worth remembering
2. Manage a tiered memory store (episodic, semantic, archival)
3. Detect conflicts between new information and existing memories
4. Assign appropriate strength, decay rate, and emotional weight to memories
5. Promote repeated episodic patterns into semantic facts
6. Ignore low-value noise that is not worth storing
7. Rank memories by relevance to a current query
8. Update existing memories when information changes

Always respond with a valid JSON object.`;

export function buildStoragePrompt(llmResponse: string, existingMemories: MemoryRecord[] = [], source: MemorySourceMetadata = {}): string {
  const payload = {
    operation: "remember_llm_response",
    conversation: [{ role: "assistant", content: llmResponse }],
    source,
    memory_store: existingMemories.slice(0, 20).map((memory) => ({
      id: memory.id,
      table: memory.table,
      content: memory.content,
      strength: memory.strength,
      tags: parseTags(memory.tags),
      saved_at: memory.created_at,
      source_kind: memory.source_kind,
      source_id: memory.source_id,
      source_timestamp: memory.source_timestamp,
      source_label: memory.source_label,
      temporal_expression: memory.temporal_expression,
      resolved_time: memory.resolved_time
    }))
  };
  return `<|system|>\n${psmSystemPrompt}\n<|user|>\nAnalyze this LLM response and return JSON only with action, memory, facts, reasoning, confidence, emotional_weight, and contradiction_score.\nWhen a durable memory contains relative time such as yesterday, last week, or next month, preserve that phrase as temporal_expression. If source.source_timestamp gives enough anchor context, also set resolved_time and resolved_time_confidence. Copy relevant source_kind, source_id, source_timestamp, and source_label into memory when provided.\nAlso extract searchable facts into facts[]. Use generic subject/predicate/value records instead of inventing schema-specific columns. Facts must be supported by evidence_text. Mark inference_kind as explicit when directly stated and inferred when implied, such as single parent implying relationship_status=single. Prefer stable predicates like relationship_status, parental_status, family_goal, career_interest, activity, location, event_date, preference, project, workflow, tool, constraint.\nExample facts: [{\"subject\":\"Caroline\",\"predicate\":\"relationship_status\",\"value\":\"single\",\"confidence\":0.75,\"inference_kind\":\"inferred\",\"evidence_text\":\"single parent\"},{\"subject\":\"Melanie\",\"predicate\":\"event_date\",\"value\":\"2022\",\"fact_type\":\"temporal_fact\",\"confidence\":0.95,\"inference_kind\":\"explicit\",\"evidence_text\":\"painting of a sunrise from 2022\"}].\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}

export function buildStorageRepairPrompt(llmResponse: string, invalidOutput: string): string {
  const payload = {
    operation: "repair_remember_json",
    assistant_response: llmResponse,
    invalid_model_output: invalidOutput,
    required_schema: {
      action: "ignore | store_episodic | promote_semantic | update_existing | flag_conflict | flag_and_store",
      memory: {
        content: "concise extracted durable memory, not the raw assistant response",
        type: "episodic | semantic",
        strength: "number between 0 and 1",
        decay_rate: "number between 0 and 1",
        emotional_weight: "number between 0 and 1",
        confidence: "number between 0 and 1",
        tags: ["short strings"],
        source_kind: "optional source type, copied from source metadata when relevant",
        source_id: "optional provenance id, copied from source metadata when available",
        source_timestamp: "optional source event timestamp, not the database save time",
        source_label: "optional human-readable source label",
        temporal_expression: "optional original relative time phrase",
        resolved_time: "optional ISO-like resolved time/date when source_timestamp anchors a relative phrase",
        resolved_time_confidence: "optional number between 0 and 1"
      },
      facts: [
        {
          subject: "person, project, event, tool, or entity the fact is about",
          predicate: "stable snake_case relation such as relationship_status, activity, event_date, career_interest",
          value: "short fact value",
          value_text: "optional string value if value is not already a short string",
          fact_type: "optional profile_fact | temporal_fact | preference_fact | project_fact | workflow_fact",
          confidence: "number between 0 and 1",
          inference_kind: "explicit | inferred | derived",
          evidence_text: "short exact phrase supporting the fact",
          temporal_expression: "optional original relative time phrase",
          resolved_time: "optional resolved date/time when source timestamp anchors the fact",
          resolved_time_confidence: "optional number between 0 and 1"
        }
      ],
      reasoning: "brief reason for the memory decision",
      confidence: "number between 0 and 1",
      emotional_weight: "number between 0 and 1",
      contradiction_score: "number between 0 and 1"
    },
    rules: [
      "Return exactly one valid JSON object and no markdown.",
      "If the response contains no durable memory, use action ignore and memory null.",
      "Do not copy the raw assistant response into memory.content.",
      "memory.content must be a concise extracted fact, plan, decision, user preference, correction, or unresolved task.",
      "facts must be an array. Use [] when there are no searchable facts.",
      "Every fact must include subject, predicate, value or value_text, confidence, inference_kind, and evidence_text.",
      "Do not invent facts unsupported by evidence_text."
    ]
  };
  return `<|system|>\n${psmSystemPrompt}\n<|user|>\nRepair the invalid remember output into valid JSON only.\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}

export function buildRecallPlanPrompt(question: string, topK: number): string {
  const payload = {
    operation: "recall_plan",
    question,
    available_tables: ["episodic", "semantic", "archival"],
    requested_top_k: topK
  };
  return `<|system|>\n${psmSystemPrompt}\n<|user|>\nCreate a recall plan as JSON only with intent, target_tables, filters, ranking_hints, temporal_intent, and top_k. PSM owns memory planning: choose the memory tiers that should be searched, but do not answer the user.\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}

export function buildContextPlanPrompt(prompt: string, topK: number): string {
  const payload = {
    operation: "context_plan",
    user_prompt: prompt,
    available_tables: ["episodic", "semantic", "archival"],
    requested_top_k: topK
  };
  return `<|system|>\n${psmSystemPrompt}\n<|user|>\nCreate a memory context recall plan as JSON only with intent, target_tables, filters, ranking_hints, temporal_intent, and top_k. PSM owns memory planning: choose the memory tiers that should be searched, but do not answer the user.\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}

export function buildContextRenderPrompt(prompt: string, memories: MemoryRecord[], topK: number): string {
  const payload = {
    operation: "render_context",
    user_prompt: prompt,
    max_items: topK,
    required_schema: {
      selected_ids: ["exact candidate memory id"],
      reasoning: "brief explanation of why these context items were selected"
    },
    instructions: [
      "This is a context selection operation only.",
      "Do not perform memory maintenance actions.",
      "Do not merge, update, delete, rewrite, deduplicate, promote, or create memory records.",
      "Use only candidate_memories provided in this payload.",
      "Every selected_ids entry must exactly match one candidate memory id.",
      "Do not invent ids, people, projects, technical skills, or facts.",
      "If no candidate memory is relevant, return {\"selected_ids\":[],\"reasoning\":\"No relevant memory.\"}.",
      "Do not return action, affected_records, affected_entries, modified_fields, merge_id, target_id, or tags as the top-level schema.",
      "Return exactly one JSON object with top-level keys selected_ids and reasoning.",
      "Select only memories that are directly useful for the current user prompt.",
      "Do not answer the user prompt."
    ],
    candidate_memories: memories.slice(0, Math.max(topK, 10)).map((memory) => ({
      id: memory.id,
      table: memory.table,
      content: memory.content,
      strength: memory.strength,
      confidence: memory.confidence,
      tags: parseTags(memory.tags)
    }))
  };
  return `<|system|>\n${psmSystemPrompt}\n<|user|>\nReturn JSON only for operation render_context. The only valid top-level shape is {"selected_ids":["EXACT_CANDIDATE_ID"],"reasoning":"..."}.
Use only candidate_memories. Do not invent ids or context. Do not return merge/update actions.
${JSON.stringify(payload)}\n<|assistant|>\n`;
}

function parseTags(value: string | null | undefined): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}
