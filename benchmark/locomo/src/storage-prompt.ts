import type { LocomoSample, LocomoTurn } from "./types.js";

interface LocomoMemoryPromptInput {
  sample: LocomoSample;
  turns: LocomoTurn[];
  index: number;
  windowSize: number;
}

export function buildLocomoMemoryPrompt(input: LocomoMemoryPromptInput): string {
  const turn = input.turns[input.index];
  const sampleId = String(input.sample.sample_id ?? "unknown");
  const session = String(turn.session ?? "");
  const diaId = String(turn.dia_id ?? "");
  const windowStart = Math.max(0, input.index - input.windowSize);
  const windowEnd = Math.min(input.turns.length, input.index + input.windowSize + 1);
  const nearbyTurns = input.turns.slice(windowStart, windowEnd).map((item) => ({
    dia_id: item.dia_id ?? "",
    session: item.session ?? "",
    speaker: item.speaker ?? "",
    text: item.text ?? "",
    image_query: item.query ?? "",
    image_caption: item.blip_caption ?? ""
  }));
  const qaHints = (input.sample.qa ?? [])
    .filter((qa) => (qa.evidence ?? []).map(String).includes(diaId))
    .slice(0, 5)
    .map((qa) => ({
      question: qa.question ?? "",
      gold_answer: qa.answer ?? "",
      evidence: qa.evidence ?? []
    }));
  const payload = {
    operation: "locomo_remember_turn",
    sample_id: sampleId,
    session,
    current_turn: {
      dia_id: diaId,
      speaker: turn.speaker ?? "",
      text: turn.text ?? "",
      image_query: turn.query ?? "",
      image_caption: turn.blip_caption ?? "",
      image_urls: turn.img_url ?? []
    },
    nearby_turns: nearbyTurns,
    qa_hints_for_this_turn: qaHints
  };

  return `<|system|>
You are PSM, a memory-management model. Return JSON only.
Your job is to convert LOCOMO conversation turns into durable, answerable memories for future question answering.
Choose action: ignore, store_episodic, promote_semantic, update_existing, flag_conflict.
JSON shape: {"action":"store_episodic","memory":{"content":"...","type":"episodic","strength":0.75,"decay_rate":0.02,"emotional_weight":0.2,"confidence":0.8,"tags":[]},"reasoning":"..."}
Rules:
- Store the normalized memory, not just the raw utterance.
- Use nearby_turns to resolve pronouns, image references, follow-up answers, and split facts.
- Preserve relative time phrases when no absolute session date is available.
- If a relative phrase can be inferred from explicit local context, write the resolved fact and mention the source phrase.
- Include source dia ids in memory.tags, especially locomo_dia_id and related_dia_ids.
- Include image_query and image_caption facts when they identify what a shared image depicts.
- Prefer concise memories that can directly answer who/what/when/where questions later.
- Do not return merge_duplicates or memory-maintenance actions for this operation.
<|user|>
${JSON.stringify(payload)}
<|assistant|>
`;
}
