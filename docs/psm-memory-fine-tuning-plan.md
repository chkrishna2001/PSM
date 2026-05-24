# PSM Memory Fine-Tuning Plan

## Goal

Train the Personal Small Model to perform PSM memory operations reliably on the model sizes and quantizations users can actually run.

The product target is not FP16 quality in a lab run. FP16 is the reference model for diagnosis. The deployable target is a quantized model, likely Q4, Q5, Q6, or Q8 depending on user hardware.

The fine-tune must optimize for:

- valid JSON under quantization
- exact output schema adherence
- speaker-aware memory extraction
- temporal extraction and normalization
- factual extraction only when directly evidenced
- update and conflict handling
- low-confidence handling for noisy or incomplete inputs
- realistic developer/project memory use cases

## Current Evidence

The LOCOMO 20-turn smoke comparison on 2026-05-20 showed a clear quantization gap:

```text
Q4_K_M:
seen=20 stored=8 ignored=0 failed=12

F16:
seen=20 stored=12 ignored=8 failed=0
```

This means the current model has more capability than the Q4 run suggests, but the deployable quant is brittle.

The F16 run still showed training-contract issues:

- The model sometimes emitted a legacy shape such as `operation` and `assistant_response` instead of the expected `action` and `memory`.
- The model sometimes wrote generic `User` even when the speaker name was available.
- The model sometimes stored conversational compliments or questions as durable memories.
- The model sometimes emitted unsupported facts, such as hobbies or project details not grounded in the current turn.

Conclusion:

```text
The problem is not simply model size.
The problem is alignment between fine-tune data, runtime prompt, output schema, product flow, and target quantization.
```

## Product Contract

The fine-tuned model should learn one canonical input-output contract. Avoid supporting multiple equivalent JSON shapes in training unless the runtime explicitly supports them.

Target output shape:

```json
{
  "action": "ignore | store_episodic | promote_semantic | update_existing | flag_conflict | flag_and_store",
  "memory": null,
  "facts": [],
  "updates": [],
  "conflicts": [],
  "reasoning": "brief grounded reason"
}
```

When `memory` is present:

```json
{
  "content": "concise durable memory",
  "type": "episodic | semantic",
  "strength": 0.0,
  "decay_rate": 0.0,
  "emotional_weight": 0.0,
  "confidence": 0.0,
  "tags": ["short", "grounded", "tags"],
  "temporal_expression": "optional original time phrase",
  "resolved_time": "optional ISO date/time or normalized date",
  "resolved_time_confidence": 0.0
}
```

When `facts` are present:

```json
{
  "subject": "direct subject from the conversation",
  "predicate": "stable_snake_case_relation",
  "value": "short value",
  "confidence": 0.0,
  "inference_kind": "explicit",
  "evidence_text": "supporting phrase from the current input",
  "temporal_expression": "optional",
  "resolved_time": "optional"
}
```

Required behavior:

- Return JSON only.
- Do not emit markdown.
- Do not emit `operation`, `assistant_response`, or any legacy schema.
- Do not copy raw benchmark or product wrappers into memory content.
- Do not write generic `User` when a speaker/person name is known.
- Do not create facts without direct evidence.
- Do not infer profile traits from weak context.
- Use `ignore` for greetings, filler, logistics, compliments, and non-durable chatter.

## Operation Types

The training data should cover distinct memory operations instead of one vague "remember" task.

### Ignore

Use for:

- greetings
- filler
- short acknowledgements
- transient logistics
- non-durable compliments
- unsupported or ambiguous references

Example target:

```json
{
  "action": "ignore",
  "memory": null,
  "facts": [],
  "updates": [],
  "conflicts": [],
  "reasoning": "Greeting only; no durable memory."
}
```

### Store Episodic

Use for specific events tied to a time, session, project, decision, task, or lived experience.

The memory should preserve:

- who did or said the thing
- what happened
- when it happened, if available
- why it matters, only if directly supported

### Promote Semantic

Use for stable traits or preferences that are explicitly supported or repeated across inputs.

Examples:

- "Alex prefers concise implementation plans."
- "Maya works primarily in React and Node.js."
- "The team uses Azure VMs without GPU access."

### Extract Facts

Facts are optional enrichment. Train the model to emit `facts: []` by default unless the fact is directly supported.

Facts must include evidence text.

Bad:

```json
{
  "subject": "Melanie",
  "predicate": "hobbies",
  "value": "painting, photography, reading"
}
```

Good:

```json
{
  "subject": "Melanie",
  "predicate": "uses_painting_for",
  "value": "expressing feelings and relaxing after a long day",
  "confidence": 0.92,
  "inference_kind": "explicit",
  "evidence_text": "Painting's a fun way to express my feelings... a great way to relax after a long day."
}
```

### Update Existing

Use when a new input revises an earlier memory.

Examples:

- changed preference
- changed job
- changed project architecture
- corrected date
- replaced tool or dependency

The output should include the new memory plus an update target when available.

### Flag Conflict

Use when the new input contradicts stored memory but the replacement is not certain.

Examples:

- "I no longer use React" when prior memory says the user specializes in React.
- "We moved from SQLite to Postgres" when prior memory says SQLite is the storage backend.

### Flag And Store

Use when the new input is worth storing and also conflicts with prior memory.

## Dataset Sources

### Local PSM Interaction Data

Use local DB/session data as the highest-value product-real source, with redaction.

Convert:

- product decisions
- bug investigations
- install failures
- module ownership
- benchmark results
- CLI commands used
- environment constraints
- user preferences about engineering workflow

Into:

- developer project memories
- module and architecture facts
- temporal project events
- toolchain constraints
- update and conflict examples

Examples to synthesize:

- "The LOCOMO notebook uses Hugging Face run IDs for artifact isolation."
- "The Q4_K_M LOCOMO smoke failed JSON on 12 of 20 turns."
- "The user prefers incremental version upgrades with build and commit steps."
- "Office VMs may lack GPU and may block admin-level npm/native installs."

### Synthetic Developer And Project Data

Create examples that match PSM's likely product users:

- tickets
- PR review conversations
- bug reports
- architecture discussions
- release notes
- deployment failures
- module refactors
- "we tried X; it failed; use Y"
- dependency/version constraints
- user-specific coding preferences

Convert into:

- episodic project memories
- semantic project facts
- conflict/update decisions
- temporal associations
- factual module ownership records

### PersonaMem

Convert:

- implicit preferences
- stable traits
- latent identity markers

Into:

- identity memories
- preference memories
- emotional memories

Training focus:

- distinguish explicit vs inferred preferences
- require evidence for facts
- avoid over-personalizing from one weak clue
- store stable preferences only when confidence is high

### LOCOMO

Convert:

- event chains
- timeline continuity
- episodic linking
- session-level temporal context
- evidence-backed QA targets

Into:

- episodic memory graphs
- retrieval chains
- temporal associations
- speaker-aware memories

Training focus:

- exact speaker attribution
- relative date resolution from session timestamp
- ignore non-durable turns
- keep current-turn grounding
- avoid leaking future QA answers into ingestion examples

### LongMemEval

Convert:

- contradictions
- updates
- revised preferences
- long-term QA dependencies

Into:

- memory update operations
- conflict arbitration examples
- superseded memory examples
- recall-context selection examples

Training focus:

- identify when new information replaces old information
- distinguish conflict from normal elaboration
- preserve source evidence

### REALTALK

Convert:

- noisy messaging
- fragmented statements
- incomplete references
- real multi-day chat dynamics

Into:

- low-confidence memories
- partial episodic traces
- ignore decisions
- clarification-needed or low-confidence extraction cases

Training focus:

- do not hallucinate missing referents
- store partial memories only when useful
- lower confidence for ambiguous claims
- avoid over-cleaning messy natural language into unsupported facts

## Dataset Generation Pipeline

Create a reproducible pipeline under `nano-psm/data-pipeline/`.

Recommended stages:

```text
raw source -> normalized conversation/event records -> labeled PSM examples -> validation split -> quantized eval reports
```

Suggested files:

```text
nano-psm/data-pipeline/
  README.md
  schemas/
    psm-training-example.schema.json
  src/
    generate-locomo-examples.ts
    generate-local-db-examples.ts
    generate-developer-synthetic-examples.ts
    validate-examples.ts
    split-dataset.ts
  data/
    raw/
    generated/
    validated/
  reports/
```

Each training example should include:

```json
{
  "instruction": "Perform the PSM memory operation for the current input. Return JSON only using the target schema.",
  "input": {
    "operation": "remember",
    "current_turn": {
      "speaker": "Caroline",
      "text": "I went to a LGBTQ support group yesterday and it was so powerful.",
      "timestamp": "2023-05-08T13:56:00"
    },
    "prior_context": [],
    "memory_store": []
  },
  "output": {
    "action": "store_episodic",
    "memory": {
      "content": "Caroline attended an LGBTQ support group on 2023-05-07 and found it powerful.",
      "type": "episodic",
      "strength": 0.85,
      "decay_rate": 0.04,
      "emotional_weight": 0.7,
      "confidence": 0.94,
      "tags": ["support_group", "lgbtq", "wellbeing"],
      "temporal_expression": "yesterday",
      "resolved_time": "2023-05-07",
      "resolved_time_confidence": 0.95
    },
    "facts": [],
    "updates": [],
    "conflicts": [],
    "reasoning": "Specific dated personal event with emotional significance."
  }
}
```

## Data Quality Gates

Before training, run validators for:

- valid JSON output
- only canonical top-level keys
- action is valid enum
- `memory` is null when action is ignore
- no generic `User` when `current_turn.speaker` is provided
- no unsupported fact without evidence text
- no future QA answer leakage
- no raw wrapper text in `memory.content`
- no legacy `operation` or `assistant_response`
- temporal fields parse when present
- source IDs are preserved when provided

Hard-negative examples should be included for every failure found in smoke tests.

## Quantization-Aware Evaluation

Every candidate checkpoint should be evaluated in this order:

```text
FP16
Q8_0
Q6_K
Q5_K_M
Q4_K_M
```

The acceptance bar should be based on the deployable quant, not FP16.

Minimum metrics:

- valid JSON rate
- expected schema rate
- ignore accuracy
- speaker attribution accuracy
- current-turn grounding rate
- temporal extraction accuracy
- resolved-time accuracy
- fact precision
- update/conflict classification accuracy
- unsupported fact rate
- generic `User` leakage rate
- raw wrapper leakage rate

Suggested gate for first usable Q4/Q5 candidate:

```text
valid JSON >= 99%
expected schema >= 98%
generic User leakage <= 1%
unsupported fact rate <= 2%
current-turn grounding >= 95%
ignore accuracy >= 90%
```

These numbers can be adjusted after a labeled validation set exists.

## Training Strategy

Start small and high quality.

Phase 1:

- 2k to 5k curated examples.
- Heavy emphasis on schema, ignore behavior, speaker grounding, temporal extraction, and facts.
- Include all known Q4/F16 failure cases as hard negatives.

Phase 2:

- 20k to 50k examples.
- Add dataset diversity from PersonaMem, LOCOMO, LongMemEval, REALTALK, local DB, and synthetic developer workflows.
- Balance operation types.

Phase 3:

- Quantization-aware checkpoint selection.
- Compare Q4, Q5, Q6, Q8, and FP16 on the same validation pack.
- Select product default and recommended model tiers.

## Product Implications

If Q4 remains weak after improved fine-tuning, do not force Q4 to do the full task.

Fallback architecture for small quantized models:

```text
deterministic exact capture first
model classifies store/ignore/update/conflict
model adds concise tags and optional temporal fields
facts/enrichment run only when confidence is high or on stronger model tier
```

This keeps the product viable on CPU-only and low-GPU machines without pretending F16 is the deployable baseline.

## Open Questions

- What is the minimum quantization level that can pass schema and grounding gates?
- Should the product parser support legacy `operation` / `assistant_response`, or should all training remove it completely?
- How much local interaction data can be safely used after redaction?
- Should facts be trained in the same model call as memory extraction, or as a second operation?
- Should recall-context selection be part of the same fine-tune or a separate fine-tune stage?

## Next Steps

1. Build the `nano-psm/data-pipeline/` pipeline skeleton.
2. Define the canonical training schema and validators.
3. Generate the first LOCOMO-derived examples without QA leakage.
4. Generate local developer/project examples from redacted PSM DB/session data.
5. Add hard negatives from Q4 and F16 smoke failures.
6. Produce a small validation set and score FP16, Q8, Q6, Q5, and Q4.
