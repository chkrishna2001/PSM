# PSM Model 50M Generative Production Plan

## Summary

Build a CPU-runnable generative PSM model that emits the actual storage JSON used by PSM, instead of extending the old Nano PSM classifier path.

The old Nano PSM effort proved useful for fast classification experiments, but it is not the production model path. It predicts labels and scores. It does not generate complete `memory.content`, `facts[]`, updates, conflicts, evidence text, or validated storage decisions. The new work should live in a separate implementation area so architectural decisions are not constrained by the failed classifier shape.

Working name:

```text
psm-model
```

Goal:

```text
input conversation/source/context -> valid PSM JSON output
```

The model should eventually support CPU-first PSM on company VMs where Qwen is too slow, while keeping Qwen as a quality fallback and teacher.

## Implementation Strategy

Move fast by splitting the work into small, reviewable pieces. Do not run this as one large rewrite.

Principles:

- create a new `psm-model` folder/package instead of expanding `nano-psm`.
- keep the old classifier only as a reference, benchmark, or optional future router.
- make the first milestone end-to-end, even if the model is tiny.
- gate every generated output with JSON/schema validation before any runtime write path.
- use Claude CLI for bounded mechanical tasks, then review and merge manually.
- prefer small commits/patches with focused tests over large speculative rewrites.

Suggested initial folder:

```text
psm-model/
  README.md
  pyproject.toml
  src/psm_model/
    __init__.py
    schema.py
    prompts.py
    tokenizer.py
    data/
    model/
    train.py
    generate.py
    evaluate.py
  tests/
  scripts/
```

The exact structure can change after inspecting existing repo conventions, but the new work should remain isolated from `nano-psm` until it has working gates.

## Work Split

### Piece 1: Contract and Gates

Deliverables:

- canonical `StorageDecision`-compatible output schema.
- strict validator for generated JSON.
- direct probe set with expected outputs.
- evaluation command that reports parse rate, schema-valid rate, action accuracy, type accuracy, and fact fields.

This is the first piece because all later work depends on knowing whether generated JSON is acceptable.

### Piece 2: Dataset Format

Deliverables:

- canonical JSONL training row format.
- converter interface for source datasets.
- small seed dataset with hand-authored examples.
- dataset gate for invalid JSON, missing evidence, bad action/type balance, duplicates, and private-data checks.

All sources must be converted into:

```text
input conversation/source/context -> exact PSM JSON output
```

Do not train on raw transcripts directly.

### Piece 3: Tokenizer

Deliverables:

- BPE/SentencePiece-style tokenizer training path.
- encode/decode tests for prompts and JSON outputs.
- reserved/control tokens for prompt boundaries and end-of-output.

The current hash tokenizer is acceptable for classification but not for exact JSON generation.

### Piece 4: Tiny Generative Model

Deliverables:

- decoder-only transformer implementation.
- tiny debug config.
- train loop that can overfit a very small batch.
- generate command that emits JSON text.
- tests for shape, loss, checkpoint save/load, and deterministic smoke generation.

The tiny model is the proof that the pipeline works before spending time on 50M training.

### Piece 5: 50M Candidate

Deliverables:

- 50M-ish config.
- resumable training checkpoints.
- metrics artifact per checkpoint.
- private Hugging Face upload script only after local artifacts are validated.
- CPU inference benchmark.

The target is around 50M parameters, not exact parameter numerology. CPU viability and JSON quality matter more than a precise count.

### Piece 6: Runtime Integration

Deliverables:

- `generate.py` or equivalent runtime entrypoint that streams JSONL inputs.
- parser/validator before memory writes.
- fallback policy for invalid JSON, invalid schema, or low confidence.
- integration path into the same `PsmService.remember()` / store path as Qwen.

This should happen only after the schema gates and direct probes are already passing.

## Claude CLI Delegation Workflow

Claude CLI is available on this machine and runs on `nemotron:free`. Use it as a helper for narrow, low-risk tasks, not as the architectural owner.

Delegation rules:

- give Claude one small task at a time.
- include exact files, expected outputs, and constraints in each prompt.
- ask for a patch or file-level implementation, not broad design.
- avoid delegating core architecture, schema decisions, model quality judgments, or runtime safety decisions.
- review every Claude-produced change before merging.
- run tests/gates after accepting any generated changes.

Good tasks for Claude:

- scaffold package files.
- write straightforward dataclasses or schema constants from an existing contract.
- add tests for already-defined behavior.
- convert a simple dataset format into canonical JSONL.
- create README usage snippets.
- fill repetitive probe fixtures.

Tasks to keep with Codex:

- folder/package architecture.
- output contract decisions.
- validation semantics.
- training/evaluation design.
- fallback/write-path safety.
- reviewing and merging generated code.
- deciding whether a checkpoint is viable.

Example Claude CLI task prompt:

```text
In C:\Users\chkri\source\repos\PSM, implement only the tests for psm-model schema validation.
Do not change production code.
Target files:
- psm-model/tests/test_schema.py
Expected behavior:
- valid StorageDecision JSON passes.
- malformed JSON fails.
- unknown action fails.
- fact without evidence_text fails.
Return a concise summary and the diff.
```

Review checklist for Claude output:

- scope stayed inside the requested files.
- no unrelated refactors.
- no weakened validation.
- no hardcoded paths that break on another machine.
- no dependency added without a clear reason.
- tests are meaningful and can fail for the right reason.

## Required Output Contract

The production model must emit JSON compatible with `PsmService.remember()` / `StorageDecision`:

```json
{
  "action": "ignore | store_episodic | promote_semantic | update_existing | flag_conflict | flag_and_store",
  "memory": {
    "content": "durable memory text",
    "type": "episodic | semantic",
    "strength": 0.0,
    "decay_rate": 0.0,
    "emotional_weight": 0.0,
    "confidence": 0.0,
    "tags": [],
    "temporal_expression": "optional",
    "resolved_time": "optional"
  },
  "facts": [
    {
      "subject": "entity",
      "predicate": "snake_case_relation",
      "value": "short value",
      "confidence": 0.0,
      "inference_kind": "explicit",
      "evidence_text": "supporting text"
    }
  ],
  "reasoning": "short reason"
}
```

The runtime must parse and validate this JSON before writing memory.

## Lean Model Output Experiment

The runtime storage contract stays `StorageDecision` JSON, but the model does not necessarily need to emit full JSON directly.

Current experiment:

- full JSON remains the canonical runtime format.
- compact JSON array is available for comparison.
- pipe-heavy tagged DSL is available as the most compact text target.
- `@tag` line DSL is available as the most learnable/debuggable text target.
- strict parser expands tagged DSL back to `StorageDecision` JSON.
- expanded output must pass the same schema validator before any memory write.

Direct probe output-token comparison with the current byte tokenizer:

```text
full JSON total:       2532 tokens
compact array total:  1702 tokens, 32.8% savings
pipe tagged DSL total: 1554 tokens, 38.6% savings
@tag DSL total:        1779 tokens, 29.7% savings
```

Decision for the next model slice:

- continue supporting full JSON for debugging and runtime compatibility.
- train/evaluate pipe-tagged DSL as the preferred small-model output format because it has explicit record boundaries and the best measured token savings.
- keep `@tag` DSL as the readability/learnability benchmark.
- never write tagged output directly; always parse, expand, and validate.
- keep compact JSON array as a benchmark but avoid it as the main target unless DSL parsing proves brittle.

This matters because context and output budget are production constraints. Reducing output tokens gives more room for current input and recalled memories without increasing model size.

Tiny overfit result:

- full JSON can overfit and produce valid JSON for all 5 probes with enough steps.
- `@tag` DSL can overfit and produce valid parsed outputs for all 5 probes.
- pipe-tagged DSL passed the all-probe generation gate:
  - parse-valid rate: 100%
  - schema-valid rate: 100%
  - action accuracy: 100%
  - memory type accuracy: 100%
  - fact count accuracy: 100%
  - average generated output: 310.8 byte tokens

Pipe delimiter risk is handled by strict escaping tests for `|`, comma, backslash, and newlines in content/evidence.

Tokenizer experiment:

- byte tokenizer is inefficient but passes the all-probe generation gate.
- fixed DSL-atom tokenizer reduced probe training tokens by about 9% and passes the all-probe generation gate.
- pattern tokenizer reduced probe training tokens by about 66% and passes the all-probe generation gate.
- byte-level BPE tokenizer reduced probe training tokens by about 67% and passes the all-probe generation gate.
- prompt tokens are now masked from loss so training optimizes output generation, not prompt reconstruction.
- the earlier BPE/pattern/DSL failures were invalid because the evaluator trained with the byte tokenizer while decoding with the supplied tokenizer. That bug is fixed.

Decision:

- use the pattern tokenizer as the current best debug tokenizer because it passed exact generated-output gates while reducing average generated output to 106.2 tokens on direct probes.
- keep byte, DSL, and BPE tokenizers available as baselines.
- do not accept token savings as success without generated output validity.
- retest tokenizer choice on larger seed and held-out sets before claiming production readiness.

50M trainability status:

- 50M preset: 16 layers, 8 heads, 512 embedding width, 2048 context length.
- parameter estimate with byte tokenizer and 2048 context: 51,571,200.
- parameter estimate with current pattern tokenizer and 2048 context: 51,661,824.
- 50M training CLI dry-run validates data/config and prints the parameter estimate.
- short-context CPU forward/backward smoke passed with the same 50M block shape.
- one-step CPU training smoke passed through the real training CLI with pattern tokenizer, context length 1536, and a 51,399,680 parameter estimate.
- real-v1 data conversion from existing Nano storage rows produced 8,082 accepted storage examples after removing recall_context and duplicate rows.
- real-v1 context-safe 2048-token splits are train 7,078, validation 640, test 285.
- real-v1 4096-vocab pattern tokenizer reduced training tokens from 9,699,025 byte tokens to 4,070,244 pattern tokens.
- one-step CPU 50M training smoke passed on real-v1-ctx2048 with 53,535,744 parameter estimate.
- saved checkpoints now include tokenizer and metadata sidecars.
- checkpoint evaluation gates saved models on generated output quality, not just loss or classifier-style counts.
- an undertrained 5-step debug checkpoint failed all generated-output gates.
- a 300-step debug checkpoint passed exact direct-probe gates after generating valid tagged storage decisions.

Current definition of "passed":

- the model output parses.
- the expanded output schema-validates.
- action/classification is correct.
- memory type is correct.
- memory content is exact.
- fact count is correct.
- fact subject/predicate/value/evidence text are exact.

## Model Concepts

### Tokenizer

The tokenizer converts text and JSON into token IDs.

A generative model needs a real tokenizer, likely BPE or SentencePiece-style, trained on:

- PSM prompts.
- PSM JSON outputs.
- memory/fact text.
- benchmark examples.

### Embedding Layer

The embedding layer converts token IDs into vectors. These vectors are the model's numeric representation of words, JSON syntax, field names, and values.

### Transformer Blocks

Transformer blocks process the token vectors.

- attention learns which parts of the input matter.
- feed-forward layers transform token representations.
- residual connections and normalization stabilize training.

### Decoder Head

The decoder head predicts the next output token.

This is the missing capability in the current Nano classifier. It is what allows the model to generate actual JSON text.

### Training

Training means:

1. Give the model input text/JSON.
2. Give the expected output JSON.
3. Ask the model to predict each next output token.
4. Measure prediction error with next-token loss.
5. Use backpropagation to update model weights.

The model gradually learns to map PSM inputs to valid PSM JSON.

### Inference

Inference means:

1. Give the trained model a new PSM input.
2. Decode output tokens until end-of-output.
3. Parse JSON.
4. Validate schema.
5. If valid, write memory.
6. If invalid or low-confidence, use repair or Qwen fallback.

## Architecture

Recommended v1:

- decoder-only transformer.
- target size: around 50M parameters.
- context length: 1024-2048 tokens.
- output: constrained JSON text.
- tokenizer: BPE/SentencePiece-style.
- checkpoint format: PyTorch first, later ONNX if useful.

Keep the existing classifier only as:

- speed benchmark.
- optional future router/scorer sidecar.
- reference for dataset lessons.
- not the production JSON generator.

## Data Plan

Existing data is useful but must be converted into a canonical production contract.

Candidate sources:

- mem0 examples.
- Letta examples.
- LoCoMo conversations.
- PSM/Qwen traces.
- current Codex conversations.
- exported ChatGPT markdown, if explicitly provided.
- public memory/preference datasets where license allows.
- synthetic examples generated by a strong teacher model and then gated.

### Minimum Dataset Targets

First real run:

- 20k high-quality examples minimum.

Production candidate:

- 50k-100k examples.

Required balance:

- ignore/noise.
- episodic event.
- semantic preference/profile.
- project/workflow memory.
- temporal/date memory.
- update existing.
- conflict.
- multi-fact extraction.
- no-fact storage.
- recall/context planning if included.

### Private Data Rules

Codex and ChatGPT data can be valuable because it is real, but it must be handled carefully:

- sanitize secrets and personal data.
- convert into evidence-backed examples.
- keep private examples local or in private Hugging Face repos.
- do not upload raw private transcripts by default.

## Evaluation Gates

Every candidate checkpoint must pass production gates before LoCoMo:

- JSON parse rate >= 99%.
- schema-valid rate >= 98%.
- action accuracy >= 95%.
- memory type accuracy >= 95%.
- ignore/noise accuracy >= 95%.
- fact extraction F1 on subject/predicate/value/evidence.
- memory content quality via exact and semantic checks.
- score MAE tracked but not treated as readiness alone.
- direct hand probes must pass.
- LoCoMo retrieval and answer eval run only after schema gate passes.

Required direct probes:

- semantic preference should produce semantic memory and preference fact.
- dated event should produce episodic memory and temporal fact.
- project instruction should produce semantic/project memory.
- low-value text like `okay thanks` should be ignored.
- contradiction should route to conflict or update.

## Fast Execution Plan

Run the work in this order:

1. Update this plan and create `psm-model` skeleton.
2. Implement contract validator and direct probes.
3. Ask Claude CLI to add schema tests and probe fixtures.
4. Review Claude output and merge only clean changes.
5. Build canonical dataset row format and gate.
6. Ask Claude CLI to scaffold simple converters once the format is fixed.
7. Implement tokenizer and tiny model.
8. Train tiny model to overfit a seed batch.
9. Implement generation and evaluation commands.
10. Scale dataset, then train the 50M candidate.
11. Integrate runtime only after gates pass.

## Runtime Integration

Replace heuristic `predict.py` reconstruction with actual JSON generation:

- `generate.py` streams JSONL inputs.
- model emits JSON text.
- parser validates output.
- invalid output triggers repair/fallback.
- valid output goes through the same `PsmService.remember()` / store path as Qwen.

Fallback policy:

- invalid JSON -> parser repair if simple.
- invalid schema -> reject or Qwen fallback.
- low confidence -> Qwen fallback.
- valid high-confidence output -> write memory.

## Time Estimate

Fast viability path:

- 0.5 day: updated plan, `psm-model` skeleton, contract, direct probes.
- 0.5-1 day: schema gate, dataset row format, seed examples.
- 1 day: tokenizer, tiny model, train/generate loop.
- 1-2 days: dataset conversion/gating for 20k+ examples.
- 1-3 days: first 50M training iterations and failure analysis.

Expected time to know viability:

- 3-7 days.

Expected time to strong production candidate:

- 1-2 weeks, assuming data quality is good.

Claude CLI should reduce wall-clock time on scaffolding and repetitive fixtures, but not replace review or quality gates.

## Assumptions

- CPU-first deployment remains a core goal.
- Qwen remains quality fallback and teacher model.
- The current 10M classifier is not enough as the full PSM model.
- A 50M generative model may be slower than current Nano, but should still be much more deployable than Qwen 1.5B.
- The project will not claim PSM model readiness until production JSON gates and LoCoMo gates pass.
