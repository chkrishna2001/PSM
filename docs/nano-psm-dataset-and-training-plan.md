# Nano PSM Dataset And Training Plan

## Goal

Build a training corpus and small model for PSM memory management that can run locally and eventually replace or augment the current LLM-backed memory operation path.

The target is not a general chatbot. The target is a compact memory-operation model that can:

- classify memory actions
- extract concise memories
- extract evidence-backed facts
- detect updates and contradictions
- create mnemonic/indexable recall cues
- select recall context from memories and indexables
- run cheaply on CPU after export

The first usable version should be trainable on a Colab T4 and resumable across accounts by syncing checkpoints and datasets to Hugging Face Hub.

## Model Direction

Do not start with a nanoGPT-style autoregressive language model from scratch. That would spend most capacity learning language and JSON.

Start with a small structured model:

```text
input text + memory-store candidates
        |
tokenizer
        |
tiny encoder
        |
multi-head outputs
```

Recommended first architecture:

- embedding dimension: 384
- transformer encoder layers: 6
- attention heads: 6
- feed-forward size: 1024
- max sequence length: 1024 or 1536
- parameters: target roughly 10M, with a smaller 4M debug config only for fast smoke tests

Reason:

- The goal is a useful memory-operation model, not a toy proof of concept.
- Previous fine-tuning work with Qwen should become teacher/baseline data, not a reason to restart with an underpowered model.
- A 10M structured model should still be practical on Colab T4 and easier to export locally than a generative LLM.

Heads:

- `action_head`: ignore, store_episodic, promote_semantic, update_existing, flag_conflict, flag_and_store, recall_context
- `memory_type_head`: episodic, semantic, none
- `content_span_head`: start/end span for source-grounded memory text
- `fact_span_head`: optional evidence/value spans
- `score_heads`: strength, decay_rate, emotional_weight, confidence, salience
- `indexable_head`: cue/key classification or sequence tagging for mnemonic tokens
- `recall_selection_head`: select memory ids/indexable ids from candidates

This can later be compared against a tiny generative model, but the structured model should be the baseline because it avoids JSON failure and is easier to export to ONNX.

## Reference: llm-from-scratch

The `angelos-p/llm-from-scratch` repo is worth using as a learning and implementation reference:

```text
https://github.com/angelos-p/llm-from-scratch
```

What it gives us:

- a clean workshop-style GPT training pipeline
- tiny/small/medium scale targets around 0.5M, 4M, and 10M parameters
- a simple transformer implementation that runs on CPU, CUDA, MPS, or Colab
- a practical reminder that character-level tokenization works for tiny datasets, while BPE needs more data
- a small, understandable training loop that is easier to adapt than a production LLM stack

What to borrow:

- config scale: keep a 4-layer, 4-head, 256-dim model as a debug baseline, but train the primary model near the 10M scale
- code organization: `model.py`, `train.py`, `generate.py` style simplicity
- device handling: automatically use CUDA, MPS, or CPU
- educational clarity: keep the first nano PSM model readable and debuggable
- Colab path: keep training runnable with a simple `python train.py`

What not to copy directly:

- do not make the primary nano PSM an autoregressive Shakespeare-style text generator
- do not rely on free-form JSON generation for the core product path
- do not use character-level tokenization once the full mixed memory dataset grows past the small prototype
- do not optimize only next-token loss; PSM needs action, span, indexable, and recall-selection metrics

Recommended use:

1. Use `llm-from-scratch` to build a tiny generative baseline on PSM JSONL.
2. Build the structured encoder/head model beside it.
3. Compare both on validation gates:
   - schema validity
   - action accuracy
   - content span F1
   - fact evidence F1
   - indexable quality
   - recall_context hit@k
4. Keep the structured model as default unless the generative baseline clearly wins without JSON/schema failures.

## Canonical Training Format

Every converted dataset row must become:

```json
{
  "instruction": "Perform the PSM memory operation for the current input...",
  "input": {
    "operation": "remember | recall",
    "source_kind": "dataset_name",
    "source_id": "stable source id",
    "current_turn": {
      "speaker": "speaker name when known",
      "text": "current utterance",
      "timestamp": "source timestamp when known"
    },
    "prior_context": [],
    "memory_store": []
  },
  "output": {
    "action": "ignore | store_episodic | promote_semantic | update_existing | flag_conflict | flag_and_store | recall_context",
    "memory": null,
    "facts": [],
    "indexables": [],
    "updates": [],
    "conflicts": [],
    "recall": {},
    "reasoning": "brief grounded reason"
  }
}
```

Stored memories must include `indexables`.

Recall rows must use:

```json
{
  "action": "recall_context",
  "memory": null,
  "facts": [],
  "indexables": [],
  "updates": [],
  "conflicts": [],
  "recall": {
    "query_intent": "temporal_recall | fact_recall | memory_recall | inference_supported_recall",
    "selected_memory_ids": [],
    "selected_indexable_keys": [],
    "max_items": 5,
    "reasoning": "grounded selection reason"
  },
  "reasoning": "brief reason"
}
```

## Current Source Compatibility

The canonical training row is intentionally richer than the current runtime `StorageDecision` contract.

Current source code already supports these training fields:

- `action` values for storage/update/conflict decisions, except `recall_context`
- `memory.content`
- `memory.type`
- `memory.strength`
- `memory.decay_rate`
- `memory.emotional_weight`
- `memory.confidence`
- `memory.tags`
- temporal memory fields: `temporal_expression`, `resolved_time`, `resolved_time_confidence`
- extracted `facts` with evidence and temporal fields

Current source code does not yet fully support these training fields as first-class runtime objects:

- `recall_context` as a `MemoryAction`
- `indexables` as persisted memory records or recall graph nodes
- `updates` and `conflicts` as fully structured runtime side effects beyond current route handling

Decision before large-scale generation:

1. Keep the training schema richer than runtime, but add a conversion bridge:
   - `nano output -> StorageDecision`
   - `output.indexables -> sidecar indexable records`
   - `output.recall -> RecallPlan/ContextRender`
2. Or update runtime types and storage so indexables become first-class product data.

Recommended path:

- Add a small bridge first so dataset work can proceed.
- Add first-class indexable persistence before ONNX runtime integration.
- Do not train large datasets against fields that validation accepts but runtime will silently discard.

Compatibility rule:

Every generated example must be valid against `nano-psm/data-pipeline/schemas/psm-training-example.schema.json`, and every non-recall output must be convertible into the current `StorageDecision` shape without losing the core memory, facts, confidence, temporal metadata, or routing action.

## Indexables And Mnemonic Training

Indexables are not optional metadata. They are a primary training target for nano PSM.

The goal is to train the model to create compact reconstructive recall cues, not just generic tags.

Good indexables:

- are short lowercase hyphenated keys
- preserve speaker or domain specificity when useful
- compress the memory into a cue that can later reactivate the larger context
- include an evidence-backed reconstructive hint
- connect to facts, temporal anchors, entities, or product decisions

Examples:

```json
{
  "kind": "mnemonic",
  "key": "earworm-sparse-recall",
  "target_type": "semantic",
  "salience": 0.93,
  "reconstructive_hint": "User thinks mnemonic cue words can reconstruct larger memories and improve PSM recall.",
  "evidence_text": "brain can remember a small phrase or a mnemonic word which can extract the whole memory",
  "tags": ["indexables", "mnemonic_memory", "recall"]
}
```

`docs/indexables-conv.txt` should be converted into curated `local_psm` training examples.

Do not train it as a raw transcript dump. Convert it into:

- semantic memories about the indexables concept
- episodic decision memories about why PSM is pursuing mnemonic recall
- indexable generation examples
- recall examples where a cue selects the right memory

Quality rule:

Indexables must be reconstructive. A key such as `memory-system` is too generic. A key such as `earworm-sparse-recall` or `compressed-recall-handles` is better because it can reactivate the actual concept.

## Source Inventory

Use `nano-psm/data-pipeline/sources/psm-source-manifest.json` as the source manifest.

### LoCoMo

Role:

- long-term episodic continuity
- session timeline
- speaker-aware memories
- temporal recall

Local path:

```text
benchmark/locomo/data/locomo10.json
```

Conversion:

- Each conversation turn becomes a `remember` example.
- QA evidence ids become `recall_context` examples.
- Do not train ingestion rows directly from QA answers.
- Use QA answers only to validate/label recall-context selection.
- Preserve `dia_id`, `session`, and source timestamp.

Outputs:

- `store_episodic` for dated personal events
- `promote_semantic` for stable preferences/identity/hobbies
- `ignore` for greetings, compliments, questions, filler
- `recall_context` for QA evidence retrieval

### PersonaMem

Role:

- preference extraction
- latent identity/profile memory
- implicit personalization

Acquisition:

```powershell
hf download bowen-upenn/PersonaMem --repo-type dataset --local-dir nano-psm/data-pipeline/data/raw/personamem
hf download bowen-upenn/PersonaMem-v2 --repo-type dataset --local-dir nano-psm/data-pipeline/data/raw/personamem-v2
```

Conversion:

- Convert explicit preferences into `promote_semantic`.
- Convert weak/ambiguous preference signals into low-confidence semantic memories or `ignore`.
- Use persona/profile labels as evidence-backed facts only when directly supported by conversation.
- Generate indexables from preference handles, for example `concise-replies`, `vegetarian-travel`, `dark-mode-tools`.

Risks:

- Over-personalizing weak evidence.
- Treating inferred identity as explicit fact.

Quality rule:

- If evidence is implicit, keep confidence lower and avoid hard profile facts.

### Local PSM Memories And Decisions

Role:

- real product memory behavior
- real user preferences and project decisions
- failure analysis from previous benchmark runs
- mnemonic/indexable concept grounding
- current PSM design constraints

Local sources:

```text
user_memory.db
docs/indexables-conv.txt
docs/psm-memory-fine-tuning-plan.md
docs/product-aligned-psm-ingestion-retrieval-fix-plan.md
docs/session-hooks-and-lean-context-plan.md
nano-psm/data-pipeline/data/generated/*.jsonl
```

Conversion:

- Export current `episodic`, `semantic`, and `memory_facts` rows into `local_psm` examples.
- Convert high-confidence real decisions into `promote_semantic` or `store_episodic`.
- Convert previous failure reports into benchmark-result memories.
- Convert indexables discussion into curated mnemonic/indexable examples.
- Preserve source file, source row id, source timestamp, and evidence text where available.
- Do not include private raw transcripts unless they are intentionally selected and reviewed.

Outputs:

- `promote_semantic` for stable user/project preferences
- `store_episodic` for dated benchmark/session outcomes
- `update_existing` for corrected implementation decisions
- `flag_conflict` or `flag_and_store` for superseded assumptions
- `recall_context` rows that select real memories and indexable keys

Quality rule:

Real PSM memories are valuable because they reflect actual product decisions, but they must still pass the same evidence and privacy checks as public datasets.

### LongMemEval

Role:

- knowledge updates
- contradiction handling
- temporal reasoning
- abstention

Acquisition:

The paper and public references identify LongMemEval as the benchmark. Prefer an official or commonly used HF dataset if available in the current environment. Candidate loaders should be configurable because names may vary.

Expected raw path:

```text
nano-psm/data-pipeline/data/raw/longmemeval
```

Conversion:

- Convert changed user information into `update_existing`.
- Convert uncertain contradiction into `flag_conflict`.
- Convert certain contradiction plus useful new memory into `flag_and_store`.
- Convert no-evidence questions into `recall_context` with empty selection and abstention reasoning.

Examples:

- prior: "User lives in Boston"; new: "I moved to Seattle" -> `update_existing`
- prior: "User prefers React"; new: "I may stop using React" -> `flag_conflict`
- question has no supporting evidence -> recall selected ids `[]`

### REALTALK

Role:

- noisy real-world conversation
- fragmented statements
- messy multi-day continuity
- incomplete references

Acquisition:

Use official dataset instructions from the REALTALK paper/source. Keep raw files under:

```text
nano-psm/data-pipeline/data/raw/realtalk
```

Conversion:

- Train `ignore` heavily for chatter, incomplete referents, and non-durable messages.
- Store partial memories only when useful and grounded.
- Lower confidence for ambiguous references.
- Build recall rows from memory probing tasks where available.

Quality rule:

- Do not repair noisy text into unsupported clean facts.

### PerLTQA

Role:

- typed memory organization
- semantic vs episodic classification
- profile/social relationship/event memory types
- memory retrieval/synthesis

Acquisition:

Use the paper or official dataset release. Keep raw files under:

```text
nano-psm/data-pipeline/data/raw/perltqa
```

Conversion:

- Semantic memories map to `promote_semantic`.
- Episodic memories map to `store_episodic`.
- Retrieval labels map to `recall_context`.
- Memory classification labels should train memory type and tag/indexable heads.

Typed memory mapping:

```text
profile -> semantic
relationship -> semantic
preference -> semantic
event -> episodic
dialogue episode -> episodic
world/background fact -> semantic only if personal memory needs it
```

### User Preference 564K

Role:

- preference extraction bootstrapping
- large-scale preference-rule supervision

Acquisition:

```powershell
hf download blackhao0426/user-preference-564k --repo-type dataset --local-dir nano-psm/data-pipeline/data/raw/user-preference-564k
```

Conversion:

- Convert preference JSON/rules into `promote_semantic`.
- Preserve condition-action structure in facts or tags.
- Generate indexables from condition, domain, and action.
- Downsample aggressively to avoid preference examples overwhelming episodic/update learning.

Example conversion:

```json
{
  "memory": {
    "content": "User prefers concise answers when asking implementation questions.",
    "type": "semantic",
    "tags": ["preference", "concise_answers", "implementation"]
  },
  "facts": [
    {
      "subject": "User",
      "predicate": "prefers_response_style",
      "value": "concise answers for implementation questions",
      "inference_kind": "explicit",
      "evidence_text": "source preference rule text"
    }
  ],
  "indexables": [
    {
      "kind": "mnemonic",
      "key": "concise-implementation-answers",
      "target_type": "semantic",
      "salience": 0.85,
      "reconstructive_hint": "User prefers concise answers for implementation questions.",
      "evidence_text": "source preference rule text",
      "tags": ["preference"]
    }
  ]
}
```

## Acquisition Plan

### Step 1: Prepare Source Directories

```powershell
npm run ft:sources
```

This creates:

```text
nano-psm/data-pipeline/data/raw/personamem
nano-psm/data-pipeline/data/raw/longmemeval
nano-psm/data-pipeline/data/raw/realtalk
nano-psm/data-pipeline/data/raw/perltqa
nano-psm/data-pipeline/data/raw/user-preference-564k
```

### Step 2: Download Public HF Sources

Install:

```powershell
pip install -U huggingface_hub
hf auth login
```

Download:

```powershell
hf download bowen-upenn/PersonaMem --repo-type dataset --local-dir nano-psm/data-pipeline/data/raw/personamem
hf download bowen-upenn/PersonaMem-v2 --repo-type dataset --local-dir nano-psm/data-pipeline/data/raw/personamem-v2
hf download blackhao0426/user-preference-564k --repo-type dataset --local-dir nano-psm/data-pipeline/data/raw/user-preference-564k
```

For LongMemEval, REALTALK, and PerLTQA, first check the official release path and license. If an HF repo exists, add it to `psm-source-manifest.json`; otherwise document manual download steps in the source README.

### Step 3: Mirror To A Private HF Dataset Repo

Because Colab sessions can die and accounts may change, all raw and generated artifacts should sync to Hugging Face Hub.

Recommended repos:

```text
chkrishna2001/nano-psm-raw-sources
chkrishna2001/nano-psm
chkrishna2001/nano-psm-checkpoints
```

Upload raw source snapshots:

```powershell
hf upload chkrishna2001/nano-psm-raw-sources nano-psm/data-pipeline/data/raw . --repo-type dataset
```

Upload generated data:

```powershell
hf upload chkrishna2001/nano-psm hf-upload\nano-psm-merged-starter . --repo-type dataset
```

## Adapter Implementation Plan

Create one adapter per source:

```text
nano-psm/data-pipeline/src/adapters/
  locomo.mjs
  personamem.mjs
  local-psm.mjs
  longmemeval.mjs
  realtalk.mjs
  perltqa.mjs
  user-preference-564k.mjs
```

Create shared adapter utilities:

```text
nano-psm/data-pipeline/src/lib/
  psm-example.mjs
  indexables.mjs
  local-psm-export.mjs
  temporal.mjs
  evidence.mjs
  split.mjs
```

Adapter output contract:

```js
export async function generateExamples(options) {
  return [
    {
      id,
      instruction,
      input,
      output
    }
  ];
}
```

Main generator should become manifest-driven:

```powershell
node nano-psm\data-pipeline\src\generate-dataset.mjs `
  --manifest nano-psm\data-pipeline\sources\psm-source-manifest.json `
  --out nano-psm\data-pipeline\data\generated `
  --max-total 10000
```

Scale targets:

```text
phase_1_schema_and_runtime_alignment: 1k-5k examples
phase_2_quality_baseline: 10k examples
phase_3_full_mixed_dataset: 50k-100k examples
phase_4_scale_after_validation: only if quality gates and evals improve
```

Do not chase dataset size before adapters, source labels, indexables, and runtime conversion are stable.

## Dataset Balancing

The dataset should not mirror raw source proportions. It should be balanced for PSM capability.

Initial action mix:

```text
ignore: 20%
store_episodic: 20%
promote_semantic: 20%
recall_context: 20%
update_existing: 8%
flag_conflict: 7%
flag_and_store: 5%
```

Source mix:

```text
LoCoMo: 18%
PersonaMem: 15%
User Preference 564K: 12%
LongMemEval: 18%
PerLTQA: 13%
REALTALK: 10%
local PSM memories and decisions: 10%
synthetic hard cases and indexables: 4%
```

The large User Preference 564K dataset must be downsampled and deduplicated. It should bootstrap preference extraction, not dominate the model.

Local PSM data should be sampled deliberately, not simply dumped. Prefer rows that teach real product constraints, corrected decisions, benchmark failures, temporal grounding, and mnemonic/indexable behavior.

## Validation Gates

Run after every generation:

```powershell
npm run ft:validate -- nano-psm/data-pipeline/data/generated/train.jsonl nano-psm/data-pipeline/data/generated/validation.jsonl
```

Required gates:

- valid JSONL
- output uses canonical keys only
- `facts` always include evidence
- `indexables` always use lowercase hyphenated keys
- named speakers do not become generic `User`
- `ignore` rows have `memory:null`
- `recall_context` rows have `memory:null`
- recall rows select ids present in `input.memory_store`
- no source QA answer leakage into ingestion labels
- temporal fields parse when present
- source ids are preserved

Quality gate command:

```powershell
node nano-psm\data-pipeline\src\gate-dataset.mjs `
  --train nano-psm\data-pipeline\data\generated\train.jsonl `
  --validation nano-psm\data-pipeline\data\generated\validation.jsonl `
  --out nano-psm\data-pipeline\reports\gate-current
```

The quality gate must pass before upload or Colab training. It writes:

```text
dataset-summary.json
action-mix.json
source-mix.json
quality-audit.json
review-sample.jsonl
```

Do not train on rows with quality failures. Warnings are review items; repeated duplicate memories, weak indexables, or source imbalance should be fixed before scaling.

Add source-specific reports:

```text
nano-psm/data-pipeline/reports/
  dataset-summary.json
  source-mix.json
  action-mix.json
  validation-errors.json
  leakage-audit.json
```

## Nano Model Training Plan

### Phase 0: Baseline Dataset

Use current generated data first:

```powershell
node nano-psm\data-pipeline\src\generate-dataset.mjs `
  --locomo benchmark\locomo\data\locomo10.json `
  --out nano-psm\data-pipeline\data\generated `
  --limit 500 `
  --recall-limit 250 `
  --synthetic-count 250
```

This validates the format and training loop before external adapters are complete.

### Phase 1: Small Structured Encoder

Create:

```text
nano-psm/
  configs/
    debug-4m.json
    primary-10m.json
  data/
  notebooks/
  scripts/
  src/nano_psm/
    train.py
    model.py
    dataset.py
    evaluate.py
    export_onnx.py
```

Training inputs:

- serialized `input` object
- optional target JSON for auxiliary text fields
- candidate memory ids/indexable keys for recall rows

Training labels:

- action class
- memory type class
- memory content span
- fact/evidence spans
- regression scores
- indexable token labels
- recall selected candidate labels

Loss:

```text
total =
  action_ce
  + memory_type_ce
  + span_ce
  + fact_span_ce
  + score_mse
  + indexable_loss
  + recall_selection_bce
```

Weight temporal, update/conflict, indexables, and recall rows higher than easy ignore rows.

### Phase 2: Optional Tiny Generative Comparison

Train a small generative model only as a comparison:

- character/BPE tokenizer
- JSON target generation
- strict schema validation after generation

This is expected to be less reliable than structured heads, but it gives a useful baseline.

### Phase 3: ONNX Export

Export the structured model:

```powershell
python nano-psm/src/nano_psm/export_onnx.py `
  --checkpoint checkpoints/best.pt `
  --out dist/nano-psm/model.onnx
```

Runtime integration:

- TypeScript loads ONNX model.
- If confidence is high, use nano PSM result.
- If confidence is low or action requires abstraction, fall back to current LLM-backed PSM.

## Colab T4 Resumable Training

### Principles

Colab can disconnect or exhaust quota. Training must treat each session as disposable.

Persistent state lives in Hugging Face Hub, not Colab disk.

Persist:

- dataset snapshots
- tokenizer
- config
- checkpoints
- optimizer state
- scheduler state
- best metrics
- run log

### HF Repos

Use three repos:

```text
HF_DATASET_REPO=chkrishna2001/nano-psm
HF_RAW_REPO=chkrishna2001/nano-psm-raw-sources
HF_CHECKPOINT_REPO=chkrishna2001/nano-psm-checkpoints
```

### Colab Bootstrap Cell

```python
!pip install -U huggingface_hub datasets torch safetensors onnx onnxruntime

from huggingface_hub import login, snapshot_download
login()

HF_DATASET_REPO = "chkrishna2001/nano-psm"
HF_CHECKPOINT_REPO = "chkrishna2001/nano-psm-checkpoints"

dataset_dir = snapshot_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    local_dir="/content/psm-data",
    resume_download=True
)

try:
    checkpoint_dir = snapshot_download(
        repo_id=HF_CHECKPOINT_REPO,
        repo_type="model",
        local_dir="/content/nano-psm-checkpoints",
        resume_download=True
    )
except Exception:
    checkpoint_dir = "/content/nano-psm-checkpoints"
```

### Training Command

```python
!python nano-psm/src/nano_psm/train.py \
  --train /content/psm-data/train.jsonl \
  --validation /content/psm-data/validation.jsonl \
  --checkpoint-dir /content/nano-psm-checkpoints \
  --resume auto \
  --device cuda \
  --batch-size 32 \
  --grad-accum 2 \
  --max-steps 20000 \
  --save-every 500 \
  --eval-every 500
```

### Checkpoint Upload Cell

Run after every save or at the end of a session:

```python
from huggingface_hub import upload_folder

upload_folder(
    repo_id=HF_CHECKPOINT_REPO,
    repo_type="model",
    folder_path="/content/nano-psm-checkpoints",
    path_in_repo=".",
    commit_message="sync nano psm checkpoint"
)
```

### Auto-Sync During Training

`train.py` should support:

```text
--hf-checkpoint-repo <repo>
--upload-every 500
```

Every upload should include:

```text
checkpoint-last.pt
checkpoint-best.pt
optimizer-last.pt
scheduler-last.pt
tokenizer.json
config.json
metrics.jsonl
trainer-state.json
```

### Resume Logic

On startup:

1. If `--resume auto`, look for `checkpoint-last.pt`.
2. Load model weights.
3. Load optimizer state.
4. Load scheduler state.
5. Load global step, epoch, best metric, RNG state.
6. Continue from the next batch.

If optimizer state is missing:

- load weights only
- restart optimizer
- mark run as `partial_resume`

### Cross-Account Resume

If one Colab account exhausts quota:

1. Open Colab in another account.
2. Authenticate to the same HF repos.
3. Snapshot-download dataset and checkpoints.
4. Run the same training command with `--resume auto`.
5. Continue uploading to the same checkpoint repo.

No Google Drive dependency is required.

## Metrics

Evaluate every checkpoint on validation split.

Core metrics:

```text
action_accuracy
ignore_precision
ignore_recall
store_precision
speaker_grounding_accuracy
memory_type_accuracy
content_span_f1
fact_evidence_f1
temporal_accuracy
update_conflict_accuracy
indexable_key_exact
indexable_token_f1
recall_context_hit_at_1
recall_context_hit_at_5
unsupported_fact_rate
generic_user_leakage_rate
```

Checkpoint selection metric:

```text
score =
  0.20 * action_accuracy
  + 0.15 * content_span_f1
  + 0.15 * recall_context_hit_at_5
  + 0.15 * indexable_token_f1
  + 0.10 * temporal_accuracy
  + 0.10 * update_conflict_accuracy
  + 0.10 * fact_evidence_f1
  - 0.15 * unsupported_fact_rate
  - 0.10 * generic_user_leakage_rate
```

Do not select a model that performs well only by predicting `ignore`.

## Autoresearch Use

Use Karpathy-style autoresearch after the baseline training loop works.

Autoresearch should optimize:

- model depth/width
- tokenization strategy
- loss weights
- class-balanced sampling
- curriculum schedule
- indexable auxiliary loss
- recall selection head design

It must not change:

- output schema
- validation gates
- source evidence requirements
- no-leakage rules

Research prompt:

```text
nano-psm/data-pipeline/autoresearch-program.md
```

Run strategy:

1. Baseline train for a short budget.
2. Agent proposes one training/model change.
3. Train for fixed time, for example 10-20 minutes on T4.
4. Evaluate using validation score.
5. Keep change only if validation score improves and gates pass.
6. Upload artifacts and experiment log to HF.

## Implementation Checklist

### Dataset

- [ ] Keep `psm-source-manifest.json` up to date.
- [ ] Decide and document runtime conversion for `indexables` and `recall_context`.
- [ ] Add a nano-output-to-runtime bridge for current `StorageDecision`, `RecallPlan`, and sidecar indexables.
- [ ] Implement `src/adapters/locomo.mjs`.
- [ ] Implement `src/adapters/personamem.mjs`.
- [ ] Implement `src/adapters/local-psm.mjs`.
- [ ] Add local PSM export for `user_memory.db`, selected docs, and prior generated examples.
- [ ] Convert `docs/indexables-conv.txt` into curated mnemonic/indexable examples.
- [ ] Implement `src/adapters/longmemeval.mjs`.
- [ ] Implement `src/adapters/realtalk.mjs`.
- [ ] Implement `src/adapters/perltqa.mjs`.
- [ ] Implement `src/adapters/user-preference-564k.mjs`.
- [ ] Add manifest-driven generator.
- [ ] Add source/action balancing.
- [ ] Add leakage audit.
- [ ] Upload generated data to HF dataset repo.

### Model

- [ ] Create `nano-psm/src/nano_psm/model.py`.
- [ ] Create `nano-psm/src/nano_psm/dataset.py`.
- [ ] Create `nano-psm/src/nano_psm/train.py`.
- [ ] Create `nano-psm/src/nano_psm/evaluate.py`.
- [ ] Create tokenizer training script.
- [ ] Add resumable checkpoint state.
- [ ] Add HF checkpoint upload.
- [ ] Add ONNX export.
- [ ] Add TypeScript inference spike.

### Colab

- [ ] Create `nano-psm/notebooks/nano-psm-train-colab.ipynb`.
- [ ] Add HF login/setup cell.
- [ ] Add dataset snapshot cell.
- [ ] Add checkpoint snapshot cell.
- [ ] Add training cell with `--resume auto`.
- [ ] Add upload checkpoint cell.
- [ ] Add validation/eval cell.

### Product Integration

- [ ] Add first-class indexable persistence or a temporary sidecar store.
- [ ] Add recall bridge from `recall_context` output to current context rendering.
- [ ] Add optional nano PSM runtime behind config.
- [ ] Use nano PSM for high-confidence store/ignore/update/conflict classification.
- [ ] Fall back to LLM-backed PSM for low-confidence or abstractive cases.
- [ ] Compare LOCOMO answer accuracy and memory write quality before switching defaults.

## Milestones

### Milestone 1: Data Foundation

Deliverables:

- compatibility decision for runtime indexables and `recall_context`
- all source folders prepared
- at least LoCoMo + local PSM + synthetic + User Preference 564K adapters
- 10k valid examples
- curated examples from `docs/indexables-conv.txt`
- source/action balance report
- runtime conversion check for non-recall rows

### Milestone 2: Resumable T4 Training

Deliverables:

- 4M debug config trains end-to-end
- 10M primary config trains on Colab T4
- checkpoint resume works after runtime restart
- checkpoint repo sync works across accounts
- baseline validation metrics generated

### Milestone 3: Full Dataset

Deliverables:

- 50k-100k examples across all sources
- validation gates pass
- recall-context and indexable metrics stable
- local PSM examples remain capped and reviewed so they improve quality without overfitting

### Milestone 4: Nano Runtime

Deliverables:

- best checkpoint exported to ONNX
- TypeScript inference prototype
- indexables persisted or bridged into recall
- side-by-side comparison with current LLM-backed PSM

## Immediate Next Step

First align the schema with the current source code path:

1. Define the runtime bridge for `indexables` and `recall_context`.
2. Implement local PSM export and curated indexables examples.
3. Split the current monolithic generator into manifest-driven adapters.
4. Generate a 1k-5k compatibility dataset and validate it.
5. Only then generate the 10k baseline dataset and begin the 4M debug training run.

Training the 10M model before this alignment would optimize against an incomplete dataset and silently discard the most important mnemonic/indexable fields.
