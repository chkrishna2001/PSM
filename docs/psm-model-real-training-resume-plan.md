# PSM Model Real Training Resume Plan

Date written: 2026-06-01

## Current Goal

Train a successful CPU-first generative PSM storage model around 50M parameters.

Success does **not** mean loss goes down. Success means generated output is correct:

- output parses.
- output schema-validates.
- action/classification is correct.
- memory type is correct.
- memory content is correct.
- fact count is correct.
- fact subject/predicate/value/evidence text are correct.

The model is for storage decisions first:

```text
input conversation/source/context -> PSM StorageDecision
```

Recall/context planning is intentionally excluded from this first storage model. `recall_context` rows are a different task and need their own output schema and gates later.

## Main Decisions Made Today

1. Use a new package: `psm-model`.
   - Do not continue the failed `nano-psm` classifier architecture as the production model.
   - Use Nano datasets only as source data after conversion and validation.

2. Output format:
   - Runtime contract remains `StorageDecision` JSON.
   - Model training target defaults to pipe-tagged DSL (`--output-format tagged`) because it is smaller than JSON and has strict parser/validator expansion.
   - Tagged output is never written directly. It must parse, expand, and pass schema validation.

3. Tokenizer:
   - Default for real training is `psm-model/tokenizers/real-v1-pattern.json`.
   - Tokenizer kind: `pattern`.
   - Vocab size: `4096`.
   - Reason: pattern tokenizer passed exact generation gates and gives strong token reduction.

4. Model:
   - Real target preset is `50m`.
   - Current 50M config with real-v1 tokenizer:

```text
context_length: 2048
n_layer: 16
n_head: 8
n_embd: 512
vocab_size: 4096
parameter_estimate: 53,535,744
```

5. Data:
   - Use existing Nano-format datasets only after conversion to canonical `psm-model` rows.
   - Drop `recall_context` for this storage run.
   - Deduplicate converted rows.
   - Filter rows exceeding 2048-token training budget.

6. Gate meaning:
   - When we say "passed", it must mean proper generated output and proper classification/content, not just counts.
   - Direct probes require exact pass.
   - Held-out real validation may need practical thresholds later, but direct probes stay exact.

## Code/Artifacts Created Today

Package and tools:

- `psm-model/src/psm_model/convert_nano_dataset.py`
  - Converts Nano JSONL rows into canonical `id/input/expected/source/split` rows.
  - Removes unsupported `recall_context`.
  - Strips fields not supported by current storage schema (`updates`, `conflicts`, `indexables`) from the training target.
  - Creates deterministic train/validation/test splits.

- `psm-model/src/psm_model/filter_by_token_budget.py`
  - Filters rows to a tokenizer/context token budget.
  - Used to create `real-v1-ctx2048`.

- `psm-model/src/psm_model/eval_checkpoint.py`
  - Gates saved checkpoints on generated-output quality.

- `psm-model/src/psm_model/gates.py`
  - Defines exact direct-probe thresholds.

Important generated artifacts:

- `psm-model/data/real-v1/`
- `psm-model/data/real-v1-ctx2048/`
- `psm-model/tokenizers/real-v1-pattern.json`
- smoke checkpoints under `psm-model/checkpoints/`

Checkpoint files are ignored by git.

## Verified Dataset State

Source datasets converted:

```text
nano-psm/data-pipeline/data/retention-blend-codex-84k/train.jsonl
nano-psm/data-pipeline/data/fast-mixed-reviewed-v2/train.jsonl
nano-psm/data-pipeline/data/codex-sessions-2026/train.jsonl
```

Conversion result:

```text
read: 13,086
accepted storage rows: 8,082
skipped recall_context: 2,709
skipped duplicates: 2,295
real-v1 splits:
  train: 7,150
  validation: 645
  test: 287
```

Context-safe filtered dataset:

```text
psm-model/data/real-v1-ctx2048/train.jsonl       7,078 rows
psm-model/data/real-v1-ctx2048/validation.jsonl    640 rows
psm-model/data/real-v1-ctx2048/test.jsonl          285 rows
```

All three context-safe splits passed `psm_model.gate_dataset`.

Action balance in context-safe train split:

```text
flag_and_store:    290
flag_conflict:     273
ignore:          2,349
promote_semantic:2,403
store_episodic:  1,337
update_existing:   426
```

Memory type balance in context-safe train split:

```text
episodic: 1,358
none:     2,349
semantic: 3,371
```

## Verified Tokenizer State

Real-v1 tokenizer command:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train_tokenizer psm-model\data\real-v1\train.jsonl psm-model\tokenizers\real-v1-pattern.json --kind pattern --vocab-size 4096 --output-format tagged
```

Result:

```text
actual_vocab_size: 4096
byte_tokens: 9,699,025
trained_tokens: 4,070,244
token_savings: 58.0%
```

This is the default tokenizer for tomorrow unless we intentionally rerun tokenizer experiments.

## Verified Training State

50M dry-run on real-v1:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\real-v1\train.jsonl --preset 50m --dry-run --tokenizer psm-model\tokenizers\real-v1-pattern.json
```

Result:

```text
examples: 7,150
vocab_size: 4096
parameter_estimate: 53,535,744
```

50M one-step smoke on context-safe real-v1:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\real-v1-ctx2048\train.jsonl --preset 50m --steps 1 --batch-size 1 --learning-rate 0.001 --tokenizer psm-model\tokenizers\real-v1-pattern.json --out psm-model\checkpoints\real-v1-50m-smoke.pt
```

Result:

```text
examples: 7,078
parameter_estimate: 53,535,744
initial_loss/final_loss for one step: 8.218459
```

This proves the real 50M training path executes. It is not quality proof.

## Verified Tests/Gates

Last full test suite:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m unittest discover -s psm-model/tests
```

Result:

```text
62 tests OK
```

Known warning:

```text
PyTorch warning: Failed to initialize NumPy: No module named 'numpy'
```

This warning did not block tests or training. Installing NumPy is optional cleanup.

## What To Do Tomorrow

### Step 1: Recheck Worktree and Tests

Run:

```powershell
git status --short
$env:PYTHONPATH='psm-model\src'
python -m unittest discover -s psm-model/tests
```

Expected:

- tests pass.
- `psm-model/` is still untracked unless committed.
- docs and `.gitignore` are modified.

### Step 2: Add Resume/Checkpoint Continuation Before Long Training

Do this before any multi-hour training.

Required changes:

- `train.py` should support:
  - `--resume CHECKPOINT`
  - `--save-every N`
  - `--metrics-out PATH`
  - periodic checkpoint naming like `real-v1-50m-step-000500.pt`
- saved metadata should include:
  - dataset path.
  - tokenizer path.
  - output format.
  - preset/config.
  - completed steps.
  - loss history summary.

Reason:

```text
Long CPU training without resume is too fragile.
```

### Step 3: Add Periodic Validation During Training

Add or script:

- direct-probe gate after checkpoint save.
- small held-out validation subset gate.
- validation loss if possible.

Do not require full validation generation every few steps because it may be slow. Use:

```text
direct probes every checkpoint
small validation sample every checkpoint
full validation at major checkpoints
```

### Step 4: Run First Real 50M Training Attempt

Start conservatively:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\real-v1-ctx2048\train.jsonl --preset 50m --steps 500 --batch-size 1 --learning-rate 0.001 --tokenizer psm-model\tokenizers\real-v1-pattern.json --out psm-model\checkpoints\real-v1-50m-step-000500.pt
```

If resume/save-every is implemented, prefer the resumable command instead.

Watch for:

- loss decreasing.
- no row-length failures.
- no checkpoint write failures.
- CPU time per step.

### Step 5: Gate The First Checkpoint

Direct probes:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.eval_checkpoint psm-model\checkpoints\real-v1-50m-step-000500.pt psm-model\data\probes\direct_probes.jsonl --max-new-tokens 500
```

Real validation sample/full validation:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.eval_checkpoint psm-model\checkpoints\real-v1-50m-step-000500.pt psm-model\data\real-v1-ctx2048\validation.jsonl --max-new-tokens 800
```

Interpretation:

- Direct probes should eventually become exact.
- Early validation will probably fail; use the report to identify failure categories.
- Do not call a checkpoint successful unless generated output is valid and correct.

### Step 6: Analyze Failures

Look at per-row reports from `eval_checkpoint`.

Classify failures:

- invalid parse.
- schema invalid.
- wrong action.
- wrong memory type.
- generated generic memory content.
- missing facts.
- wrong evidence text.
- too long / no `END`.

This determines whether to adjust:

- data quality.
- tokenizer.
- learning rate.
- output format.
- model size/context.
- grammar-constrained decoding.

### Step 7: Decide Whether To Use Paid/Teacher Model

Do **not** send all data through a paid model immediately.

Use a paid/strong model in targeted ways:

1. Audit/relabel 500-2,000 high-value rows.
2. Generate more examples for weak categories:
   - `flag_conflict`
   - `update_existing`
   - temporal episodic memory
   - multi-fact extraction
   - hard ignore/noise
3. Judge model outputs on failed validation rows.

Only scale paid labeling after the first 50M checkpoint shows the main failure modes.

### Step 8: Next Dataset Improvements

Likely next data work:

- increase `flag_conflict` and `update_existing` counts.
- add more direct probes for each action type.
- add temporal probes.
- add multi-fact probes.
- add negative probes where facts must not be invented.
- eventually add recall as a separate task/schema, not mixed into storage.

## Commands To Recreate Current Data

Convert:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.convert_nano_dataset psm-model\data\real-v1 nano-psm\data-pipeline\data\retention-blend-codex-84k\train.jsonl nano-psm\data-pipeline\data\fast-mixed-reviewed-v2\train.jsonl nano-psm\data-pipeline\data\codex-sessions-2026\train.jsonl
```

Train tokenizer:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train_tokenizer psm-model\data\real-v1\train.jsonl psm-model\tokenizers\real-v1-pattern.json --kind pattern --vocab-size 4096 --output-format tagged
```

Filter context:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.filter_by_token_budget psm-model\data\real-v1\train.jsonl psm-model\data\real-v1-ctx2048\train.jsonl --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format tagged
python -m psm_model.filter_by_token_budget psm-model\data\real-v1\validation.jsonl psm-model\data\real-v1-ctx2048\validation.jsonl --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format tagged
python -m psm_model.filter_by_token_budget psm-model\data\real-v1\test.jsonl psm-model\data\real-v1-ctx2048\test.jsonl --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format tagged
```

Gate:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.gate_dataset psm-model\data\real-v1-ctx2048\train.jsonl
python -m psm_model.gate_dataset psm-model\data\real-v1-ctx2048\validation.jsonl
python -m psm_model.gate_dataset psm-model\data\real-v1-ctx2048\test.jsonl
```

50M smoke:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\real-v1-ctx2048\train.jsonl --preset 50m --steps 1 --batch-size 1 --learning-rate 0.001 --tokenizer psm-model\tokenizers\real-v1-pattern.json --out psm-model\checkpoints\real-v1-50m-smoke.pt
```

## Resume Priority

Tomorrow, start with:

```text
Implement resume/save-every/metrics in train.py.
```

Then run:

```text
real-v1 50M training for 500 steps.
```

Then gate:

```text
direct probes + validation sample.
```
