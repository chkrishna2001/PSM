# PSM Model

Experimental generative PSM model package.

**Training:** see [docs/psm-model/training-playbook.md](../docs/psm-model/training-playbook.md) and [docs/psm-model/session-log.md](../docs/psm-model/session-log.md).

This package is intentionally separate from `nano-psm`. The first goal is not model size; it is a strict, testable path from model text output to validated PSM storage JSON.

## Current Slice

- strict `StorageDecision` JSON validator.
- hand-authored direct probe fixtures.
- canonical training-row JSONL gate.
- prompt rendering for storage generation.
- dependency-free byte, DSL, pattern, and BPE tokenizer paths.
- tiny decoder-only transformer code path, gated behind optional PyTorch.
- strict generation gate for parse/schema/action/type/content/fact exactness.
- stdlib unit tests for schema, data, tokenizer, model, training, and gate behavior.

## Commands

Run the current tests from repo root:

```powershell
python -m unittest discover -s psm-model/tests
```

Evaluate direct probes:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.evaluate psm-model\data\probes\direct_probes.jsonl
```

Gate canonical dataset rows:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.gate_dataset psm-model\data\probes\direct_probes.jsonl
```

Convert existing Nano storage rows into real generative storage data:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.convert_nano_dataset psm-model\data\real-v1 nano-psm\data-pipeline\data\retention-blend-codex-84k\train.jsonl nano-psm\data-pipeline\data\fast-mixed-reviewed-v2\train.jsonl nano-psm\data-pipeline\data\codex-sessions-2026\train.jsonl
```

Current `real-v1` conversion:

```text
read: 13,086
accepted storage rows: 8,082
skipped recall_context rows: 2,709
skipped duplicate rows: 2,295
splits: train 7,150 / validation 645 / test 287
```

Train a tokenizer:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train_tokenizer psm-model\data\probes\direct_probes.jsonl psm-model\tokenizers\probe-pattern.json --kind pattern --vocab-size 512 --output-format tagged
```

Tokenizer status after fixing evaluator tokenizer plumbing:

```text
byte tokenizer:    passes direct-probe generation gate; avg generated output 310.8 tokens.
DSL tokenizer:     passes direct-probe generation gate; 8.8% train-token savings; avg generated output 278.2 tokens.
pattern tokenizer: passes direct-probe generation gate; 66.0% train-token savings; avg generated output 106.2 tokens.
BPE tokenizer:     passes direct-probe generation gate; 67.0% train-token savings; avg generated output 172.0 tokens.
```

The pattern tokenizer is the current best debug tokenizer because it keeps exact generated-output validity while reducing output length the most on the direct probes.

Train the current real-v1 tokenizer:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train_tokenizer psm-model\data\real-v1\train.jsonl psm-model\tokenizers\real-v1-pattern.json --kind pattern --vocab-size 4096 --output-format tagged
```

Current real-v1 tokenizer result:

```text
rows: 7,150
byte tokens: 9,699,025
pattern tokens: 4,070,244
token savings: 58.0%
```

Filter real-v1 to the 2048 context budget:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.filter_by_token_budget psm-model\data\real-v1\train.jsonl psm-model\data\real-v1-ctx2048\train.jsonl --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format tagged
python -m psm_model.filter_by_token_budget psm-model\data\real-v1\validation.jsonl psm-model\data\real-v1-ctx2048\validation.jsonl --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format tagged
python -m psm_model.filter_by_token_budget psm-model\data\real-v1\test.jsonl psm-model\data\real-v1-ctx2048\test.jsonl --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format tagged
```

Current context-safe split:

```text
train: 7,078 rows
validation: 640 rows
test: 285 rows
```

Prepare gated rows into prompt/completion text:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.prepare_dataset psm-model\data\probes\direct_probes.jsonl out\psm-model-probes.train.jsonl
```

Train a tiny debug checkpoint after installing PyTorch:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\probes\direct_probes.jsonl --steps 300 --batch-size 5 --learning-rate 0.001 --tokenizer psm-model\tokenizers\probe-pattern.json --out psm-model\checkpoints\debug-probe-pass.pt
```

Validate the 50M preset without writing a checkpoint:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\probes\direct_probes.jsonl --preset 50m --dry-run
```

Run a one-step 50M trainability smoke:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\probes\direct_probes.jsonl --preset 50m --context-length 1536 --steps 1 --batch-size 1 --learning-rate 0.001 --tokenizer psm-model\tokenizers\probe-pattern.json --out psm-model\checkpoints\50m-trainability-smoke.pt
```

This proves the 50M training path executes on CPU. It is not a model-quality gate.

Run the current real-v1 50M smoke:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.train psm-model\data\real-v1-ctx2048\train.jsonl --preset 50m --steps 1 --batch-size 1 --learning-rate 0.001 --tokenizer psm-model\tokenizers\real-v1-pattern.json --out psm-model\checkpoints\real-v1-50m-smoke.pt
```

Current real-v1 50M smoke:

```text
examples: 7,078
vocab_size: 4,096
parameter_estimate: 53,535,744
initial_loss/final_loss for one step: 8.218459
```


Run the end-to-end overfit smoke:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.smoke_overfit psm-model\data\probes\direct_probes.jsonl --probe-id ignore_noise --steps 120
```

This smoke is the first readiness gate: the model must train, generate JSON from a prompt, and pass strict schema validation. Loss-only success is not enough.

Compare model output formats:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.compare_formats psm-model\data\probes\direct_probes.jsonl
```

The current experiment compares full JSON, compact JSON arrays, pipe-heavy tagged DSL, and `@tag` line DSL. The model may emit the lean format, but runtime still expands it to normal `StorageDecision` JSON before validation and writes.

Run the overfit smoke with `@tag` output:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.smoke_overfit psm-model\data\probes\direct_probes.jsonl --probe-id ignore_noise --steps 120 --output-format at_tag
```

Run the all-probe generation gate:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.eval_generation psm-model\data\probes\direct_probes.jsonl --output-format tagged --steps 300
```

Gate a saved checkpoint:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.eval_checkpoint psm-model\checkpoints\debug-probe-pass.pt psm-model\data\probes\direct_probes.jsonl --max-new-tokens 500
```

This gate requires all of the following to be exact on direct probes:

```text
parse_valid_rate
schema_valid_rate
action_accuracy
memory_type_accuracy
memory_content_exact_rate
fact_count_accuracy
facts_exact_rate
```

That means "passed" means the model generated a valid storage decision and the generated classification, memory content, and fact/evidence fields matched the expected output.

Use `--eval` to score a separate held-out file:

```powershell
$env:PYTHONPATH='psm-model\src'
python -m psm_model.eval_generation out\seed.train.jsonl --eval out\seed.validation.jsonl --output-format tagged --steps 300
```

Current direct-probe generation gate for `--output-format tagged` with the pattern tokenizer:

```text
parse_valid_rate: 1.0
schema_valid_rate: 1.0
action_accuracy: 1.0
memory_type_accuracy: 1.0
memory_content_exact_rate: 1.0
fact_count_accuracy: 1.0
facts_exact_rate: 1.0
avg_generated_tokens: 106.2
```

The saved-checkpoint gate was verified both ways:

```text
undertrained 5-step debug checkpoint: failed generated-output gate.
300-step debug checkpoint: passed generated-output gate with exact action/type/content/facts on all 5 direct probes.
```

## Prod memory (isolated)

The production-memory initiative (grounding eval, future `prod-extraction-v1` curriculum) lives in **`prod-memory/`** — not mixed with gate/train code in `src/psm_model/` or LoCoMo in `benchmark/locomo/`.

See [prod-memory/README.md](prod-memory/README.md).

## Claude Worker Rule

Claude CLI workers must use isolated git worktrees under `.worktrees/`. Codex owns the main worktree and reviews any worker diff before merging.
