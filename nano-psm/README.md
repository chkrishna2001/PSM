# Nano PSM

Nano PSM is the local structured memory-operation model for PSM.

It is not a general chatbot. It is trained to classify and produce memory operations:

- ignore/store/update/conflict decisions
- episodic vs semantic memory typing
- source-grounded memory and fact extraction
- mnemonic/indexable cue generation
- recall-context selection from candidate memories and indexables

## Layout

```text
nano-psm/
  configs/                 model and training configs
  data/                    local dataset mount point, not committed
  notebooks/               Colab and experiment notebooks
  scripts/                 developer convenience scripts
  src/nano_psm/            model, dataset, training, eval, export code
  checkpoints/             local checkpoints, not committed
```

## First Milestone

The first runnable milestone is a compatibility training pass:

1. Generate 1k-5k validated examples from the canonical PSM training schema.
2. Train the debug config end to end.
3. Confirm action, memory type, indexable, and recall-selection metrics are emitted.
4. Move to the primary 10M config only after the data/runtime contract is stable.

## Smoke Command

Use an existing generated dataset:

```powershell
python nano-psm\src\nano_psm\train.py `
  --config nano-psm\configs\debug-4m.json `
  --train nano-psm\data-pipeline\data\generated-local-psm\train.jsonl `
  --validation nano-psm\data-pipeline\data\generated-local-psm\validation.jsonl `
  --checkpoint-dir nano-psm\checkpoints
```

## Local CPU Smoke

For local verification only:

```powershell
python -m venv nano-psm\.venv
nano-psm\.venv\Scripts\python -m pip install --upgrade pip
nano-psm\.venv\Scripts\python -m pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
```

Run one CPU step:

```powershell
nano-psm\.venv\Scripts\python nano-psm\src\nano_psm\train.py `
  --config nano-psm\configs\debug-4m.json `
  --train hf-upload\nano-psm-merged-starter\train.jsonl `
  --validation hf-upload\nano-psm-merged-starter\validation.jsonl `
  --checkpoint-dir nano-psm\checkpoints\local-smoke `
  --max-steps 1 `
  --eval-every 1 `
  --save-every 1 `
  --device cpu
```

This confirms code correctness, not model quality. Real training should run in Colab.

## Data Quality Gate

Training data must pass the fine-tune data gate before upload or Colab training:

```powershell
node nano-psm\data-pipeline\src\gate-dataset.mjs `
  --train nano-psm\data-pipeline\data\generated-local-psm\train.jsonl `
  --validation nano-psm\data-pipeline\data\generated-local-psm\validation.jsonl `
  --out nano-psm\data-pipeline\reports\gate-local-psm
```

The gate writes:

```text
dataset-summary.json
action-mix.json
source-mix.json
quality-audit.json
review-sample.jsonl
```

Do not train on data with quality failures.

After a gate passes, prepare the HF upload folder:

```powershell
node nano-psm\data-pipeline\src\prepare-hf-dataset.mjs `
  --data-dir nano-psm\data-pipeline\data\merged-starter `
  --report-dir nano-psm\data-pipeline\reports\gate-merged-starter `
  --out hf-upload\nano-psm-merged-starter `
  --name nano-psm-merged-starter-gated
```

Current starter dataset repo:

```text
chkrishna2001/nano-psm
```

## Colab

Use `notebooks/nano-psm-data-and-train-colab.ipynb` for Hugging Face dataset sync, checkpoint resume, and Colab training.

The first training pass should use:

```text
HF_DATASET_REPO=chkrishna2001/nano-psm
HF_CHECKPOINT_REPO=chkrishna2001/nano-psm-checkpoints
config=nano-psm/configs/debug-4m.json
max_steps=500
```

Only increase `max_steps` or move to `configs/primary-10m.json` after validation metrics and checkpoint upload work.

Inspect validation mistakes before scaling data or model size:

```powershell
python nano-psm\src\nano_psm\inspect_predictions.py `
  --config nano-psm\configs\debug-4m.json `
  --validation hf-upload\nano-psm-merged-starter\validation.jsonl `
  --checkpoint nano-psm\checkpoints\local-smoke\checkpoint-best.pt `
  --device cpu `
  --limit 30
```

## Configs

- `configs/debug-4m.json`: fast smoke-test model.
- `configs/primary-10m.json`: first serious model target.
