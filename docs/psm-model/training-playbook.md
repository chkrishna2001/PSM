# PSM 50M training playbook

Single source of truth for the generative storage model restart.

## Goal

One ~50M `psm-model` decoder that emits schema-valid `StorageDecision` output: action, memory type/content, facts, temporal fields. The standalone action classifier is **diagnostic/bridge only**.

## Success rule

A training step succeeds only when **model-only** action metrics improve on held-out probes — not when loss drops or output is merely valid.

## Runtime

| Environment | Use |
|-------------|-----|
| Local machine | Gate 0–2, classifier, 500-step proof |
| Hugging Face | Private repos; upload gated data/checkpoints after local gates |
| Colab | Long runs **only after** local Gate 2 passes |

Local defaults:

```powershell
$env:PYTHONPATH='psm-model\src'
--device auto
--cuda-memory-fraction 0.5
```

## Checkpoint denylist (never `--resume` for main path)

```text
real-v2-50m-concept-repair-step-005300.pt
real-v2-50m-step-001200.pt
real-v2-50m-action-first-v1-step-003100.pt
real-v2-50m-action-head-repair.pt
real-v2-50m-action-head-freeze.pt
real-v2-50m-action-head-freeze-v2.pt
```

Also see `psm-model/checkpoints/DENYLIST.txt`.

## Gate 0 — Data

```powershell
$env:PYTHONPATH='psm-model\src'
.\.venv\Scripts\python.exe -m psm_model.filter_label_risks `
  psm-model\data\curriculum\psm-50m-full-storage-v1.jsonl `
  psm-model\data\curriculum\psm-50m-full-storage-v1-filtered.jsonl `
  --drop-severity high

.\.venv\Scripts\python.exe -m psm_model.label_audit `
  psm-model\data\curriculum\psm-50m-full-storage-v1-filtered.jsonl `
  --fail-on-high-risk

.\.venv\Scripts\python.exe -m psm_model.make_action_first_curriculum `
  psm-model\data\curriculum\psm-50m-action-first-v1-filtered.jsonl `
  psm-model\data\curriculum\psm-50m-full-storage-v1-filtered.jsonl

.\.venv\Scripts\python.exe -m psm_model.combine_jsonl `
  psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl `
  psm-model\data\direct-behavior-v1\test.jsonl `
  psm-model\data\hard-behavior-v1\test.jsonl `
  psm-model\data\nano-hf-storage-v1\test.jsonl
```

Pass: `ignore_fraction <= 0.40`, zero high-severity rows in filtered training set.

## Gate 1 — Classifier

```powershell
.\.venv\Scripts\python.exe -m psm_model.action_classifier train `
  psm-model\data\curriculum\psm-50m-action-first-v1-filtered.jsonl `
  --out psm-model\checkpoints\psm-action-classifier-v2-filtered.pt `
  --steps 1000 --device auto --sampling action_balanced `
  --eval-every 100 --abort-after-step 300 --collapse-threshold 0.8 `
  --probe psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl
```

Pass: expanded-probe macro >= 0.90, collapse_fraction <= 0.80. If fail, tighten filter (`--drop-severity medium`) or fix labels — do not start 50M.

## Gate 2 — Phase 1 scratch 50M

No `--resume`. Prefer **mixed** action curriculum (storage + direct-behavior) so manual probes are not out-of-distribution.

```powershell
.\.venv\Scripts\python.exe -m psm_model.make_action_first_curriculum `
  psm-model\data\curriculum\psm-50m-action-direct-v1.jsonl `
  psm-model\data\direct-behavior-v1\train.jsonl --copies 4

.\.venv\Scripts\python.exe -m psm_model.combine_jsonl `
  psm-model\data\curriculum\psm-50m-action-mixed-v1.jsonl `
  psm-model\data\curriculum\psm-50m-action-first-v1-filtered-ctx2048.jsonl `
  psm-model\data\curriculum\psm-50m-action-direct-v1.jsonl

.\.venv\Scripts\python.exe -m psm_model.filter_by_token_budget `
  psm-model\data\curriculum\psm-50m-action-mixed-v1.jsonl `
  psm-model\data\curriculum\psm-50m-action-mixed-v1-ctx2048.jsonl `
  --tokenizer psm-model\tokenizers\real-v1-pattern.json --max-tokens 2049 --output-format action
```

```powershell
.\.venv\Scripts\python.exe -m psm_model.train `
  psm-model\data\curriculum\psm-50m-action-mixed-v1-ctx2048.jsonl `
  --out psm-model\checkpoints\real-v3-50m-action-mixed-v1.pt `
  --steps 500 --batch-size 1 --preset 50m `
  --learning-rate 0.0003 --min-learning-rate 0.0001 --warmup-steps 50 `
  --device auto --cuda-memory-fraction 0.5 `
  --save-every 100 `
  --metrics-out psm-model\checkpoints\real-v3-50m-action-scratch-v1.metrics.jsonl `
  --output-format action --sampling action_balanced `
  --action-span-loss-weight 1 --structural-loss-weight 1 `
  --eval-every 100 --abort-after-step 300 --collapse-threshold 0.8 `
  --probe psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl
```

Eval after training (numbers **and** qualitative smoke on manual probes):

```powershell
.\.venv\Scripts\python.exe -m psm_model.gate_checkpoint `
  psm-model\checkpoints\real-v3-50m-action-mixed-v1.pt `
  --mode phase1-action --device cpu --output-format action

.\.venv\Scripts\python.exe -m psm_model.action_smoke `
  psm-model\checkpoints\real-v3-50m-action-mixed-v1.pt `
  psm-model\data\direct-behavior-v1\manual-probe.jsonl `
  --device cpu --output-format action --prefix-eval
```

Pass: expanded macro >= 0.85, manual model action >= 0.80, >= 4 distinct predicted actions, **and** manual `match_rate` >= 0.80 in `action_smoke`. Abort if one action > 80%, macro < 0.50 at step 500, or no improvement by step 300.

## Gate 3 — Phase 2 full StorageDecision

Only after Gate 2 passes. Resume from Gate-2 checkpoint only.

```powershell
.\.venv\Scripts\python.exe -m psm_model.train `
  psm-model\data\curriculum\psm-50m-full-storage-v1-filtered.jsonl `
  --out psm-model\checkpoints\real-v3-50m-full-v1.pt `
  --resume psm-model\checkpoints\real-v3-50m-action-scratch-v1.pt `
  --steps 3000 --batch-size 1 --preset 50m `
  --output-format tagged --sampling action_balanced `
  --device auto --cuda-memory-fraction 0.5 `
  --save-every 200
```

```powershell
.\.venv\Scripts\python.exe -m psm_model.eval_checkpoint `
  psm-model\checkpoints\real-v3-50m-full-v1.pt `
  psm-model\data\probes\direct_probes.jsonl --device auto
```

Pass: direct probes exact on all metrics (see `psm_model.gates`).

## Gate 4 — Product

Do **not** wire into `psm-core` until Gate 3 model-only generation passes. `safe_generate` remains an optional experiment bridge.

## Colab (after Gate 2)

1. Upload filtered curricula + passing checkpoints to a **private** HF repo via `hf upload`.
2. Clone repo in Colab GPU runtime; `snapshot_download` artifacts.
3. Use same train flags; see `psm-model/notebooks/psm-50m-product-safe-gate-colab.ipynb` for gate eval pattern (update checkpoint paths; do not use denylisted checkpoints as training bases).
4. Nano Colab notebooks under `nano-psm/notebooks/` remain useful for HF sync workflow only.

## HF data sources (local copies)

```text
hf-upload/nano-psm-retention-blend-codex-84k/
hf-upload/nano-psm-fast-mixed-10k/
hf-upload/nano-psm-codex-sessions-gpt41-mini-200/
```

Convert with `python -m psm_model.convert_nano_dataset` when refreshing from HF.

## Session log

Append every run to [session-log.md](session-log.md): date, command, checkpoint path, gate JSON, pass/fail.
