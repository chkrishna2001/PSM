# Prod memory (isolated)

**This tree is separate from legacy `psm_model` gate/train code and from `benchmark/locomo`.**

Use it for the production-memory initiative only:

- Phase 1: grounding eval on prod-shaped `remember_target` fixtures
- Phase 5: `prod-extraction-v1` curriculum builder + validation
- Results land in `results/` here — not mixed with gate eval artifacts

## Do not

- Add gate-4/5/6 scripts here
- Import from `build_gate6_train_v1` or other legacy curriculum builders
- Put LoCoMo QA gold labels in fixtures

## Run grounding eval

From repo root:

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -m prod_memory.eval_grounding `
  --checkpoint psm-model\checkpoints\step-058000.pt `
  --checkpoint-label 058000 `
  --device cpu
```

Output: `psm-model/prod-memory/results/prod-grounding-baseline.json`

## Build curriculum mix

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -m prod_memory.build_prod_extraction_v1 `
  --direct-probes psm-model\data\probes\direct_probes.jsonl
```

Output: `psm-model/prod-memory/data/prod-extraction-v1.jsonl` + `.manifest.json`

### Upload curriculum to HF

```powershell
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
python -m prod_memory.upload_hf
```

Remote: `chkrishna2001/psm-50m-action-mixed-v1` → `prod-memory/prod-extraction-v1.jsonl`

### RunPod train (v3 curriculum, resume prod-memory stem)

Full launch checklist and **known failure modes** are in [`docs/psm-model/training-playbook.md`](../../docs/psm-model/training-playbook.md) (section “Prod-memory RunPod”). Summary:

1. Set `HF_TOKEN` (`o chinnahftoken`) + `DATASET_HF_TOKEN` (local cache)
2. `deploy` → warm `train-prod-memory --pod-id` (not `--no-warm-pod`)
3. `verify-pod --tmux-session psm-prod-memory --train-log /tmp/psm-prod-memory-train.log --stop-on-fail`

```powershell
python psm-model\scripts\runpod_ctl.py train-prod-memory `
  --pod-id <pod_id> --proxy-user <pod_id>-<suffix>@ssh.runpod.io `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-step-060000.pt `
  --curriculum psm-model/prod-memory/data/prod-extraction-v2.jsonl `
  --target-steps 65000 --keep-pod
```

### Colab smoke train

Open [`notebooks/prod-extraction-v1-colab.ipynb`](notebooks/prod-extraction-v1-colab.ipynb) in Colab (GPU runtime).

Secrets: `HF_TOKEN` (model repo), `DATASET_HF_TOKEN` (dataset repo).

## Run tests

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -m unittest discover -s psm-model\prod-memory\tests -v
```

## Layout

```
prod-memory/
  README.md
  prod_memory/          # package (prod_memory, not psm_model)
    grounding.py        # bleed + overlap metrics (mirrors psm-core guards)
    eval_grounding.py   # checkpoint eval CLI
    curriculum_sources.py
    indexable_labels.py
    row_validation.py
    build_prod_extraction_v1.py
  data/                 # built curriculum JSONL + manifest
  notebooks/            # Colab smoke train
  fixtures/cases.json   # held-out prod suites
  results/              # eval JSON output
  tests/                # unit tests (fast, no GPU)
```

Plan docs: [docs/plans/psm-production-memory/phase-1-baseline-eval.md](../../docs/plans/psm-production-memory/phase-1-baseline-eval.md)

Legacy model evals stay in `psm_model/eval_*.py` and `psm-model/scripts/`.
