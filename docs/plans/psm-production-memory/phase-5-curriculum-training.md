# Phase 5 — Curriculum and training

**Status:** v3/v4 RunPod iteration complete (2026-06-20) — **v5 suite micro-run ready for review**  
**Goal:** Train extraction from `remember_target` (= `llmResponse`), not gate preservation.  
**Depends on:** [Phase 2](phase-2-chunking-pipeline.md), [Phase 3](phase-3-indexables-workflows.md), [Phase 4](phase-4-guardrails-prod.md)

**Infrastructure:** HF dataset + Colab notebook **or** prod-memory RunPod (`runpod_ctl.py train-prod-memory`). Eval bar is always Phase 1 harness.

**2026-06-21 iteration:** [phase-5-failure-mining-2026-06-21.md](phase-5-failure-mining-2026-06-21.md) — v4 ×40 fail-copy rejected; v5 builder follows v1 mix with suite-focused micro-runs. **Mandatory pre-flight:** [training-pitfalls.md](../../psm-model/training-pitfalls.md).

---

## Mix profile: `prod-extraction-v1`

Builder: [`psm-model/prod-memory/prod_memory/build_prod_extraction_v1.py`](../../../psm-model/prod-memory/prod_memory/build_prod_extraction_v1.py)

| Source | Copies | Role |
|--------|--------|------|
| expanded-probe (or direct fallback) | **×2** | Regression only |
| recall plan | **×50** | Regression |
| assistant plans/handoffs | **×15** | Primary |
| workflows + indexables | **×10** | Primary |
| nano/chatgpt prod-normalized | **×10** | Diversity (optional paths) |
| short technical rules | **×5** | Same skill, different vocab |
| ignore/noise | **×8** | Suppress over-store |

### Build (local)

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -m prod_memory.build_prod_extraction_v1 `
  --direct-probes psm-model\data\probes\direct_probes.jsonl
```

### Upload to HF dataset repo

Uses `DATASET_HF_TOKEN` (local `~/.cache/huggingface/token`):

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
python -m prod_memory.upload_hf
```

**On HF:** [`chkrishna2001/psm-50m-action-mixed-v1`](https://huggingface.co/datasets/chkrishna2001/psm-50m-action-mixed-v1)

- `prod-memory/prod-extraction-v1.jsonl` (1353 rows)
- `prod-memory/prod-extraction-v1.manifest.json`

---

## Colab smoke train

Notebook: [`psm-model/prod-memory/notebooks/prod-extraction-v1-colab.ipynb`](../../../psm-model/prod-memory/notebooks/prod-extraction-v1-colab.ipynb)

| Parameter | Value |
|-----------|-------|
| Resume checkpoint | **`058000`** from [`subbu83/psm-50m-mixed-v1-run`](https://huggingface.co/subbu83/psm-50m-mixed-v1-run) |
| Do not resume | `062000` (more gate-heavy direction) |
| Target steps | **078000** (+2000 smoke) |
| LR | **2e-5** (min 5e-6) |
| Batch | 8 (use 4 if OOM) |
| Save / regression eval | every **500** steps |
| Prod grounding eval | every saved step ≥ 58500 via `prod_memory.eval_grounding` |

### Colab secrets

| Secret | Repo |
|--------|------|
| `HF_TOKEN` | Model repo `subbu83/psm-50m-mixed-v1-run` (read + write for upload) |
| `DATASET_HF_TOKEN` | Dataset repo `chkrishna2001/psm-50m-action-mixed-v1` (read) |

Asset download CLI (same as notebook cell 3):

```bash
PYTHONPATH=psm-model/src:psm-model/prod-memory python -m prod_memory.colab_sync download --root .
```

---

## Tasks

- [x] Create `build_prod_extraction_v1.py` in isolated `prod-memory/` tree.
- [x] Add grounding validator at dataset build time.
- [x] Port indexable label generation.
- [x] Synthetic plans/workflows/technical/noise seed rows.
- [x] Upload dataset to HF; document revision.
- [x] Colab notebook + `colab_sync` / `upload_hf` helpers.
- [ ] Run Colab smoke train from 058000.
- [ ] Compare prod grounding eval JSON to Phase 1 baseline.
- [ ] Ingest Cursor skills as additional workflow source bucket.

---

## Files

| Path | Role |
|------|------|
| `prod-memory/prod_memory/build_prod_extraction_v1.py` | Mix builder |
| `prod-memory/prod_memory/upload_hf.py` | Dataset upload |
| `prod-memory/prod_memory/colab_sync.py` | HF download/upload for Colab |
| `prod-memory/notebooks/prod-extraction-v1-colab.ipynb` | Colab smoke train |
| `prod-memory/prod_memory/eval_grounding.py` | Post-train prod eval |

---

## Exit criteria

- [ ] Grounding on plan + workflow suites improves **≥ 15pp** vs Phase 1 baseline.
- [ ] Bleed ≤ 2%.
- [ ] Regression gate still passes (parse ≥ 95%, action ≥ 85% on ×2 expanded).

---

## v5 suite micro (plan-aligned, 2026-06-21)

Builder: [`build_prod_extraction_v5_suite_micro.py`](../../../psm-model/prod-memory/prod_memory/build_prod_extraction_v5_suite_micro.py)

| Parameter | v5a (plan_chunks) |
|-----------|-------------------|
| Resume | **058000** gate stem only |
| Focus | plan fixtures ×5 + plan handoff seeds ×15 |
| Anchors | recall ×50, expanded ×2, cursor-01 ×5, noise ×8 |
| Target steps | ~59200 (+1200) |
| Abort if | cursor_shaped effective drops vs 058000 |

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -m prod_memory.build_prod_extraction_v5_suite_micro --focus-suite plan_chunks
```

Output: `psm-model/prod-memory/data/prod-extraction-v5.jsonl` — upload to HF before train.

---

## Results

| Artifact | Notes |
|----------|-------|
| HF `prod-memory/prod-extraction-v1.jsonl` | 1353 rows uploaded |
| `prod-memory/data/prod-extraction-v1.manifest.json` | Local + HF copy |
| v3 teacher train 060→065 | Lateral (1/10) — do not promote |
| v4 fixture-repair 058→060 | Regression (0/10) — rejected |
| v5 plan_chunks mix | Built locally — pending HF upload + train sign-off |
| Colab / RunPod smoke | Pending v5a |
