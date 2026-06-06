# PSM 50M — end of day handoff (2026-06-04)

**Read first tomorrow:** this file + [session-log.md](session-log.md) + [training-playbook.md](training-playbook.md).

Training is **stopped** for the day. No `psm_model` Python processes should be running.

---

## Tomorrow: resume command (single job only)

```powershell
cd C:\Users\chkri\source\repos\PSM
$env:PYTHONPATH='psm-model\src'

.\.venv\Scripts\python.exe -m psm_model.train `
  psm-model\data\curriculum\psm-50m-action-mixed-v1-ctx2048.jsonl `
  --out psm-model\checkpoints\real-v3-50m-action-mixed-v1.pt `
  --resume auto --steps 8000 `
  --batch-size 1 --preset 50m `
  --learning-rate 0.0003 --min-learning-rate 0.0001 --warmup-steps 50 `
  --device auto --cuda-memory-fraction 0.5 `
  --save-every 200 `
  --metrics-out psm-model\checkpoints\real-v3-50m-action-mixed-v1.metrics.jsonl `
  --output-format action --sampling action_balanced `
  --action-span-loss-weight 1 --structural-loss-weight 1 `
  --action-span-weight promote_semantic=2 `
  --action-span-weight ignore=3 `
  --action-span-weight update_existing=2 `
  --eval-every 200 --abort-after-step 500 --collapse-threshold 0.85 `
  --probe psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl
```

**Before starting:** confirm Task Manager shows **one** `psm_model.train` command line (mixed curriculum only). Kill any stray `action-first-v1-filtered-ctx2048` train.

**Checkpoint state:** `real-v3-50m-action-mixed-v1-step-000200.pt` (step **200** / target 8000). Metrics: `real-v3-50m-action-mixed-v1.metrics.jsonl`.

**Qualitative eval (required; do not trust probe macro alone):**

```powershell
.\.venv\Scripts\python.exe -m psm_model.action_smoke `
  psm-model\checkpoints\real-v3-50m-action-mixed-v1-step-000200.pt `
  psm-model\data\direct-behavior-v1\manual-probe.jsonl `
  --device auto --output-format action --prefix-eval
```

Gate (numbers only):

```powershell
.\.venv\Scripts\python.exe -m psm_model.gate_checkpoint `
  psm-model\checkpoints\real-v3-50m-action-mixed-v1-step-000200.pt `
  --mode phase1-action --device auto --output-format action
```

Pass bar: expanded macro ≥ **0.85**, manual ≥ **0.80**, ≥ 4 distinct actions, **and** manual `action_smoke` `match_rate` ≥ **0.80**.

---

## External data: LoCoMo, REALTALK, and prior conversion

You already have substantial **nano-psm** training data from public / adapted sources. It was **converted** into `psm-model` generative JSONL via `python -m psm_model.convert_nano_dataset` (see `psm-model/README.md`).

| Source | Where it enters the pipeline | Converted `psm-model` artifacts |
|--------|------------------------------|--------------------------------|
| **LoCoMo** | `nano-psm/data-pipeline/src/generate-dataset.mjs` (`--locomo benchmark/locomo/data/locomo10.json`); also in retention blends | Mixed into nano JSONL → `convert_nano_dataset` → e.g. `psm-model/data/fast-mixed-10k/`, `psm-model/data/nano-hf-storage-v1/` |
| **REALTALK** | `generate-fast-mixed-dataset.mjs` (`realtalk-limit` ~900 from `nano-psm/data-pipeline/data/raw/realtalk-mteb/realtalk-training.jsonl`) | Same path as fast-mixed → converted storage rows |
| **PersonaMem, LongMemEval, user-preference-564k** | Also in `fast-mixed-10k` generator | Same |
| **Codex / Nemotron sessions** | `retention-blend-codex-84k`, `codex-sessions-2026`, HF uploads | `psm-model/data/real-v1/`, lines in `psm-50m-full-storage-v1.jsonl` with `source_kind` like `codex_session_nemotron` |

**Letta:** In this repo, “Letta” appears as a **benchmark competitor** (Mem0/Zep/Letta), not as a labeled training corpus. There is no `letta` adapter in `nano-psm/data-pipeline/sources/adapter-plan.md`. If you have separate Letta-export data, treat it like other raw sources: normalize → `convert_nano_dataset` → filter → combine.

### Already-converted datasets on disk (use tomorrow)

| Path | Rows (approx) | Notes |
|------|---------------|--------|
| `psm-model/data/nano-hf-storage-v1/` | 12,708 accepted | From HF: retention-blend + fast-mixed + codex sessions (`conversion-report.json`) |
| `psm-model/data/fast-mixed-10k/` | 8,022 accepted | LoCoMo/REALTALK/PersonaMem/LongMemEval mix (`conversion-report.json`) |
| `psm-model/data/real-v1/` | 8,082 accepted | `convert_nano_dataset` from retention-blend + fast-mixed-reviewed + codex-sessions |
| `psm-model/data/curriculum/psm-50m-full-storage-v1.jsonl` | 25,767 | **Current Phase 1 base** (filtered → action-first → ctx2048) |
| `psm-model/data/curriculum/psm-50m-action-mixed-v1-ctx2048.jsonl` | 33,197 | storage ctx2048 + direct-behavior ×4 |

**Tomorrow data idea (not done today):** Re-build mixed curriculum to include **converted LoCoMo/REALTALK rows** explicitly (e.g. `combine_jsonl` of `nano-hf-storage-v1/train.jsonl` or `fast-mixed-10k` action-first copies + direct-behavior), not only synthetic `direct-behavior-v1` + full-storage. That aligns training distribution with benchmark-style conversation.

Adapter plan (sources): `nano-psm/data-pipeline/sources/adapter-plan.md`.

---

## What we learned today (nano-psm lesson)

| Checkpoint | Expanded probe macro | Manual smoke `match_rate` |
|------------|----------------------|---------------------------|
| scratch step 3500 | 0.68 | 0.30 |
| scratch step 5000 | 0.56 | **0.10** |
| scratch step 3000 (best probe) | 0.72 | 0.30 |

**Conclusion:** High expanded-probe scores can hide complete failure on 10 hand-authored manual probes. Always run `action_smoke` on `manual-probe.jsonl`.

---

## Runs today

### Completed / paused

1. **Gate 0–1:** filtered curriculum + classifier PASS (`psm-action-classifier-v2-filtered.pt`).
2. **scratch-v1** `real-v3-50m-action-scratch-v1`: 500 → ~5600 steps on storage-only; best probe step **3000** (macro 0.72); manual still ~0.30; step 5000 manual **0.10**.
3. **mixed-v1** `real-v3-50m-action-mixed-v1`: started fresh; **stopped at step 200** (storage + direct-behavior curriculum).

### Stopped intentionally

- Storage-only resume toward 20k (killed; parallel GPU conflict).
- `gate_checkpoint` on step 5000 (killed mid-run).
- Duplicate mixed training processes (cleaned up).

### Do not resume (denylist)

See `psm-model/checkpoints/DENYLIST.txt` and playbook — v2/repair/action-head checkpoints.

**Scratch reference only:** `real-v3-50m-action-scratch-v1-step-005200.pt` / metrics through ~5.2k steps — diagnostic, not the main path going forward.

---

## Code added today

- `psm-model/src/psm_model/action_smoke.py` — qualitative inference + prefix scores.
- Mixed curriculum artifacts under `psm-model/data/curriculum/psm-50m-action-*-mixed*`.
- `train.py` probe eval, collapse abort; `gate_checkpoint --mode phase1-action`.

---

## Phase status

| Gate | Status |
|------|--------|
| 0 Data filter | PASS |
| 1 Classifier | PASS |
| 2 Phase 1 50M | **FAIL** — use mixed-v1; fold in LoCoMo/REALTALK converted data next |
| 3 Full StorageDecision | Blocked |
| 4 Product / psm-core | Deferred |

---

## Pitfalls for tomorrow

1. **Only one training process** — Windows shows 2 PIDs per job (venv + pythoncore); two *different* command lines = bug.
2. **Eval:** `action_smoke` every 500–1000 steps, not only `gate_checkpoint`.
3. **numpy warning** in venv is harmless for training; optional `pip install numpy` if needed elsewhere.
