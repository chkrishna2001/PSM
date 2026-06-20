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

## RunPod sizing (50M @ batch-size 1)

Telemetry on RTX 4090 pods showed ~1–3 GiB VRAM and ~2 GiB RAM during training — the 50M decoder at batch-size 1 does not need a 4090 or a high-RAM host.

| Setting | Default (`runpod_ctl.py`) | Notes |
|---------|---------------------------|-------|
| GPU | **RTX 3090** | 24 GiB VRAM is plenty (~3–6 GiB used). L4 / 3060 12GB may work; override with `--gpu` if unavailable. |
| Volume | **20 GiB** | OK with HF sync every 10 min + `keep-local=2` (~1.3 GiB step saves). Without sync, fills fast at `--save-every 200`. |
| Container disk | **10 GiB** | Stock PyTorch image + apt/pip only. |

Override only when needed:

```powershell
python psm-model\scripts\runpod_ctl.py train-gate4 --deploy `
  --gpu "NVIDIA GeForce RTX 4090" `
  --volume-gb 40 `
  --container-disk-gb 20
```

Delete pods after upload + eval — idle GPU billing continues until the pod is stopped or deleted.

### GPU availability (RunPod GraphQL)

RunPod exposes real-time stock via **GraphQL** (`gpuTypes` → `lowestPrice.stockStatus`: `High` / `Medium` / `Low` / `None`). The REST deploy API does not list availability — check before deploy:

```powershell
# PSM preference list (3090 → 4080 → L4 → 4090)
python psm-model\scripts\runpod_ctl.py list-gpus

# Pick first available from that list
python psm-model\scripts\runpod_ctl.py pick-gpu

# Deploy using first available GPU (same preference order)
python psm-model\scripts\runpod_ctl.py train-gate4 --deploy --auto-gpu --target-steps 36000
```

Requires GraphQL Read/Write on your API key (the RunPod warning about unrestricted access is expected). If `list-gpus` returns **403 error 1010**, that is Cloudflare blocking Python's default `User-Agent` — not missing key permissions. Current `runpod_ctl.py` sends a browser User-Agent to avoid this.

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
  --steps 12800 --batch-size 1 --preset 50m `
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

## Gate 4 — Expanded product bar

Gate 3 (`direct_probes`, 5 rows) proves the full StorageDecision head on canonical cases. Gate 4 is the **ship bar** on the budget-filtered expanded probe (~913 rows, prompts ≤1536 tokens).

**Do not** set `psmModel.enabled: true` by default until Gate 4 passes. `--psm-model` on `remember` remains opt-in for probe-shaped smoke.

### Eval (GPU — RunPod)

```powershell
python psm-model\scripts\runpod_ctl.py eval-gates --deploy --expanded --delete-after `
  --proxy-user <pod_id>-<suffix> --pull-reports psm-model\checkpoints\gate-eval
```

Gate 4 uses `--gate-mode expanded` (see `psm_model.gates.EXPANDED_PROBE_THRESHOLDS`).

### Pass criteria (`EXPANDED_PROBE_THRESHOLDS`)

| Metric | Minimum |
|--------|---------|
| `parse_valid_rate` | 0.95 |
| `schema_valid_rate` | 0.95 |
| `action_accuracy` | 0.85 |
| `memory_type_accuracy` | 0.70 |
| `memory_content_exact_rate` | 0.50 |
| `fact_count_accuracy` | 0.70 |
| `facts_exact_rate` | 0.50 |

Local repro (slow on CPU):

```powershell
$env:PYTHONPATH='psm-model\src'
.\.venv\Scripts\python.exe -m psm_model.eval_checkpoint `
  psm-model\checkpoints\real-v3-50m-full-v2.pt `
  psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl `
  --device cuda --output-format tagged --gate-mode expanded
```

### Failure analysis

After eval, bucket parse vs action vs content failures:

```powershell
.\.venv\Scripts\python.exe -m psm_model.analyze_eval_report `
  psm-model\checkpoints\gate-eval\gate4-full-expanded.json --gate-mode expanded
```

RunPod `runpod_eval_gates.sh` writes `gate4-failure-analysis.json` automatically when expanded eval runs.

### Training (production path — `gate4-train-v1`)

Curriculum: `build_gate4_train_v1` — **no 25k base dilution**:

| Slice | Source | Default weight |
|-------|--------|----------------|
| Expanded full DSL | `expanded-probe-v1-filtered.jsonl` ×40 | ~60% |
| Parse drills | `generate_direct_behavior_curriculum` promote/store ×25 | ~25% |
| Stratified real | promote/store sample from `full-storage-v1-filtered` (max 2500) | ~15% |
| Direct anchors | `direct_probes.jsonl` ×500 | regression guard |

Resume: `real-v3-50m-full-v2-step-022800.pt` (Gate 3 pass). Train with `--output-format tagged`, heavier `promote_semantic` / `store_episodic` span weights.

```powershell
python psm-model\scripts\runpod_ctl.py train-gate4 --deploy `
  --target-steps 36000 `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-022800.pt `
  --proxy-user <pod_id>-<suffix> `
  --timeout-sec 28800
```

Local curriculum build:

```powershell
$env:PYTHONPATH='psm-model\src'
.\.venv\Scripts\python.exe -m psm_model.build_gate4_train_v1 `
  psm-model\data\curriculum\psm-50m-gate4-train-v1.jsonl `
  --direct-probes psm-model\data\probes\direct_probes.jsonl `
  --expanded-probes psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl `
  --stratified-source psm-model\data\curriculum\psm-50m-full-storage-v1-filtered.jsonl
```

Milestones during training: `parse_valid_rate` ≥ 0.90 on expanded full eval before chasing content metrics. `--target-steps` is absolute (36000 = +13200 from 22800).

After training: `upload-gate4`, then `eval-gates --expanded`. Legacy dilution curriculum: `--curriculum-builder legacy`.

### Product smoke (additional, not gated in CI yet)

- Manual full-output smoke: `match_rate` ≥ 0.80 on `manual-probe.jsonl`
- 20–30 real-chat `remember --psm-model` E2E cases with parse-failure → ignore/repair

## Gate 5 — Recall / context planning (anti-collapse)

Gate 4 proves **write** path (StorageDecision). Gate 5 adds **read** path (`recall_plan`, `context_plan` JSON) without letting storage metrics collapse.

**Indexables are deferred** — recall training uses `target_tables`, `ranking_hints`, and `temporal_intent` only. Add indexables as a later Gate 6+ curriculum once recall planning passes.

### Build curriculum + probes

```powershell
$env:PYTHONPATH = 'psm-model\src'
.\.venv\Scripts\python.exe -m psm_model.generate_recall_curriculum `
  psm-model\data\curriculum\psm-50m-recall-plan-v1.jsonl

.\.venv\Scripts\python.exe -m psm_model.build_gate5_train_v1 `
  psm-model\data\curriculum\psm-50m-gate5-train-v1.jsonl `
  --expanded-copies 25 --direct-copies 100 --recall-copies 20
```

Default mix keeps **~65–75% storage mass** (expanded + direct anchors) and **~25–35% recall** — do not train recall-only on top of 48000.

### Train (resume from Gate 4 best)

```powershell
.\.venv\Scripts\python.exe -m psm_model.train `
  psm-model\data\curriculum\psm-50m-gate5-train-v1.jsonl `
  --resume psm-model\checkpoints\real-v3-50m-full-v2-step-048000.pt `
  --out psm-model\checkpoints\real-v3-50m-full-v2.pt `
  --steps 51000 --batch-size 8 --preset 50m `
  --output-format tagged --sampling random `
  --device auto --save-every 200 `
  --probe psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl `
  --eval-every 400
```

Use `--sampling random` for mixed storage+recall curricula. Action-span loss applies to storage rows only.

### Dual gate eval (must pass both)

```powershell
.\.venv\Scripts\python.exe -m psm_model.eval_dual_gate `
  psm-model\checkpoints\real-v3-50m-full-v2-step-051000.pt `
  --storage-probe psm-model\data\direct-behavior-v1\expanded-probe-v1-filtered.jsonl `
  --recall-probe psm-model\data\curriculum\psm-50m-recall-plan-v1.jsonl `
  --device cuda
```

| Gate | Metric | Minimum |
|------|--------|---------|
| **Storage (Gate 4 bar)** | `parse_valid_rate` | 0.95 |
| | `action_accuracy` | 0.85 |
| **Recall (Gate 5)** | `parse_valid_rate` | 0.95 |
| | `target_tables_exact_rate` | 0.90 |
| | `target_tables_primary_rate` | 0.95 |
| | `ranking_hints_score` | 0.50 |

Promote only when **dual gate passes**. If storage regresses, increase storage copies or reduce LR before adding recall mass.

### RunPod (`runpod_ctl.py train-gate5`)

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()

# New pod (cold bootstrap + wait for train + dual eval)
python psm-model\scripts\runpod_ctl.py train-gate5 --deploy --gpu "NVIDIA L4" `
  --target-steps 51000 --batch-size 8 --learning-rate 1e-4 --eval-every 400 `
  --pull-reports psm-model\checkpoints\gate-eval

# Existing warm pod (scripts+src sync, tmux only — verify GPU util)
python psm-model\scripts\runpod_ctl.py train-gate5 --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix> --warm-pod `
  --target-steps 51000 --batch-size 8

# Dual eval only (checkpoint already on HF)
python psm-model\scripts\runpod_ctl.py eval-gate5-dual --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix> --eval-step 51000 --sync-src `
  --pull-reports psm-model\checkpoints\gate-eval --delete-after
```

Tmux on pod: `psm-gate5` (train), `psm-gate5-sync` (HF upload), `psm-gate5-eval` (dual eval). Metrics: `real-v3-50m-full-v2-gate5.metrics.jsonl`. Report: `gate-eval/gate5-dual-step-051000.json`.

Before first launch, tar-push local `psm-model/src` + `psm-model/scripts` (ctl does this on warm/cold paths). Optionally upload new modules to HF dataset `psm-code/` for cold clones without `--sync-src`.

## Prod-memory RunPod (teacher v3 curriculum)

Resume prod-memory stem from prior smoke (e.g. **060000 → 065000**). Curriculum on dataset repo `chkrishna2001/psm-50m-action-mixed-v1` (`prod-memory/prod-extraction-v2.jsonl` — v3 content mirrored there; explicit v3 path needs script download support).

**Do not walk away until Phase 3 verify passes.** Idle pod + GPU 0% = billing with no training.

### Launch (two-phase — same as Gate 5)

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'

# Phase 1 — deploy (~2 min)
python psm-model\scripts\runpod_ctl.py deploy --auto-gpu --name psm-train-prod-memory --wait-ssh 300
python psm-model\scripts\runpod_ctl.py ssh-info <pod_id>

# Phase 2 — warm start ONLY (--pod-id; never --no-warm-pod after tar-push)
python psm-model\scripts\runpod_ctl.py train-prod-memory `
  --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix>@ssh.runpod.io `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-step-060000.pt `
  --tokenizer psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-step-060000.tokenizer.json `
  --curriculum psm-model/prod-memory/data/prod-extraction-v2.jsonl `
  --target-steps 65000 `
  --keep-pod

# Phase 3 — verify within 90s (mandatory; stop pod if fail)
python psm-model\scripts\runpod_ctl.py verify-pod `
  --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix>@ssh.runpod.io `
  --tmux-session psm-prod-memory `
  --train-log /tmp/psm-prod-memory-train.log `
  --stop-on-fail
```

Tmux: `psm-prod-memory` (train), `psm-prod-memory-sync` (HF upload every 120s). Train log: `/tmp/psm-prod-memory-train.log`.

`train-prod-memory` tar-pushes **`psm-model/src`** (full tree), **`psm-model/scripts`**, **`psm-model/prod-memory`** before starting tmux. Do not use `--no-warm-pod` on a pod that already received tar-push — partial `/workspace/PSM` breaks `git clone` in cold bootstrap.

### Known failures (2026-06-20 — do not repeat)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Repository not found` on checkpoint download | Pod `RUNPOD_SECRET_HF_TOKEN_C` is wrong; `hf download` without `--token` uses it | `runpod_start_prod_memory_train_only.sh` passes `--token "${HF_TOKEN:-}"`; always export `HF_TOKEN` from `o chinnahftoken` in launch env |
| `ModuleNotFoundError: psm_model.configs` | Only `train.py` was synced, not full `psm_model` package | `train-prod-memory` must push all of `psm-model/src` (fixed in ctl) |
| Pod up 5+ min, GPU 0%, no tmux | Launch failed silently after bootstrap | Run `verify-pod` with `--tmux-session psm-prod-memory` within 90s; `--stop-on-fail` to kill idle billing |
| Cold bootstrap + tar-push | `/workspace/PSM` exists but is not a git repo → `git pull` fails, incomplete tree | After `deploy`, use **warm** `train-prod-memory --pod-id` only |
| `verify-pod` says tmux missing on prod run | Default verify checks `psm-gate5` | Pass `--tmux-session psm-prod-memory --train-log /tmp/psm-prod-memory-train.log` |

### Pre-flight (agent checklist)

1. `HF_TOKEN` = `o chinnahftoken` (model repo checkpoints on `subbu83/...`)
2. `DATASET_HF_TOKEN` = local HF cache (curriculum on `chkrishna2001/...`)
3. Confirm resume `.pt` exists on HF (e.g. `...-prod-memory-step-060000.pt`)
4. Deploy → warm `train-prod-memory` → **verify within 90s**
5. GPU util >0% or stop pod — never leave billing after a failed launch

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
