---
name: runpod-gpu-train
description: >-
  Launch and verify PSM GPU training on RunPod without idle billing. Use when
  deploying pods, starting Gate 4/5 train/eval, checking tmux/CUDA/GPU util,
  or debugging "pod running but GPU at 0%". Covers HF tokens, two-phase launch,
  verify-pod, and hard SSH timeouts.
---

# RunPod GPU training (PSM)

**Goal:** GPU billing only while `psm_model.train` (or eval) is actually on CUDA. Never leave a pod running after a silent launch failure.

Read also: `.cursor/rules/runpod-auto-delete.mdc`, `docs/psm-model/training-playbook.md`, **`docs/psm-model/training-pitfalls.md`** (mandatory before prod-memory curriculum/train).

## HF tokens (set before every launch)

| Env var | Source | Used for |
|---------|--------|----------|
| `HF_TOKEN` | `o chinnahftoken` → clipboard | **Model repo** `subbu83/psm-50m-mixed-v1-run` (checkpoints) |
| `DATASET_HF_TOKEN` | `~/.cache/huggingface/token` | **Dataset repo** probes/curriculum |

Pod env `{{ RUNPOD_SECRET_HF_TOKEN_C }}` is **not** sufficient alone for agent launches — always pass both tokens via `runpod_ctl.py` `extra_env` (train-gate5 does this when env is set locally).

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'
```

## Two-phase launch (required)

**Never** run `train-gate5 --deploy` as a single blocking foreground command and walk away. It uses an 8h SSH timeout and hides failures behind apt/git/HF download.

### Phase 1 — Provision pod (~2–5 min, hard stop on failure)

```powershell
python psm-model\scripts\runpod_ctl.py deploy --auto-gpu --name psm-gate5 --wait-ssh 300
python psm-model\scripts\runpod_ctl.py ssh-info <pod_id>   # get proxy-user
```

GPU fallback tries SECURE then COMMUNITY across `PSM_GPU_PREFERENCES` (A5000 → L4 → 3090 → …).

### Phase 2 — Start training (~3 min SSH, not 8h)

Warm path (existing pod): sync scripts only, start tmux, exit.

```powershell
python psm-model\scripts\runpod_ctl.py train-gate5 `
  --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix>@ssh.runpod.io `
  --profile recall-heavy `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-057000.pt `
  --tokenizer psm-model/checkpoints/real-v3-50m-full-v2-step-057000.tokenizer.json `
  --target-steps 58000 --batch-size 16 --learning-rate 5e-5 `
  --keep-pod
```

Cold bootstrap (first time on pod, downloads checkpoint): add `--no-warm-pod` (longer; still use Phase 3).

**Do not** `--sync-src` on warm pods (multi-minute hang). Cold only.

### Phase 3 — Verify within 90s (mandatory)

| Exit | `job_state` | Meaning |
|------|-------------|---------|
| 0 | `training` or `eval_finished` | Healthy or fully done |
| 1 | — | Launch/verify failure (no tmux, no CUDA, GPU too low during train) |
| 2 | `train_finished` or `idle_billing` | Train done or pod idle — **pull artifacts first**, then stop (not delete) unless checklist complete |

```powershell
python psm-model\scripts\runpod_ctl.py verify-pod --pod-id <id> --proxy-user <user>
# exit 2 → pull eval/logs/metrics locally + verify HF → then stop-pod (never delete without checklist)
python psm-model\scripts\runpod_ctl.py verify-pod ... --stop-on-fail   # auto-stop only, never auto-delete
```

Re-run anytime (≤60s). Use in `/loop` or cron to catch finished jobs.

Bootstrap window (apt/git, no GPU yet): `verify-pod --no-require-gpu` — only checks tmux + process. Re-run with `--require-gpu` within 10 min.

### Phase 4 — Monitor (non-blocking)

```powershell
# Repeat; exits in ≤60s
python psm-model\scripts\runpod_ctl.py verify-pod --pod-id <id> --proxy-user <user>
```

Dashboard: GPU util >0%, VRAM climbing. CPU-only 30+ min = failed launch.

## tmux sessions (Gate 5)

| Session | Purpose |
|---------|---------|
| `psm-gate5` | `psm_model.train` |
| `psm-gate5-sync` | HF upload every 120s |
| `psm-gate5-eval` | dual gate eval (post-train) |

Train tmux **must** export `PSM_RUNPOD=1` or training falls back to CPU silently.

## verify-pod checks

1. `tmux has-session -t psm-gate5`
2. `pgrep psm_model.train`
3. `torch.cuda.is_available()`
4. `nvidia-smi` util ≥ `--min-gpu-pct`
5. Tail `/tmp/psm-gate5-train.log`

Built into `runpod_ctl.py` `train-gate5` warm path via `_verify_pod_job` (~15s). Use standalone `verify-pod` anytime after.

## Failure playbook (don't burn GPU hours)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| GPU 0%, CPU busy 20+ min | apt/git/HF download, or hung SSH | `verify-pod`; if no tmux → stop pod, fix tokens, relaunch Phase 2 |
| `Repository not found` on HF | Wrong `HF_TOKEN` | `o chinnahftoken` |
| `FileNotFoundError` tokenizer | Warm start without checkpoint | `--no-warm-pod` once, or warm script HF download |
| `CUDA unknown error` | Bad GPU state on host | Stop/delete pod, redeploy (new machine) |
| Local terminal silent 30+ min | Blocking 8h `train-gate5 --deploy` | Kill local python; use two-phase launch |
| SSH probe timeout | Stacked SSH sessions | Stop pod; one launch at a time |

**Idle >5 min with GPU 0% after bootstrap should complete → `stop-pod` only.** Never `delete-pod` until eval reports, logs, and checkpoints are local + on HF (see `.cursor/rules/runpod-auto-delete.mdc`).

## Gate 5 quick reference

- Resume: `057000` → target `058000`, recall-heavy curriculum
- After train: dual eval @ target step; promote only if `passed: true`
- Keep pod until HF triple verified for resume step

## Prod-memory quick reference

Same two-phase launch as Gate 5. **Playbook:** `docs/psm-model/training-playbook.md` → “Prod-memory RunPod”.

```powershell
python psm-model\scripts\runpod_ctl.py train-prod-memory `
  --pod-id <pod_id> --proxy-user <user> `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-step-060000.pt `
  --target-steps 65000 --keep-pod

python psm-model\scripts\runpod_ctl.py verify-pod `
  --pod-id <pod_id> --proxy-user <user> `
  --tmux-session psm-prod-memory `
  --train-log /tmp/psm-prod-memory-train.log `
  --stop-on-fail
```

**Never `--no-warm-pod` after deploy tar-push.** **Never skip verify-pod** — GPU 0% + no tmux = stop pod immediately.

## Agent rules

1. Always `--proxy-user` on existing pods.
2. Always `verify-pod` after launch; use `--stop-on-fail` on first deploy of the day.
3. Never stack concurrent SSH train launches on one pod.
4. Background long jobs only **after** Phase 3 passes.
5. Confirm RunPod dashboard GPU util before ending session.
