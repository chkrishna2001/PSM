# PSM 50M — end of day handoff (2026-06-15)

**Read first next session:** this file → `.cursor/skills/runpod-gpu-train/SKILL.md` → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on RunPod.** All pods stopped/deleted. **No GPU billing.**

**Training for phase 2 is done.** **Dual eval @ 058000 was never completed.** Ship bar still **not met**.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Best storage checkpoint** | **048000** (Gate 4 promoted) |
| **Latest trained weights** | **058000** on HF (`subbu83/psm-50m-mixed-v1-run`) — phase 2 recall-heavy train finished |
| **Phase 2 train** | **Complete** — resumed 057000 → **058000** on A5000; user confirmed ~98% GPU during train |
| **Dual gate @ 051000** | Storage **PASS**, recall **FAIL** (~4.3% parse on 23 probes) |
| **Dual gate @ 058000** | **Not run** — eval blocked by RunPod container/SSH failures |
| **HF model repo** | `subbu83/psm-50m-mixed-v1-run` (`o chinnahftoken`) |
| **HF dataset repo** | `chkrishna2001/psm-50m-action-mixed-v1` |
| **RunPod pods** | **None** (deleted 2026-06-15) |
| **Ship bar** | **Not met** — need `eval_dual_gate` `passed: true` on storage **and** recall |

### Local artifacts

| Path | Notes |
|------|--------|
| `psm-model/checkpoints/gate-eval/gate5-dual-step-051000.json` | Phase 1 dual eval (storage pass, recall fail) |
| `psm-model/data/curriculum/psm-50m-gate5-train-v2-recall-heavy.jsonl` | Phase 2 mix (~38.5% recall) |
| `psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl` | 23 recall probes |
| `psm-model/checkpoints/gate5-*.log` | Session eval/deploy logs (no successful eval report) |

**Missing:** `psm-model/checkpoints/gate-eval/gate5-dual-step-058000.json`

---

## What we accomplished this session

### Phase 2 training (success)

1. Resumed **057000 → 058000** on pod `vi2zs6e89z5k4q` (`psm-gate5-3090-v2`, RTX A5000).
2. Recall-heavy curriculum; training completed with high GPU util.
3. Checkpoint **058000** uploaded to HF.
4. Pod stopped after train finished (no post-train eval on warm-start path).

### Dual eval (not completed)

Multiple eval attempts failed:

| Issue | Impact |
|-------|--------|
| Warm eval path has no post-train dual eval | Train finished without running eval |
| Wrong `verify-pod` defaults (`psm_model.train` not `eval_dual_gate`) | False “idle” detection; one pod stopped mid-eval at **64% GPU** |
| Too many pods created (`eval-v2`, `eval-v3`, …) | Wasted billing; user stopped session |
| SSH `container not found` while API shows `RUNNING` | Script push fails (`PSM_PUSH_OK` never received); eval never starts |
| COMMUNITY 3090 host had `cuda_available=False` | Eval ran on CPU, 0% GPU (earlier session) |

### Pods (all deleted)

| Pod | Name | Notes |
|-----|------|--------|
| `vi2zs6e89z5k4q` | psm-gate5-3090-v2 | Training pod — deleted after train |
| `4ub6fcfpb0apl6` | psm-gate5-eval-v2 | A5000 SECURE — eval reached 64% GPU then stopped by bad verify |
| `v65au7oae5e1q7` | psm-gate5-eval-v3 | RTX 2000 Ada — broken container; **deleted end of session** |

---

## Code / ops changes (local, uncommitted)

| Change | File(s) |
|--------|---------|
| Atomic checkpoint save (`.tmp` → rename) | `psm-model/src/psm_model/train.py` |
| Two-phase eval: bootstrap → start tmux → verify → wait | `runpod_ctl.py` `cmd_eval_gate5_dual` |
| Eval wait script (no full re-bootstrap) | `runpod_wait_gate5_dual.sh` |
| `verify-pod`: `gpu_active` state; don't false-fail on wrong proc pattern | `runpod_ctl.py` |
| Push: detect `container not found`; check stdout+stderr for `PSM_PUSH_OK` | `runpod_ctl.py` `_ssh_push_dir` |
| SSH probe rejects dead container | `runpod_ctl.py` `_ssh_probe` |
| CUDA preflight in eval script | `runpod_eval_gate5_dual.sh` |
| HF checkpoint download on warm train start | `runpod_start_gate5_train_only.sh` |
| Restart-and-eval helper | `runpod_ctl.py` / `run_gate5_eval_now.py` |
| RunPod GPU train skill + rule updates | `.cursor/skills/runpod-gpu-train/SKILL.md`, `runpod-auto-delete.mdc` |

---

## Next session — priority order

### P0 — Dual eval @ 058000 (only task until report exists)

**Do not re-train** unless eval fails and you have a clear plan. Weights at 058000 are on HF.

1. **One pod only.** Deploy **one** SECURE GPU (A5000 preferred; avoid COMMUNITY 3090 hosts with broken CUDA).
2. **Do not use `train-gate5 --deploy` as a blocking 8h command.** Use skill two-phase: `deploy` → job → `verify-pod`.
3. **Run eval:**

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'

# Phase 1: one pod
python psm-model\scripts\runpod_ctl.py deploy --auto-gpu --name psm-gate5-eval --wait-ssh 300
python psm-model\scripts\runpod_ctl.py ssh-info <pod_id>   # proxy-user

# Phase 2: eval (no --sync-src on warm pod)
python psm-model\scripts\runpod_ctl.py eval-gate5-dual `
  --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix>@ssh.runpod.io `
  --eval-step 58000 `
  --pull-reports psm-model\checkpoints\gate-eval `
  --keep-pod

# Phase 3: verify eval (correct session + process!)
python psm-model\scripts\runpod_ctl.py verify-pod `
  --pod-id <pod_id> `
  --proxy-user <user> `
  --tmux-session psm-gate5-eval `
  --process-pattern eval_dual_gate
```

4. **Before leaving:** confirm dashboard GPU util >0% during eval, or stop/delete pod.
5. **Pull** `gate5-dual-step-058000.json`; check `passed: true` for both gates.
6. **Stop/delete pod** after eval + HF verify.

### P0 — Container health check

If SSH returns `container not found` while pod is `RUNNING`:

```powershell
python psm-model\scripts\runpod_ctl.py stop-pod <id>
# wait 15s
# POST start via API or dashboard Resume
# poll until: nvidia-smi works over SSH
```

**Do not deploy a second pod** while one exists. Restart the same pod first.

### P1 — If recall fails @ 058000

- Read `gate5-dual-step-058000.json` recall section.
- Options: extend train 058000 → 062000 recall-heavy; expand recall curriculum; lower LR recall micro-run.
- See prior handoff [2026-06-12-end-of-day-handoff.md](2026-06-12-end-of-day-handoff.md) P1 section.

### P1 — Product

- Daemon/hook still need full `PsmModelRuntime` recall path at inference.
- `psmModel.enabled` still opt-in.

---

## Lessons (avoid repeat waste)

1. **Never create a new pod** while an existing pod can be restarted (same volume).
2. **`verify-pod` for eval** must use `--tmux-session psm-gate5-eval --process-pattern eval_dual_gate`.
3. **`PSM_PUSH_OK` failure** usually means dead container, not a bad tar push.
4. **Warm train script does not run dual eval** — always run `eval-gate5-dual` after train completes.
5. **Idle >5 min at 0% GPU after bootstrap** → stop pod immediately.

---

## Gate thresholds (reminder)

**Ship:** `eval_dual_gate` → `passed: true` on storage **and** recall.

Phase 1 @ 051000: storage pass, recall fail. Phase 2 goal: recall pass @ 058000 after recall-heavy training.

---

## Env setup (PowerShell)

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'
```
