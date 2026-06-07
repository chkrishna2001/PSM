# RunPod + SSH ops cheat sheet (PSM)

**Purpose:** Stop re-learning the same RunPod proxy SSH failures. Use `psm-model/scripts/runpod_ctl.py` for automation; use this doc when debugging or running commands by hand.

**Last verified:** 2026-06-07 (Gate 2+3 GPU eval PASS via proxy SSH; pod deleted after pull).

---

## Current model status (quick)

| Gate | Checkpoint | Status |
|------|------------|--------|
| 3 Full StorageDecision | `psm-model/checkpoints/real-v3-50m-full-v2.pt` | **PASS** (CPU + CUDA) |
| 2 Phase 1 action | `psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.pt` | **PASS** (CUDA) |
| 4 Product / psm-core | `remember --psm-model` | **PASS** (semantic upsert works) |

GPU reports: `psm-model/checkpoints/gate-eval/summary.json`

---

## Rules that bit us (read before SSH)

1. **RunPod proxy SSH ignores remote command arguments.**  
   `ssh user@ssh.runpod.io "echo hello"` opens an interactive shell; it does **not** run `echo hello`.  
   **Always** use piped `bash -s` (same pattern as `_ssh_probe` in `runpod_ctl.py`).

2. **Use `-tt`** on proxy SSH (PTY required).

3. **Do not pipe multiline scripts raw on Windows** — CRLF breaks bash; long one-line `printf '%s' '<base64>'` can stall on PTY.  
   `runpod_ctl.py` ships scripts via **chunked heredoc → base64 decode → bash**.

4. **Direct TCP often times out** from home network; **proxy SSH works**. Try proxy first.

5. **`scp` fails through proxy** (`subsystem request failed`). Pull reports via `eval-gates --pull-reports` (SSH tar fallback) or manual `cat` per file.

6. **GraphQL `podHostId` may 403** on your API key. Copy **proxy user** from RunPod Connect tab: `{pod_id}-{suffix}` (e.g. `znq97fgibrg758-64411407`). Cache: `psm-model/checkpoints/.runpod-ssh-cache.json`.

7. **Gate 3 eval must use `--output-format tagged`** when checkpoint `.meta.json` is empty (defaults to `json` → 0% parse).

8. **Local eval: `--device cpu` only** (local CUDA crashed laptop). Train/eval on RunPod GPU.

9. **PowerShell:** no bash `<<<`; use `runpod_ctl.py` or Python `subprocess` with `input=` for SSH stdin.

10. **Delete idle pods** when done (`delete-pod`) — eval pods on 4090 ~$0.69/hr.

---

## API key (Windows)

```powershell
cd C:\Users\chkri\source\repos\PSM
# Loads key into clipboard via 1Password CLI; or set once per session:
$env:RUNPOD_API_KEY = '<your-key>'
```

---

## `runpod_ctl.py` — preferred commands

All from repo root:

```powershell
cd C:\Users\chkri\source\repos\PSM
```

### List / lifecycle

```powershell
python psm-model\scripts\runpod_ctl.py list-pods

python psm-model\scripts\runpod_ctl.py deploy --name psm-eval --wait-ssh 300

python psm-model\scripts\runpod_ctl.py stop-pod <pod_id>
python psm-model\scripts\runpod_ctl.py delete-pod <pod_id>
python psm-model\scripts\runpod_ctl.py delete-all   # careful
```

### SSH discovery (do this after every new deploy)

```powershell
# 1) Get pod id from list-pods or deploy output
python psm-model\scripts\runpod_ctl.py ssh-info <pod_id> --proxy-user <pod_id>-<suffix>

# 2) Write ~/.ssh/config (host runpod-psm + runpod-psm-proxy)
python psm-model\scripts\runpod_ctl.py ssh-config <pod_id> --proxy-user <pod_id>-<suffix>

# 3) Wait until proxy answers
python psm-model\scripts\runpod_ctl.py wait-ssh <pod_id> --proxy-user <pod_id>-<suffix> --timeout-sec 420
```

**Where to get `--proxy-user`:** RunPod web UI → Pod → Connect → SSH over proxy → user is `something@ssh.runpod.io` → use the part before `@`.

### GPU gate eval (one shot)

```powershell
# Fresh pod, eval, pull reports, delete pod
python psm-model\scripts\runpod_ctl.py eval-gates `
  --deploy `
  --delete-after `
  --pull-reports psm-model\checkpoints\gate-eval `
  --proxy-user <pod_id>-<suffix> `
  --timeout-sec 7200

# Existing pod (already running)
python psm-model\scripts\runpod_ctl.py eval-gates `
  --pod-id <pod_id> `
  --proxy-user <pod_id>-<suffix> `
  --pull-reports psm-model\checkpoints\gate-eval `
  --timeout-sec 7200

# Include Gate 4 expanded probe (920 rows, slower)
python psm-model\scripts\runpod_ctl.py eval-gates --deploy --expanded --delete-after ...
```

---

## Manual SSH (when debugging `runpod_ctl`)

**Works — connectivity probe:**

```powershell
ssh.exe -tt -i $env:USERPROFILE\.ssh\id_ed25519 `
  <proxy-user>@ssh.runpod.io bash -s
```

Then type (or pipe via Python `subprocess` `input=`):

```bash
echo ssh-ready
exit
```

**Does not work on proxy:**

```powershell
# BAD — opens interactive shell, remote command ignored
ssh.exe -tt -i ... <proxy-user>@ssh.runpod.io "echo hello"
```

**Run a one-liner via bash -s (Python pattern):**

```python
import subprocess, os
SSH = ["ssh.exe", "-tt", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
       "-o", "StrictHostKeyChecking=accept-new", "<proxy-user>@ssh.runpod.io", "bash", "-s"]
subprocess.run(SSH, input="hostname\nexit\n", text=True, capture_output=True, timeout=60)
```

**Check eval artifacts on pod:**

```python
cmds = "find /workspace/PSM/psm-model/checkpoints/gate-eval -type f\nexit\n"
subprocess.run(SSH, input=cmds, text=True, capture_output=True, timeout=60)
```

---

## Local eval (CPU) — copy-paste

```powershell
cd C:\Users\chkri\source\repos\PSM
$env:PYTHONPATH = 'psm-model\src'

# Gate 3 — MUST include --output-format tagged
.\.venv\Scripts\python.exe -m psm_model.eval_checkpoint `
  psm-model\checkpoints\real-v3-50m-full-v2.pt `
  psm-model\data\probes\direct_probes.jsonl `
  --device cpu --output-format tagged

# Gate 2 phase-1 action
.\.venv\Scripts\python.exe -m psm_model.gate_checkpoint `
  psm-model\checkpoints\real-v3-50m-action-mixed-v2-step-009800.pt `
  --mode phase1-action --device cpu --output-format action
```

---

## Product model (psm-core) — copy-paste

```powershell
cd C:\Users\chkri\source\repos\PSM
npm run build

node src\psm-cli\dist\cli.js remember `
  --llm-response "I prefer SQLite for local prototypes because it is easy to inspect." `
  --psm-model --no-embeddings --json
```

Use **`cli.js`** (not `index.js`). Flag `--psm-model` routes storage to `real-v3-50m-full-v2.pt` via `remember_cli`.

---

## HF downloads

```powershell
# Checkpoints
hf download chkrishna2001/psm-50m-mixed-v1-run `
  psm-model/checkpoints/real-v3-50m-full-v2.pt `
  psm-model/checkpoints/real-v3-50m-full-v2.tokenizer.json `
  psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.pt `
  psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.tokenizer.json `
  --local-dir .

# Probes live in repo: psm-model/data/probes/direct_probes.jsonl
# Dataset repo paths vary; prefer local copy or pod eval script downloads.
```

---

## Files to know

| Path | Role |
|------|------|
| `psm-model/scripts/runpod_ctl.py` | RunPod REST + SSH automation |
| `psm-model/scripts/runpod_eval_gates.sh` | Bootstrap + Gate 2/3 eval on pod |
| `psm-model/checkpoints/.runpod-ssh-cache.json` | `{pod_id: proxy-user}` cache |
| `~/.ssh/config` | `Host runpod-psm` / `runpod-psm-proxy` (written by `ssh-config`) |
| `psm-model/checkpoints/gate-eval/` | Pulled GPU eval JSON reports |

---

## Known open issues

| Issue | Workaround |
|-------|------------|
| GraphQL 403 for `podHostId` | `--proxy-user` from Connect tab + cache file |
| Direct TCP SSH timeout | Use proxy only |
| `scp` via proxy fails | `eval-gates --pull-reports` or SSH `cat` per JSON file |
| NumPy missing in `.venv` | Warning only; `pip install numpy` when convenient |
| `remember` flaky if `source_timestamp` sent on non-temporal text | Fixed in `remember_cli.py` — don't regress |

---

## When automation fails — checklist

1. `list-pods` — pod `RUNNING`?
2. `ssh-info <pod_id> --proxy-user ...` — fresh suffix after redeploy?
3. `wait-ssh` — proxy probe prints `ssh-ready`?
4. Manual `bash -s` + `hostname` — same as above?
5. On pod: `ls /workspace/PSM/psm-model/checkpoints/gate-eval/` — reports present?
6. Pull: re-run `eval-gates --pull-reports` or cat files manually.
7. **Delete pod** if idle.
