# HF LoCoMo + Colab CLI — end of day handoff (2026-06-26)

**Read first next session:** this file → `psm-model/scripts/colab_preflight.sh` → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on Colab or RunPod.** All Colab sessions **terminated**. RunPod pod `cyveakf0qhvqih` **EXITED** (credits).

**Ship bar (gate):** **10/10** `classify_match` on `psm-model/prod-memory/fixtures/cases.json` — **met** (prompt + parse fixes).

**Tomorrow goal:** finish **full LoCoMo** ingest (resume **2960/5882**) + retrieval eval on **Colab T4 via WSL**, without repeating today's setup errors.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Gate adapter** | `hf-prod-v5k-gate-distill-qwen0.5b` — **10/10** fixtures |
| **Extract adapter** | `hf-prod-v5k-extract-qwen0.5b` |
| **LoCoMo ingest (RunPod)** | **2960 / 5882** turns (50.3%), **0 failed** at last good sync |
| **LoCoMo ingest (Colab)** | Multiple attempts; **no new turns** — infra/script issues (see below) |
| **Retrieval eval** | **Not run** (ingest incomplete) |
| **HF model repo** | `krishnach7262/psm-prod-memory-hf` |
| **HF token** | `o krishnachhftoken` → clipboard (`HF_TOKEN`) |

### LoCoMo checkpoint (resume from here)

| Artifact | Path |
|----------|------|
| **Checkpoint DB** | `benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-n2960-checkpoint.db` |
| **Live sync copy** | `benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db` |
| **Decisions** | **2961** rows (SQLite verified) |
| **Resume offset** | **`2960`** (`--offset 2960` in `ingest-psm-model.ts`) |
| **Remaining** | **~2922** turns + retrieval eval |

---

## Colab CLI (WSL) — setup done

| Step | Status |
|------|--------|
| Ubuntu WSL | `Ubuntu-24.04`, user **`chkri`** |
| Colab CLI | `pipx install google-colab-cli` → `~/.local/bin/colab` |
| Google OAuth | **Done** (`colab sessions` works) |
| HF_TOKEN → WSL | **`WSLENV=HF_TOKEN/u`** from PowerShell (required) |

**Do not use** default `docker-desktop` WSL — no bash/python.

---

## Errors hit today (and fixes)

| # | Error | Cause | Fix (in repo) |
|---|--------|--------|----------------|
| 1 | `set: pipefail: invalid option` | Windows **CRLF** in `*.sh` | LF scripts; run `wsl sed -i 's/\r$//'` if editor reintroduces CRLF |
| 2 | `HF_TOKEN missing` in WSL | PowerShell env not passed | `$env:WSLENV = "HF_TOKEN/u"` before `wsl` |
| 3 | `colab: command not found` | Wrong WSL user / PATH | Use **`-u chkri`** and `~/.local/bin` |
| 4 | `colab exec -c` / `console -c` | **Invalid API** — exec is Python-only | `colab_locomo_hf_driver.py` → `bash colab_locomo_hf.sh` |
| 5 | `spawn .venv/bin/python ENOENT` | Colab has no venv | `--python python3` in `colab_locomo_hf.sh` |
| 6 | Stale GitHub clone | `git clone` missing local HF/two-pass code | **`colab_pack_repo.sh`** uploads local tarball (~18MB) |
| 7 | **6.4GB tarball** | Included `psm-model/prod-memory/checkpoints` | Pack excludes checkpoints/results; adapters fetched on VM |
| 8 | `tar: *.py: Cannot stat` | Glob in tar args | Removed; use directory paths only |
| 9 | Transient upload DNS fail | Colab proxy flake | `colab_upload()` 5× retry in automate |
| 10 | Ingest exit 1 ~5 min | Likely **stale dist** / missing `hf` CLI / no prebuild | Pack includes **`dist/` + `src/psm-core/dist/`**; `huggingface_hub[cli]` + `hf_download()` fallback |

---

## Scripts (tomorrow)

| Script | Purpose |
|--------|---------|
| `colab_wsl_setup.sh` | One-time: pipx + `google-colab-cli` |
| `colab_auth_wsl.sh` | One-time OAuth (`colab sessions`) |
| `colab_preflight.sh` | Check colab, HF_TOKEN, checkpoint, OAuth |
| `colab_pack_repo.sh` | `npm run build` + pack tarball (validates **&lt;250MB**, required files) |
| `colab_locomo_launch.sh` | Preflight + full automate (entry point) |
| `colab_locomo_hf_automate.sh` | `colab new` → upload → exec → 5-min sync → `stop` |
| `colab_locomo_hf.sh` | Remote: extract repo, HF adapters, ingest, eval |
| `colab_locomo_hf_driver.py` | Python wrapper for `colab exec` |
| `colab_smoke.sh` | **3 turns** smoke (`LOCOMO_LIMIT=3`, 30 min timeout) |
| `colab_locomo_from_windows.ps1` | PowerShell launcher |

---

## Tomorrow — recommended order

### 1. Smoke test (do this first)

```powershell
o krishnachhftoken
$env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:WSLENV = "HF_TOKEN/u"
wsl -d Ubuntu-24.04 -u chkri bash /mnt/c/Users/chkri/source/repos/PSM/psm-model/scripts/colab_smoke.sh
```

**Pass criteria:** `ingested 3` in Colab output; `pod-sync/locomo-hf-prod-v5k-two-pass-nfull.ingest.log` shows real remember traffic (not `ENOENT`).

### 2. Full resume

```powershell
o krishnachhftoken
$env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:WSLENV = "HF_TOKEN/u"
.\psm-model\scripts\colab_locomo_from_windows.ps1
```

Or from WSL:

```bash
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN='...'
bash /mnt/c/Users/chkri/source/repos/PSM/psm-model/scripts/colab_locomo_launch.sh
```

**Expect:** ~3–6+ hours on T4 for remaining **2922** turns (sequential two-pass). Sync every **5 min** to `benchmark/locomo/results/pod-sync/`.

### 3. Monitor

```bash
/home/chkri/.local/bin/colab status -s psm-locomo-hf
/home/chkri/.local/bin/colab sessions
python psm-model/scripts/_locomo_progress.py   # if DB synced locally
```

```powershell
# decision count from synced DB
python -c "import sqlite3; c=sqlite3.connect(r'benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db'); print(c.execute('select count(*) from decisions').fetchone()[0])"
```

### 4. After ingest completes

Retrieval eval JSON should land at:

`benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull-results.json`

(pulled by automate `cleanup` + periodic sync)

---

## Code changes (uncommitted — needed for Colab)

| File | Change |
|------|--------|
| `benchmark/locomo/src/ingest-psm-model.ts` | `--offset` for resume |
| `prod_memory/hf_prompts.py` | User text in inference prompt |
| `prod_memory/eval_classify.py` | `store_episodic` support |
| `prod_memory/lean_format.py` | Parse store variants |
| `psm_model/hf_remember_server.py` | Long-lived two-pass HF server |
| `src/psm-core/src/remember-server.ts` | Spawn HF server when adapters set |
| `psm-model/scripts/colab_*.sh` | WSL/Colab automation (this session) |

**Important:** Colab must use **local pack**, not GitHub clone — GitHub may lag uncommitted fixes.

---

## RunPod LoCoMo (fallback)

| Pod | Status | Notes |
|-----|--------|-------|
| `cyveakf0qhvqih` | **EXITED** | Last good sync **2026-06-26T14:34:29Z**; proxy `cyveakf0qhvqih-644111e0` |

Resume (if credits restored):

```powershell
python psm-model/scripts/_run_locomo_hf_full.py --pod-id cyveakf0qhvqih --proxy-user cyveakf0qhvqih-644111e0 --limit 0
python psm-model/scripts/_watch_locomo_sync.py --pod-id cyveakf0qhvqih --proxy-user cyveakf0qhvqih-644111e0 --interval-sec 300
```

**Do not** `--deploy` without `--force-deploy` — avoids duplicate pods.

---

## Architecture reminder

```text
Node ingest-psm-model.js
  → PsmModelRuntime (hf-two-pass)
    → remember-server.ts spawns python3 -m psm_model.hf_remember_server
      → gate LoRA (ignore/store) → extract LoRA (minimal_extract)
  → MemoryStore (SQLite)
```

Colab env: `PSM_RUNPOD=0`, `device=cuda`, `PYTHONPATH=psm-model/src:psm-model/prod-memory`.

---

## If smoke fails tomorrow

1. Read **`benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.ingest.log`**
2. Common lines:
   - `ENOENT` python → `--python python3` missing on remote script re-upload
   - `hf: command not found` → `pip install 'huggingface_hub[cli]'` on VM
   - `remember server exited` → CUDA/OOM; try `COLAB_GPU=L4` or reduce batch
3. Re-upload only scripts (session still up):

   ```bash
   colab upload -s psm-locomo-hf /mnt/c/.../colab_locomo_hf.sh /content/colab_locomo_hf.sh
   ```

4. CRLF regression: `wsl sed -i 's/\r$//' psm-model/scripts/colab*.sh`

---

## Not started / deferred

- LoCoMo **answer** eval (OpenRouter) — after retrieval eval
- Upload final LoCoMo results to HF
- Git commit of today's fixes
- LoCoMo ≥95% parse gate for production LoCoMo sign-off

---

## One-liner state

**Gate 10/10. LoCoMo half-done on RunPod (2960/5882). Colab WSL path built and OAuth done; tomorrow: smoke 3 turns → full resume with `colab_locomo_launch.sh` and local 18MB pack — not GitHub clone.**
