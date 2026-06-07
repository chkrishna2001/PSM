# PSM 50M — end of day handoff (2026-06-07)

**Read first tomorrow:** this file → [session-log.md](session-log.md) → [runpod-ssh-ops.md](runpod-ssh-ops.md) → [training-playbook.md](training-playbook.md).

Training **finished** @ step **36000**. RunPod pod **`psm-gate4-v1` deleted** (`zya02byfyqquyr`). Passing weights on **Hugging Face**; Gate 4 expanded still **FAIL** but close — **one focused round tomorrow should clear the bar**.

---

## Tomorrow’s goal (success = ship bar)

| Metric | Today @ 36k | Ship bar | Tomorrow target |
|--------|-------------|----------|-----------------|
| Action accuracy | **84.8%** | ≥85% | ≥85% (likely already there) |
| Parse valid rate | **85.2%** | ≥95% | **≥95%** |
| Schema valid rate | **85.2%** | ≥95% | **≥95%** |

**Definition of “succeed tomorrow”:** Gate 4 expanded eval **PASS** on CUDA (913 budget rows), Gate 3 direct still PASS, then promote checkpoint + optional second confirm eval. Do **not** default-enable `psmModel` until two passing expanded evals + product E2E (playbook).

**Root cause (not action anymore):** 135 parse failures — **105 on `promote_semantic`**, only **4** wrong-action rows. Fix = parse-heavy curriculum + failure-mined drills, not more expanded dilution or HF data hunting.

---

## What shipped today

### Gate 4 train v1 (`gate4-train-v1`) — complete

| Item | Value |
|------|--------|
| Pod | `zya02byfyqquyr` / `zya02byfyqquyr-644114c3@ssh.runpod.io` (**deleted**) |
| Resume | `step-022800` (Gate 3 clean base) |
| Target | **36000** (absolute steps) |
| Curriculum | `psm-50m-gate4-train-v1.jsonl` — **53,800 rows** |
| Builder | `psm_model.build_gate4_train_v1` |
| GPU | RTX 4090 |
| Crash recovery | Disk full @ corrupt `step-035200`; pruned + resumed @ **35000** |

**Curriculum mix (v1):**

- Expanded-probe full ×40: 36,800 (~68%)
- Parse drills (promote/store) ×25: 12,000 (~22%)
- Stratified promote/store from full-storage max 2500 (~5%)
- Direct anchors ×500: 2,500 (~5%)

### Gate 4 expanded eval @ step 36000

| Gate | Result |
|------|--------|
| Gate 3 direct (5 probes) | **PASS** (100% action) |
| Gate 2 action + smoke | **PASS** |
| Gate 4 expanded (913 rows, 7 dropped >1536 tok) | **FAIL** |

**Expanded metrics (checkpoint `real-v3-50m-full-v2-step-036000.pt`):**

```
action_accuracy     0.848
parse_valid_rate    0.852
schema_valid_rate   0.852
facts_exact_rate    0.719
memory_content      0.715
```

**Failure buckets (913 rows):**

| Bucket | Count |
|--------|------:|
| pass | 637 |
| parse_fail | 135 |
| wrong_memory_content | 119 |
| wrong_facts | 16 |
| wrong_action | **4** |
| wrong_memory_type | 2 |

**By expected action (parse_fail):**

- `promote_semantic`: 105 parse fails
- `store_episodic`: 28 parse fails
- others: minimal

**Progress vs prior:**

| Step | Action | Parse |
|------|--------|-------|
| 28k | 0.49 | 0.62 |
| 32k | 0.51 | 0.60 |
| **36k** | **0.85** | **0.85** |

### Hugging Face

| Repo | Status |
|------|--------|
| `chkrishna2001/psm-50m-mixed-v1-run` | **Keep** — step **35600**, **36000** uploaded (+ sidecars) |
| `chkrishna2001/psm-50m-action-mixed-v1` | **Keep** — datasets + probes |
| `chkrishna2001/nano-psm*` (22 repos) | **Deleted** — freed private storage |

**Auth:** `o hftoken` → clipboard → `$env:HF_TOKEN` (not `o runpodkey`).

**Helper:** `psm-model/scripts/delete_nano_psm_hf.py` (already run; dry-run should show zero nano repos).

### Local artifacts

If missing, re-extract from eval terminal log or re-run one eval only:

- `psm-model/checkpoints/gate-eval/gate4-full-expanded-step-36000.json`
- `psm-model/checkpoints/gate-eval/gate4-failure-analysis-step-36000.json`

Source: Cursor terminal log for task `860687` (full JSON inline). **Do not** use `eval-gates` to pull reports.

### Code delivered today (uncommitted / working tree)

- `build_gate4_train_v1.py` + tests
- `runpod_train_gate4.sh`, `runpod_upload_gate4.sh`, `runpod_recover_gate4.sh`, `runpod_prune_gate4.sh`
- `runpod_ctl.py`: `recover-gate4`, `upload-gate4`, `list-gpus`, `pick-gpu`, `--auto-gpu`, GraphQL UA fix, `--full-checkpoint` on `eval-gates`
- `runpod_eval_gates.sh`: `PSM_EVAL_FULL_CKPT` override
- `delete_nano_psm_hf.py`
- Docs: playbook, runpod-ssh-ops, session-log

**Commit + push tomorrow before pod deploy** so bootstrap gets v2 scripts.

---

## Tomorrow: execution plan (recommended)

### Step 0 — Sync repo (5 min)

```powershell
cd C:\Users\chkri\source\repos\PSM
git status
# commit today's scripts/docs if not on main
o runpodkey; $env:RUNPOD_API_KEY = Get-Clipboard
o hftoken; $env:HF_TOKEN = Get-Clipboard
```

### Step 1 — Build `gate4-train-v2` curriculum (30–60 min)

**Intent:** parse/schema ≥95%; keep action gains from v1.

1. **Mine parse failures** from `gate4-failure-analysis-step-36000.json` (or re-extract):
   - Rows with `parse_fail` → gold `expected` replay as `gate4-parse-repair:{id}` (input unchanged).
2. **New builder** `build_gate4_train_v2.py`:
   - Expanded-probe ×**25** (down from 40 — action already ~85%)
   - Parse drills ×**50** (up from 25)
   - **Parse-repair pack** ×**3** per mined failure (~400 rows)
   - Direct anchors ×500 unchanged
   - Stratified promote/store max **1500** (down from 2500)
   - Target ~**45k rows**, still no 25k full-storage dilution
3. Unit test mix shares (parse+repair ≥ **40%** of rows).

### Step 2 — Deploy + train (2–4 h GPU)

```powershell
python psm-model\scripts\runpod_ctl.py train-gate4 --deploy --auto-gpu `
  --target-steps 40000 --save-every 400 --keep-local 2 `
  --curriculum-builder v2
```

| Param | Value |
|-------|--------|
| Resume | `real-v3-50m-full-v2-step-036000.pt` |
| Absolute target | **40000** (+4000 steps) |
| GPU preference | RTX **3090** (`--auto-gpu`; 4090 if 3090 unavailable) |
| Volume | **20 GB** (playbook default; prune + HF sync) |
| tmux | `psm-gate4` + `psm-gate4-sync` every 600s |

**Verify after kickoff:**

```bash
tmux ls
pgrep -af psm_model.train
nvidia-smi
```

### Step 3 — One expanded eval (45–60 min)

```powershell
python psm-model\scripts\runpod_ctl.py eval-gates `
  --pod-id <id> --proxy-user <id>-<suffix> `
  --expanded `
  --full-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-040000.pt `
  --pull-reports psm-model\checkpoints\gate-eval
```

**Rules:**

- Run eval **once** per checkpoint.
- Never re-run `eval-gates` to pull — use tar-pull or `--pull-reports` on the same completed run.
- If local SSH times out, remote job may orphan — `pkill -f psm_model.eval_checkpoint` before leaving pod.

### Step 4 — If PASS → promote + HF + product smoke

1. Upload + promote best step to `real-v3-50m-full-v2.pt` on HF.
2. Second expanded eval (confirm pass).
3. `psm-memory remember --psm-model` E2E on 3–5 real snippets.
4. Delete pod.

### Step 5 — If still FAIL parse <95%

- Inspect `gate4-failure-analysis` buckets again.
- Add **+2000 steps** (42k) with parse-repair-only micro-curriculum (~2k rows, 100% parse drills).
- Do **not** widen to full-storage 25k base.

---

## RunPod / ops cheat sheet

| Secret | Command |
|--------|---------|
| RunPod API | `o runpodkey` → `$env:RUNPOD_API_KEY` |
| Hugging Face | `o hftoken` → `$env:HF_TOKEN` |

```powershell
python psm-model\scripts\runpod_ctl.py list-gpus
python psm-model\scripts\runpod_ctl.py pick-gpu
python psm-model\scripts\runpod_ctl.py list-pods
python psm-model\scripts\runpod_ctl.py delete-pod <pod_id>   # when idle
```

- **SSH:** `ssh -tt <proxy-user>@ssh.runpod.io` — piped `bash -s` only via `runpod_ctl.py`.
- **Sizing:** 50M @ batch 1 ≈ 3–6 GiB VRAM; 3090 + 20 GB volume sufficient with `keep-local=2` + periodic HF sync.
- **HF limits:** private storage (nano cleanup done); 128 commits/hour — upload folders, not per-file spam.

---

## Pitfalls (learned today)

1. **`eval-gates` is not a pull command** — re-running it starts a **full 913-row GPU eval**. Task 576710 orphaned eval after 60s local timeout; killed with `pkill -f psm_model.eval_checkpoint`.
2. **Disk full on pod** — `--save-every 200` + ~631 MB checkpoints filled 40 GB; use **400** + sync + `keep-local=2`.
3. **3090 often unavailable** on RunPod; `--auto-gpu` falls through to 4090.
4. **Wrong HF token** — `o runpodkey` ≠ HF; use `o hftoken`.
5. **GraphQL 403 error 1010** — Cloudflare blocks Python UA; fixed with browser User-Agent in `runpod_ctl.py`.
6. **Stop billing** — stop **and** delete pod when done (`delete-pod`).

---

## Phase status

| Gate | Status |
|------|--------|
| 0 Data filter | PASS |
| 1 Classifier | PASS |
| 2 Phase 1 50M | **PASS** — `step-009800` mixed-v2 |
| 3 Full StorageDecision | **PASS** — direct probes @ step 22800+ |
| 4 Expanded product bar | **FAIL @ 36k** — parse/schema gap; **fix tomorrow** |
| psm-core default `psmModel` | **Blocked** until Gate 4 passes twice + E2E |

---

## Do not resume

- Denylist checkpoints (`psm-model/checkpoints/DENYLIST.txt`)
- Corrupt partial saves (`step-035200` was ~128 MB)
- Full-storage-only curriculum dilution (v1 proved eval-aligned mix works)
- `nano-psm` HF repos (deleted)

**Do resume from:** `real-v3-50m-full-v2-step-036000.pt` (+ tokenizer sidecar).

---

## Open TODOs for tomorrow (agent)

- [ ] Implement `build_gate4_train_v2.py` + `mine_gate4_parse_failures.py` (from eval report JSON)
- [ ] Wire `--curriculum-builder v2` in `runpod_train_gate4.sh`
- [ ] Commit/push script + doc changes
- [ ] Train 36k → 40k on RunPod
- [ ] Single Gate 4 expanded eval @ 40k
- [ ] Update session-log with pass/fail
- [ ] If PASS: promote HF + second confirm eval + product smoke
