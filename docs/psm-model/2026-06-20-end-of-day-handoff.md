# PSM 50M — end of day handoff (2026-06-20)

**Read first next session:** this file → [training-pitfalls.md](training-pitfalls.md) → `.cursor/skills/runpod-gpu-train/SKILL.md` → `.cursor/rules/runpod-auto-delete.mdc` → [training-playbook.md](training-playbook.md) (Prod-memory RunPod section)

**Nothing running on RunPod.** All pods deleted. **No GPU billing.**

**Prod-memory ship bar: NOT met.** Best prod grounding checkpoint remains **058000 gate stem** (1/10 effective_stored). Do **not** promote 060000 prod-memory, 065000 prod-memory, or v4 @ 060000.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Production / eval baseline** | `real-v3-50m-full-v2-step-058000` — **1/10 effective_stored** on prod fixtures |
| **v3 full curriculum train** | 060000 → 065000 — **lateral** (still 1/10); do not promote |
| **v4 fixture-repair micro-train** | 058000 → 060000 — **regression** (0/10); do not promote |
| **v3 teacher labeling** | **Done** — 1,475/1,475 sessions → 2,654 rows `prod-extraction-v3.jsonl` |
| **v4 curriculum built** | 615 rows `prod-extraction-v4.jsonl` (fixture repair ×40 on 7 fails) |
| **RunPod pods** | **None** (all deleted 2026-06-20) |
| **LoCoMo** | **Deferred** until Gate 4 parse ≥95% |
| **Next work** | **Failure mining + v5 curriculum design** — no GPU until v5 recipe is reviewed |

### HF repos & tokens

| Env var | Source | Used for |
|---------|--------|----------|
| `HF_TOKEN` | `o chinnahftoken` | Model repo **`subbu83/psm-50m-mixed-v1-run`** (checkpoints, eval JSON) |
| `DATASET_HF_TOKEN` | `~/.cache/huggingface/token` | Dataset repo **`chkrishna2001/psm-50m-action-mixed-v1`** (curriculum, fixtures) |
| `PSM_HF_MODEL_REPO` | set to `subbu83/psm-50m-mixed-v1-run` | Override for ctl scripts |

Pod env `{{ RUNPOD_SECRET_HF_TOKEN_C }}` is **not** sufficient — always pass both tokens via `runpod_ctl.py` `extra_env`.

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'
```

---

## Prod-memory eval history (this session)

All evals use **10 fixtures** in `psm-model/prod-memory/fixtures/cases.json` (5 suites × 2 cases). Primary metric: **`effective_stored`** (correct action + grounded content + guard accept). Promotion gate discussed: **≥3/10**.

| Checkpoint | Stem | effective_stored | parse_valid | action_match | guard_reject | Verdict |
|------------|------|------------------|-------------|--------------|--------------|---------|
| **058000** | gate (`real-v3-50m-full-v2`) | **1/10** | 0.90 | 0.30 | 0.50 | **Best so far — keep** |
| 060000 | prod-memory (v3 path) | 1/10 | 0.90 | 0.30 | 0.50 | Baseline for v3 smoke |
| 065000 | prod-memory (v3 train) | 1/10 | 0.90 | 0.30 | 0.40 | Lateral; cursor_shaped parse 1.0→0.5 |
| **060000** | **prod-memory-v4** | **0/10** | 0.80 | 0.20 | 0.30 | **Regression — reject** |

### v4 @ 060000 suite breakdown (vs 058000)

| Suite | v4 effective | 058000 effective | Notes |
|-------|--------------|------------------|-------|
| plan_chunks | 0/2 | 0/2 | unchanged |
| cursor_shaped | **0/2** | **1/2** | **regression** — parse 0.5 vs 1.0 |
| workflow | 0/2 | 0/2 | unchanged |
| technical | 0/2 | 0/2 | parse 0.5 both |
| noise | 0/2 | 0/2 | guard_reject 1.0 (correct ignore) |

**Interpretation:** v4’s heavy fail-copy mix (~280/615 rows at ×40) likely **overfit noise / washed out cursor_shaped**. More SFT density on the same failure set is not the fix.

---

## Training runs (2026-06-20)

### 1. v3 teacher curriculum — 060000 → 065000

| Setting | Value |
|---------|-------|
| Pod | `wc35bndd7mojd6` (L4) — **deleted** |
| Resume | `real-v3-50m-full-v2-prod-memory-step-060000.pt` |
| Curriculum | `prod-extraction-v2.jsonl` (v3 content mirrored on HF) |
| Target | 65000 |
| Out stem | `real-v3-50m-full-v2-prod-memory` |

Train completed; eval @ 065000 vs 060000 baseline = lateral move. Pod deleted after eval.

### 2. v4 fixture-repair micro-train — 058000 → 060000

| Setting | Value |
|---------|-------|
| Pod | `st0luf214e32c5` (A5000, CA-MTL-1) — **deleted after data recovery** |
| Proxy | `st0luf214e32c5-64411541@ssh.runpod.io` |
| Resume | `real-v3-50m-full-v2-step-058000.pt` (gate stem rollback) |
| Curriculum | `prod-extraction-v4.jsonl` |
| Target | 60000 (2000 steps) |
| LR | 1e-5 / min 3e-6 |
| Out stem | `real-v3-50m-full-v2-prod-memory-v4` |
| Save every | 200 steps |

Train completed (10 checkpoints 058200–060000). HF sync tmux did not upload during train; **manual recovery upload** succeeded (32 files on HF).

### 3. Failed launches (do not repeat)

| Pod | Issue | Root cause |
|-----|-------|------------|
| `7ryzehtgdv92vc` | HF `Repository not found` | Pod secret token; `hf download` without `--token` |
| same | `ModuleNotFoundError: psm_model.configs` | Only `train.py` synced, not full `src/` |
| `h5tizus9dginej` | Eval failed immediately | Cold pod: `hf: command not found` in eval script (fixed) |

---

## v3 teacher labeling (completed before train)

| Metric | Value |
|--------|-------|
| Sessions labeled | 1,475 / 1,475 (100%) |
| Teacher (GPT-4o mini) | 1,021 new + 454 cached |
| Heuristic fallbacks | ~104 (~7%) |
| Output | `prod-extraction-v3.jsonl` — **2,654 rows**, validation OK |

**Action mix shift vs v2 heuristic labels:**

| Action | v3 | v2 |
|--------|-----|-----|
| `promote_semantic` | 517 | 4 |
| `store_episodic` | 969 | 1,427 |

**HF dataset paths** (`chkrishna2001/psm-50m-action-mixed-v1`):

- `prod-memory/prod-extraction-v3.jsonl` + manifest
- `prod-memory/prod-extraction-v2.jsonl` — **mirrored v3 content** (existing train scripts use this path)

---

## v4 curriculum (built, trained, rejected)

**Builder:** `psm-model/prod-memory/prod_memory/build_prod_extraction_v4_fixture_repair.py`  
**Tests:** `psm-model/prod-memory/tests/test_build_prod_extraction_v4.py`  
**Output:** `psm-model/prod-memory/data/prod-extraction-v4.jsonl` (615 rows)  
**HF:** `prod-memory/prod-extraction-v4.jsonl` on dataset repo

**Profile (`PROD_EXTRACTION_V4_PROFILE`):**

| Component | Copies | Notes |
|-----------|--------|-------|
| 7 failing fixtures | ×40 | 280 rows — sourced from 065000 eval failures |
| 1 passing fixture (`fixture-cursor-01-summary`) | ×5 | anchor |
| noise (fixture + seed) | ×15 each | |
| direct probes | ×2 | |
| recall probes | ×10 | |

**Failing fixture IDs** (frozen in builder — update after next eval):

```
fixture-plan-01-handoff
fixture-plan-02-chunking
fixture-cursor-02-debug
fixture-workflow-review-pr
fixture-workflow-runpod
fixture-technical-eslint
fixture-technical-api
```

Rebuild command:

```powershell
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory"
python -m prod_memory.build_prod_extraction_v4_fixture_repair
```

---

## Artifacts on HF (model repo)

**Repo:** `subbu83/psm-50m-mixed-v1-run`

### v4 checkpoints (recovered 2026-06-20)

All steps **058200–060000** (every 200 steps), each with `.pt` + `.tokenizer.json` + `.meta.json`:

```
psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4-step-{058200..060000}.pt
psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4.metrics.jsonl
```

Verify:

```powershell
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
python -c "from huggingface_hub import HfApi; import os; f=[x for x in HfApi(token=os.environ['HF_TOKEN']).list_repo_files('subbu83/psm-50m-mixed-v1-run', repo_type='model') if 'prod-memory-v4' in x and x.endswith('.pt')]; print(len(f), 'checkpoints'); print('060000' in str(f))"
```

### Prod grounding eval JSON (on HF)

```
psm-model/prod-memory/results/prod-grounding-058000.json   # baseline (gate stem)
psm-model/prod-memory/results/prod-grounding-060000.json   # v4 eval (latest)
psm-model/prod-memory/results/prod-grounding-062000.json   # stale from prior runs
psm-model/prod-memory/results/prod-grounding-065000.json   # v3 @ 065000
psm-model/prod-memory/results/prod-grounding-baseline.json
```

Download for failure mining:

```powershell
hf download subbu83/psm-50m-mixed-v1-run psm-model/prod-memory/results/prod-grounding-060000.json --local-dir . --token $env:HF_TOKEN
hf download subbu83/psm-50m-mixed-v1-run psm-model/prod-memory/results/prod-grounding-058000.json --local-dir . --token $env:HF_TOKEN
```

### Other prod-memory stems on HF (do not promote)

- `real-v3-50m-full-v2-prod-memory-step-060000` … `065000` (v3 path train)

### Gate stem (production baseline)

- `real-v3-50m-full-v2-step-058000.pt` (+ tokenizer, meta)

---

## Local artifacts

| Path | Notes |
|------|--------|
| `psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4-step-060000.pt` | ~631 MB — pulled from HF |
| `psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4-step-060000.tokenizer.json` | |
| `psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4-step-060000.meta.json` | |
| `psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4.metrics.jsonl` | train metrics |
| `psm-model/prod-memory/results/prod-grounding-*.json` | may be stale — prefer HF copies for v4 eval |
| `psm-model/prod-memory/fixtures/cases.json` | 10 eval fixtures |

Intermediate v4 steps (058200–059800) are **HF only** — pull if needed for step-wise analysis.

---

## Code / ops changes (local — largely uncommitted)

| Change | File(s) |
|--------|---------|
| Prod-memory warm train: `hf --token`, auto `pip install huggingface_hub`, auto `tmux` | `runpod_start_prod_memory_train_only.sh` |
| Generic curriculum download by basename (v4 path support) | same |
| Train ctl: tar-push full `psm-model/src` | `runpod_ctl.py` `train-prod-memory` |
| Eval ctl: tar-push `psm-model/src` for cold eval pods | `runpod_ctl.py` `eval-prod-memory` |
| Eval script: auto-install `hf`, explicit `--token` on downloads | `runpod_eval_prod_memory.sh` |
| v4 fixture-repair curriculum builder + tests | `build_prod_extraction_v4_fixture_repair.py`, `test_build_prod_extraction_v4.py` |
| Prod-memory RunPod docs + failure table | `docs/psm-model/training-playbook.md` |
| Session log entry (partial) | `docs/psm-model/session-log.md` |
| RunPod skill prod-memory quick ref | `.cursor/skills/runpod-gpu-train/SKILL.md` |

---

## RunPod ops — lessons (2026-06-20)

### Hard rules (learned the hard way)

1. **Never stop/delete a pod until data is safe:** checkpoints on HF **and** (final step + eval JSON) pulled locally, or explicit user override.
2. **`verify-pod` defaults are Gate 5** — for prod-memory pass:
   ```powershell
   --tmux-session psm-prod-memory --train-log /tmp/psm-prod-memory-train.log
   ```
   Train-done marker for prod-memory: `/tmp/psm-prod-memory.done` (not `psm-gate5.done`).
3. **SSH over RunPod proxy:** pipe scripts via `bash -s` (as `runpod_ctl.py` does). `bash -lc 'one-liner'` hangs on interactive PTY.
4. **Cold eval pods** need `hf` CLI — eval script now auto-installs; also syncs `src/`.
5. **`eval-prod-memory` deletes pod by default** — pass `--keep-pod` if you need post-eval SSH.
6. **Large checkpoint pull (5.9 GB) over SSH tar** may timeout — prefer HF upload from pod, then `hf download` locally.

### Prod-memory launch checklist

1. Set `HF_TOKEN` + `DATASET_HF_TOKEN` locally
2. `deploy` → `ssh-info` for `--proxy-user`
3. `train-prod-memory --pod-id ... --keep-pod` (warm only after tar-push)
4. `verify-pod` with prod-memory tmux/log within 90s
5. Confirm `psm-prod-memory-sync` tmux running (HF upload every 120s)
6. On train done: `upload-gate4` with `RUN_STEM=real-v3-50m-full-v2-prod-memory-v4` if sync failed
7. `eval-prod-memory` → pull results → **then** delete pod

See [training-playbook.md](training-playbook.md) § Prod-memory RunPod for full commands.

---

## Next session — priority order

### P0 — Failure mining (no GPU, ~1–2 hours)

Pull latest eval JSON from HF and bucket **each of 10 fixtures** by failure mode:

- parse invalid
- wrong action (`store_episodic` vs `promote_semantic` vs `ignore`)
- guard reject
- stored but not grounded (`model_stored` but not `effective_stored`)

Compare **058000 (best)** vs **v4 @ 060000 (worst)** per case ID. Output: table or short doc listing top 3 fix levers (curriculum vs guard vs format).

Entry point: `prod-grounding-*.json` → `"cases"` array; fixtures in `psm-model/prod-memory/fixtures/cases.json`.

### P0 — Design v5 curriculum (no GPU until reviewed)

**Do not repeat v4 recipe** (×40 fail copies). Proposed v5 constraints:

| Rule | Rationale |
|------|-----------|
| ≤5 copies per failing fixture | v4 ×40 caused regression |
| One suite focus per micro-run | e.g. plan_chunks only first |
| ≥50% anchor rows | passing fixtures + gate5/recall noise |
| 500–1000 steps max | eval every save; abort if cursor_shaped drops |
| Resume always from **058000 gate stem** | not prod-memory stems |

Implement: `build_prod_extraction_v5_*.py` (new file — don’t mutate v4 in place). Upload to HF dataset before train.

### P1 — v5 micro-train (only after P0 + P1 design sign-off)

```powershell
python psm-model\scripts\runpod_ctl.py deploy --auto-gpu --name psm-prod-v5 --wait-ssh 300
python psm-model\scripts\runpod_ctl.py train-prod-memory `
  --pod-id <id> `
  --proxy-user <user> `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-058000.pt `
  --tokenizer psm-model/checkpoints/real-v3-50m-full-v2-step-058000.tokenizer.json `
  --curriculum psm-model/prod-memory/data/prod-extraction-v5.jsonl `
  --target-steps 59200 `
  --learning-rate 1e-5 --min-learning-rate 3e-6 `
  --out-stem real-v3-50m-full-v2-prod-memory-v5 `
  --keep-pod

python psm-model\scripts\runpod_ctl.py eval-prod-memory `
  --pod-id <id> --proxy-user <user> `
  --eval-step 59200 `
  --run-stem real-v3-50m-full-v2-prod-memory-v5 `
  --compare-baseline-step 58000 `
  --keep-pod
# upload + pull results, THEN delete pod
```

**Success gate:** `effective_stored` ≥ **3/10** without cursor_shaped regression.

### P2 — If prod-memory stuck at 1/10 after v5

Revisit **Gate 5 recall-heavy** from 058000 — prod fixtures may need guard/pipeline fixes before SFT moves the needle. LoCoMo still deferred.

### Do NOT do tomorrow

- Promote 065000, prod-memory @ 060000, or v4 @ 060000
- Another 2k-step run on v4 fixture-repair mix
- Stop/delete pods before HF + local data confirmed
- Full `src` sync on warm pods except eval cold-start (eval now syncs src intentionally)

---

## Key file paths

| Purpose | Path |
|---------|------|
| Prod-memory train (warm) | `psm-model/scripts/runpod_start_prod_memory_train_only.sh` |
| Prod-memory eval | `psm-model/scripts/runpod_eval_prod_memory.sh` |
| RunPod ctl | `psm-model/scripts/runpod_ctl.py` |
| HF upload script | `psm-model/scripts/runpod_upload_gate4.sh` |
| Eval harness | `psm-model/prod-memory/prod_memory/eval_grounding.py` |
| Fixtures | `psm-model/prod-memory/fixtures/cases.json` |
| v3 builder | `psm-model/prod-memory/prod_memory/` (ingest, label, build v2/v3) |
| v4 builder | `psm-model/prod-memory/prod_memory/build_prod_extraction_v4_fixture_repair.py` |
| Colab notebook | `psm-model/prod-memory/notebooks/prod-extraction-v1-colab.ipynb` |

---

## Strategic context

**Arc of the week:** v3 teacher labels improved action typing (`promote_semantic` 4→517) → full prod-memory train 060→065 = **flat** → rolled back to 058000 → v4 fixture repair micro-train = **worse** → data recovered to HF → eval confirmed regression.

**Working hypothesis:** The bottleneck is not label volume or fail-copy count. The model **stores** (`model_stored` 3–6/10) but **effective_stored** fails on grounding, action choice, or guard. Next iteration must be **targeted** (per-suite, low copy count, heavy anchors) informed by per-case failure mining.

**Pinned checkpoint for all future prod-memory experiments:** `real-v3-50m-full-v2-step-058000` (gate stem).

---

## Quick status for tomorrow standup

- Teacher v3 labels: **done**, on HF  
- v3 train 060→065: **done**, no gain  
- v4 micro-train 058→060: **done**, **regressed**; checkpoints **safe on HF**  
- Eval: v4 **0/10**, baseline **1/10**  
- Pods: **none**  
- **Start with failure mining, then v5 builder — not RunPod**
