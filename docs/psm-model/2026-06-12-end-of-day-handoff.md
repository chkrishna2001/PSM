# PSM 50M — end of day handoff (2026-06-12)

**Read first tomorrow:** this file → [training-playbook.md](training-playbook.md) Gate 5 section → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on RunPod** — pod `q8cewx8srcf3oh` deleted after HF upload. **No GPU billing.**

**Gate 4 storage bar still met.** **Gate 5 recall not ship-ready** — dual gate failed at 051000; phase-2 recall training crashed before 058000 eval.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Best storage checkpoint** | **048000** (Gate 4 promoted) — parse ~99.6%, action ~99.6% |
| **Latest trained weights** | **057000** on HF (phase 2 partial); also **56800**, **57000** synced |
| **Phase 2 train** | Crashed ~step **57200** during `torch.save` (zip write error) |
| **Dual gate @ 051000** | Storage **PASS**, recall **FAIL** (~4.3% parse on 23 probes) |
| **Dual gate @ 058000** | **Not run** (train did not reach target) |
| **HF model repo** | `subbu83/psm-50m-mixed-v1-run` (`o chinnahftoken`) |
| **HF dataset repo** | `chkrishna2001/psm-50m-action-mixed-v1` |
| **RunPod pods** | **None** |
| **Ship bar** | **Not met** — need recall gate pass on dual eval |

### Local artifacts

| Path | Notes |
|------|--------|
| `psm-model/checkpoints/gate-eval/gate5-dual-step-051000.json` | Phase 1 dual eval (storage pass, recall fail) |
| `psm-model/data/curriculum/psm-50m-gate5-train-v1.jsonl` | Phase 1 mix (~**0.4%** recall — root cause of recall fail) |
| `psm-model/data/curriculum/psm-50m-gate5-train-v2-recall-heavy.jsonl` | Phase 2 mix (~**38.5%** recall, 29.9k rows) |
| `psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl` | 23 recall/context probes |
| `probes/expanded-probe-v1-filtered.jsonl` | Storage eval probes (913 rows) |

---

## What we accomplished today

### Gate 5 phase 1 (L4 pod `4c8fxr2bnerow3`, deleted)

1. Deployed L4, fixed bootstrap (missing expanded probes, `infer_row_task` export).
2. Trained **048000 → 051000** on `psm-50m-gate5-train-v1.jsonl` (storage-heavy mix).
3. In-train expanded probe @ 051000: **100%** action-prefix accuracy.
4. Dual eval @ 051000: **storage PASS**, **recall FAIL** (model still emits storage format).
5. Uploaded 051000 + 050800 to HF; pod deleted when moving to A5000.

### Gate 5 phase 2 (A5000 pod `q8cewx8srcf3oh`, deleted)

1. Built **recall-heavy** curriculum (`--profile recall-heavy`: expanded×20, recall×500).
2. Deployed **RTX A5000** @ **$0.27/hr** (`--auto-gpu`, A5000 preferred over L4).
3. Resumed **051000 → target 058000**, batch **16**, LR **5e-5**.
4. Training ran ~3h; metrics showed `recall_plan` rows in batches (curriculum working).
5. **Crash** during checkpoint save near step 57200:
   ```
   RuntimeError: unexpected pos 410302336 vs 410302228
   ```
   Last **good** local save: **057000**; **56800** / **57000** also on pod before delete.
6. **HF upload** before delete: synced steps **56800**, **57000**; `RESUME_STEP=057000`.
7. Pod **deleted** — no billing.

### Code / ops changes (local, may be uncommitted)

| Change | File(s) |
|--------|---------|
| `infer_row_task` export | `psm-model/src/psm_model/data/__init__.py` |
| Gate5 probe bootstrap fallback | `runpod_train_gate5.sh`, `runpod_start_gate5_train_only.sh` |
| Curriculum `--profile` (`bridge` / `recall-heavy`) | `build_gate5_train_v1.py` |
| A5000 default GPU + preference order | `runpod_ctl.py` (`DEFAULT_GPU`, `PSM_GPU_PREFERENCES`) |
| Phase 2 train-gate5 defaults | resume **051000**, target **58000**, profile **recall-heavy** |
| Tar-push artifacts on train-gate5 | `runpod_ctl.py` (probes, curriculum) |

---

## Root cause: why recall failed phase 1

Phase 1 curriculum `psm-50m-gate5-train-v1.jsonl` was **~99.6% storage / ~0.4% recall** (460 recall rows vs 23k storage). Playbook “25–35% recall” assumed `direct_probes.jsonl` on disk (not present). **3000 steps at ~0.4% recall mass cannot teach `recall_plan` JSON.**

Phase 2 fixed mix to **~38.5% recall** but train crashed before dual eval.

---

## Tomorrow — priority order

### P0 — Finish recall training + dual gate

1. **Resume from HF** — use `real-v3-50m-full-v2-step-057000.pt` (or eval **57000** first if triple intact on HF).
2. **Curriculum:** `psm-50m-gate5-train-v2-recall-heavy.jsonl` (pre-built locally; rebuild with `--profile recall-heavy` if needed).
3. **Train** 057000 → **058000** (or 06000 if more recall steps needed), batch 16, LR 5e-5.
4. **HF sync** every 120s (`tmux psm-gate5-sync`) — cold deploy did not leave sync log; wire explicitly on warm start.
5. **Dual eval** @ target step:
   ```powershell
   python psm-model\scripts\runpod_ctl.py eval-gate5-dual --pod-id <id> `
     --proxy-user <pod>-<suffix>@ssh.runpod.io --eval-step 58000 `
     --pull-reports psm-model\checkpoints\gate-eval
   ```
6. **Promote only if** `passed: true` (storage + recall).

### P0 — Checkpoint save reliability

- Investigate `torch.save` zip error on RunPod volume (disk full? concurrent write?).
- Consider `atomic` save (write `.tmp` then rename) in `train.py` `_save_training_checkpoint`.
- Verify corrupt partial at ~57200 is **not** on HF (upload reported `uploaded_count: 0` for new files; 56800/57000 synced).

### P1 — If recall still fails @ 058000

- Extend to **062000** with same recall-heavy mix.
- Expand `generate_recall_curriculum.py` (more than 23 probes / paraphrases).
- Optional: dedicated recall-only micro-run at lower LR with storage eval every 400 steps.

### P1 — Product (unchanged from 06-11)

- Daemon / hook still need full `PsmModelRuntime` path for recall at inference (not only storage).
- `psmModel.enabled` still opt-in.

---

## RunPod quick commands

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'
$env:DATASET_HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()

# Deploy A5000 (preferred) + phase 2 resume
python psm-model\scripts\runpod_ctl.py train-gate5 --deploy --auto-gpu `
  --name psm-gate5-a5000 --profile recall-heavy `
  --curriculum psm-model/data/curriculum/psm-50m-gate5-train-v2-recall-heavy.jsonl `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-057000.pt `
  --tokenizer psm-model/checkpoints/real-v3-50m-full-v2-step-057000.tokenizer.json `
  --target-steps 58000 --batch-size 16 --learning-rate 5e-5 `
  --keep-pod --proxy-user <pod>-<suffix>
```

**GPU policy:** A5000 ($0.27/hr, 24GB) first; L4 fallback. **Never delete pod** until HF triple for resume step verified.

---

## Gate thresholds (reminder)

**Storage (Gate 4):** parse ≥95%, action ≥85%, …  
**Recall (Gate 5):** parse ≥95%, `target_tables_exact` ≥90%, `ranking_hints_score` ≥0.50, …  
**Ship:** `eval_dual_gate` → `passed: true` on both.

---

## Session log pointer

See [training-playbook.md](training-playbook.md) Gate 5 / RunPod sections for curriculum build and ctl flags.
