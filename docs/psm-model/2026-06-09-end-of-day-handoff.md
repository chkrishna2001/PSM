# PSM 50M — end of day handoff (2026-06-09)

**Read first tomorrow:** this file → [session-log.md](session-log.md) → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on RunPod** — pod `6c9efizq1aoocf` deleted. Gate 4 parse still **FAIL**; LoCoMo deferred.

---

## Where we are

| Item | Status |
|------|--------|
| RunPod pods | **None** (billing stopped) |
| Gate 4 expanded @ **42,000** (best) | **FAIL** — parse **88.2%**, action **88.1%** |
| Latest eval @ **42,400** (micro v3) | **FAIL** — parse **87.6%**, action **87.5%** |
| Micro v2 @ 42,800 | **FAIL** — parse **86.3%** (regression plateau) |
| Ship bar | parse/schema **≥95%**, action **≥85%** |
| HF model repo | `subbu83/psm-50m-mixed-v1-run` |
| HF resume weights | **42000** + **42400** triples uploaded before delete |
| LoCoMo | **Deferred** until parse ≥95% |

**Action passes bar on all runs. Parse/schema stuck ~87–88%** — same failure mode: `promote_semantic` tagged output (missing `R:`, malformed `F:` pipe fields, missing `reasoning`, `memory.strength` not numeric).

---

## What happened today (timeline)

1. **Pod** `6c9efizq1aoocf` / `psm-gate4-recover`, RTX A5000 ($0.27/hr), proxy `6c9efizq1aoocf-64411022@ssh.runpod.io`
2. **Micro v2** (42k→42.8k, structural_loss=4): completed; eval **86.3%** — no gain vs 43.4k regression
3. **Micro v3** (42k→42.4k, structural_loss=2, 18× repair copies): completed; eval **87.6%** — small bump, still below 42k baseline
4. **Polling failures:** background polls printed **39+ GPU 0%** lines without acting (`ls | tail -1` picked 043400 over 042800; eval done flag never set)
5. **HF upload before delete:** `pod_hf_push_steps.py` pushed **42000** + **42400** `.pt`/tokenizer/meta to `subbu83/psm-50m-mixed-v1-run`
6. **Pod deleted** with `--force-delete-pod` (local `file_exists` verify false-negative without token; uploads confirmed on pod)

---

## Eval scorecard (expanded, 913 probes)

| Step | Run | Parse/schema | Action | Parse fails | Notes |
|------|-----|--------------|--------|-------------|-------|
| 42000 | baseline | **88.2%** | 88.1% | 108 | **Best — resume here** |
| 42800 | micro v2 | 86.3% | 86.3% | 125 | Same as 43.4k plateau |
| 42400 | micro v3 | 87.6% | 87.5% | 113 | +1.3pp vs v2, −12 fails |

Local reports:
- `psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042000.json` (if pulled earlier)
- `psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042800.json`
- `psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042400.json`

Curricula mined:
- `psm-model/data/curriculum/gate4-parse-repair-step-043400.jsonl` (125 rows)
- `psm-model/data/curriculum/gate4-parse-repair-step-042800.jsonl` (125 rows)
- `psm-model/data/curriculum/psm-50m-gate4-train-micro-v2.jsonl`
- `psm-model/data/curriculum/psm-50m-gate4-train-micro-v3.jsonl`

---

## Why numbers are going down (not up)

**42k is the best checkpoint found — not a stepping stone.** Every micro run that trains *past* 42k has moved away from it:

| Checkpoint | Parse | vs 42k |
|------------|-------|--------|
| 42k baseline | **88.2%** | — |
| 42.4k micro v3 | 87.6% | −0.6pp |
| 42.8k micro v2 / 43.4k micro v1 | 86.3% | −1.9pp |

We are not slowly climbing toward 95%. We are **perturbing a good weights snapshot** and mostly getting worse; v3 only partially recovers.

### Root causes (why micro loops stall)

1. **Failures are concentrated** — ~100/125 parse fails are `promote_semantic` (malformed `F:` pipe fields, missing `reasoning`, `memory.strength` not numeric). Mining 125 rows and copying 12–18× is narrow fine-tuning, not fixing the underlying skill.
2. **Late-stage forgetting** — at 42k the model is near a local optimum on the main curriculum. A 98% parse-repair micro set + structural loss shifts weights toward repair templates and away from the distribution that produced 88.2%.
3. **Train metrics ≠ ship metrics** — training loss drops and inline probes look strong (~99% action), but **expanded eval (913 probes)** is flat or down. That gap is the whole problem.
4. **Same intervention → same plateau** — v1 @ 43.4k, v2 @ 42.8k, v3 @ 42.4k all land in **86–88%**. Repeating micro v4/v5 without changing the intervention class will likely burn GPU for the same churn.

### Is there an end to the current loop?

**Not with micro-only iterations from 42k.** The bar is 95% (~7pp above current best). Three micro attempts did not trend up. The honest ceiling for **50M + free tagged generation + no decode constraints** may be ~88% unless something structural changes (data mix, format drills in main training, rejection sampling, larger model, or constrained decode at product boundary).

### Is this how everyone does it?

**No.** Mine-failures → micro-train → eval → repeat is fine when each iteration clearly moves the needle. When three iterations regress or plateau, teams **change the intervention**, not just hyperparameters.

| Approach | Fits pure tagged decode (no hybrid)? |
|----------|--------------------------------------|
| Constrained / grammar-guided decoding | No — not free generation |
| Rejection sampling (train on parse-valid completions only) | **Yes** |
| Fact-format drills in **main** curriculum (not 400-step bursts) | **Yes** — `build_gate4_fact_format_drills.py` |
| Longer main run from 36k/42k with revised mix | **Yes** |
| RL/DPO with parse reward | **Yes** (heavier) |
| 400–800 step micro bursts on mined failures | What we did — **not working here** |

---

## Tomorrow’s fork in the road

**Do not spend RunPod money until we pick a path and define pass/fail before launch.**

| Path | What it is | Expected outcome |
|------|------------|------------------|
| **Pivot (recommended)** | Fact-format drills in a **longer main run** from 36k or 42k; and/or **rejection sampling**; optionally revisit decode constraints | Only path with a real shot past 88% without another micro churn loop |
| **Keep looping** | Micro v4/v5 — more repair copies, shorter steps, structural_loss=0 | Likely more **86–88%** churn; only try if pivot is blocked |

### Recommended: one non-micro experiment (sketch before $)

**Hypothesis:** `promote_semantic` fact-line format must be learned in the **main data distribution**, not patched in 400-step micro runs.

**Proposal A — fact-format main run (preferred)**
- Resume: **42k** (or 36k if we want more steps headroom)
- Curriculum: main gate4 mix + `build_gate4_fact_format_drills.py` woven in (high share of `F:` pipe / `R:` / `memory.strength` drills)
- Train: **2000–4000 steps** (not 400), `structural_loss_weight=0` or very low
- **Pass:** expanded eval parse **≥ 90%** (intermediate bar) — must **beat 88.2%** or discard weights and do not promote
- **Fail:** stop run early if inline probe parse collapses; do not train past best eval step

**Proposal B — rejection sampling**
- Generate on probe set; keep only parse-valid rows; mix into curriculum; main train from 42k
- Same pass/fail: must beat **88.2%** on expanded eval or revert to 42k

**Hard rule:** **42k is frozen production candidate.** No checkpoint is promoted unless it beats **88.2%** on expanded eval. Never resume 42400/42800/43400.

### If pivot is rejected: micro v4 (fallback only)

- Resume: **HF `step-042000.pt`**
- Mine **042400** failures (113 rows)
- `structural_loss_weight=0`, 400 steps → 42400, pin `GATE4_PINNED_STEPS=42000`
- Expectation: low — treat as last micro attempt before mandatory pivot

---

## First commands tomorrow (after choosing path)

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey; o chinnahftoken
$env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'

# 1) Confirm no pods
python psm-model/scripts/runpod_ctl.py list-pods

# 2) PIVOT (recommended): sketch curriculum + pass/fail in chat BEFORE deploy
#    e.g. fact-format drills main run 42k → 44000, structural_loss=0
# python psm-model/scripts/runpod_ctl.py train-gate4 `
#   --deploy --gpu "NVIDIA RTX A5000" `
#   --name psm-gate4-recover `
#   --proxy-user <proxy>@ssh.runpod.io `
#   --curriculum-builder v4 `   # or new builder for fact-format main mix
#   --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-042000.pt `
#   --target-steps 44000 `
#   --structural-loss-weight 0 `
#   --keep-pod

# 2-alt) FALLBACK ONLY: micro v4 (expect 86–88% churn)
# python psm-model/scripts/runpod_ctl.py train-gate4 `
#   --deploy ... --curriculum-builder micro `
#   --curriculum psm-model/data/curriculum/psm-50m-gate4-train-micro-v4.jsonl `
#   --resume-checkpoint ...-step-042000.pt `
#   --target-steps 42400 --structural-loss-weight 0 --keep-pod

# 3) After train: eval (warm path)
python psm-model/scripts/launch_gate4_eval_now.py --checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-042400.pt

# 4) Upload pinned steps BEFORE delete
python psm-model/scripts/runpod_ctl.py upload-gate4 --pod-id <id> --proxy-user <proxy>@ssh.runpod.io
# or: pod_hf_push_steps.sh with STEPS=42000,42400

# 5) Delete only after HF triples verified
python psm-model/scripts/runpod_ctl.py delete-pod <pod_id>
```

**Launch checklist (learned today):**
- Use `--proxy-user` + warm-pod path (no full src sync hang)
- Verify GPU **after 45s** — not during checkpoint load
- Poll scripts must check **target step file exists**, not `ls | tail -1`
- On first `GPU 0%` + report file present → pull eval immediately
- Fix `runpod_upload_gate4.sh` CRLF on pod before relying on upload script

---

## New scripts today (use these)

| Script | Purpose |
|--------|---------|
| `launch_micro_v2_now.py` / `launch_micro_v3_now.py` | Warm-pod micro train |
| `launch_gate4_eval_now.py --checkpoint ...` | Warm-pod expanded eval |
| `pull_eval_step.py <step>` | Pull report + print metrics |
| `pod_hf_fetch_42000.sh` | Restore resume ckpt from HF |
| `pod_hf_push_steps.sh` | Upload step triples (bypasses CRLF upload script) |
| `poll_micro_v2_chain.py` | Train→eval chain (fixed target-step check) |
| `poll_pod_gpu.py` | Now exits on idle (no infinite GPU 0% spam) |

---

## Registry

Updated `psm-model/checkpoints/gate4-checkpoint-registry.json`:
- **best:** step 42000 @ 88.2% parse (on HF)
- **latest_eval:** step 42400 @ 87.6%
- **recovery:** resume 42000, micro v3 curriculum

---

## Not doing until parse ≥95%

- LoCoMo ingest
- Product E2E wiring
- Pod auto-delete without HF upload
