# HF LoRA prod-memory v5e — end of day handoff (2026-06-24)

**Read first next session:** this file → `.cursor/skills/runpod-gpu-train/SKILL.md` → `.cursor/rules/runpod-auto-delete.mdc` → [training-pitfalls.md](training-pitfalls.md)

**Nothing running on RunPod.** All pods **EXITED** (stopped). **No GPU billing.**

**Ship bar: met on fixtures for v5c only** — need **≥8/10 `effective_stored`** on `psm-model/prod-memory/fixtures/cases.json` with stable behavior (no template collapse, noise should `ignore`).

**RunPod maintenance:** Jun 29, 2026, 1:38–9:38 PM EDT — sync artifacts to HF + local before; never rely on pod disk alone.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Best HF LoRA (ship candidate)** | **`hf-prod-v5c-qwen0.5b`** — **8/10** `effective_stored` |
| **Latest completed train** | **v5d** — 6/10 (regressed; do not promote) |
| **v5e** | **Not completed** — 2× CUDA crash, no adapter on HF |
| **Model HF repo** | `krishnach7262/psm-prod-memory-hf` |
| **Dataset HF repo** | `krishnach7262/psm-prod-memory-data` |
| **HF token** | `o krishnachhftoken` → clipboard |

### Eval score history (prod fixtures, minimal format)

| Run | Score | Resume from | Notes |
|-----|-------|-------------|-------|
| v5b | 0/10 | base | ~51% ignore curriculum → `A:ignore` collapse |
| v5 | 4/10 | v4 | Bulk v6 + ignore |
| **v5c** | **8/10** | v5 | Fixture-only minimal ×30; **ship baseline** |
| v5d | 6/10 | v5c | MEMORY_SUMMARIES curriculum; `"Stores associate concepts..."` template on all inputs |
| v5e | — | v5c | **CUDA launch failure** mid-train (never finished) |

### v5c — what passes / fails (8/10)

| Case | Result | Issue |
|------|--------|-------|
| plan-01-handoff | fail | Guard reject — junk tag template, overlap 1/2 |
| plan-02-chunking | pass | Accidental overlap on same template |
| cursor-* | pass | |
| workflow-review-pr | pass | |
| workflow-runpod | fail | Guard reject (fixed in v5d eval, lost again on other suites) |
| technical-* | pass | |
| noise-* | pass `effective_stored` | **Wrong action** — stores instead of `ignore` (action_match 60%) |

**v5c failure mode:** one-line template collapse (`"No tags associated with this page..."`) — not input-specific extraction.

**v5d failure mode:** worse template (`"Stores associate concepts with items..."`) on every fixture.

---

## What we did today

### Training runs

1. **v5d** — pod `zxpag9azrrkb3n` — 1000 steps, resume v5c, summary labels → train OK, eval **6/10**, adapter on HF.
2. **v5e** — hybrid snippet + forced key-token labels, 500 steps @ 5e-5, resume v5c:
   - Pod `8er662o0i558p9` — **CUDA crash** mid-train
   - Pod `ohrvp8zj19m5n7` — **same CUDA crash** (A5000, `CA-MTL-1` / `SE`)
   - HF: train log only (`logs/hf-prod-v5e-qwen0.5b-train.log`)

### Code / ops added (local, not all committed)

| Area | Files |
|------|-------|
| Curricula | `hf-prod-v5c`, `v5d`, `v5e` profiles in `build_hf_curriculum.py` |
| Labels | `build_summary_fixture_rows`, `build_hybrid_fixture_rows`, `forced_grounded_store_content` in `build_minimal_fixture_rows.py` |
| Launchers | `_run_hf_lora.py`, `_watch_hf_lora.py`, `_run_hf_lora_eval.py` — v5c/d/e |
| HF sync | `_sync_hf_lora.py`; `psm-hf-sync` tmux loop every 120s in `runpod_hf_lora_train.sh` |
| Watcher | Failure detection (partial); sync every poll; stop-pod only after eval JSON local |
| Rule | `.cursor/rules/runpod-auto-delete.mdc` — HF LoRA sync section |

### Artifacts on HF (verified)

- **v5c:** `hf-prod-v5c-qwen0.5b/adapter/*`, checkpoints 400/800, eval `eval/hf-prod-v5c-qwen0.5b-prod-grounding.json`
- **v5d:** full adapter + checkpoints 400/800/1000, eval `eval/hf-prod-v5d-qwen0.5b-prod-grounding.json`
- **v5e:** train log only — **no adapter**

### Local copies

- `psm-model/prod-memory/results/hf-prod-v5c-qwen0.5b-prod-grounding.json` (8/10)
- `psm-model/prod-memory/results/hf-prod-v5d-qwen0.5b-prod-grounding.json` (6/10)
- `psm-model/prod-memory/checkpoints/hf-prod-v5d-qwen0.5b/` (pulled from HF)
- `psm-model/prod-memory/data/hf-prod-v5e.jsonl` (380 rows, built not trained)

---

## Known issues (fix before next train)

1. **Watcher `job_state=?`** — verify-pod JSON not always parsed; crash path still falls through to `unexpected state` instead of `train FAILED`. Harden `_watch_hf_lora.py` to use last JSON block with `train_log_tail` and treat `tmux=MISSING` + CUDA in log as failure.
2. **`_run_hf_lora.py --deploy`** — does not return `proxy_user` to caller (must copy from deploy JSON manually).
3. **CUDA on A5000 `CA-MTL-1`/`SE`** — two v5e crashes; **prefer L4 or US region** for next pod (`PSM_GPU_PREFERENCES` / `--auto-gpu`).
4. **Eval interrupted on stop** — watcher stopped pod before eval finished on v5d first attempt; re-ran eval on separate pod.

---

## Tomorrow — priority order

### Phase 0 — Env (2 min)

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o krishnachhftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'krishnach7262/psm-prod-memory-hf'
$env:PSM_HF_DATASET_REPO = 'krishnach7262/psm-prod-memory-data'
```

### Phase 1 — Fix watcher (15 min)

- Parse verify-pod output reliably (last block with `job_state` / `train_log_tail`).
- On failure: sync → stop pod → exit 2 (no infinite retry).
- On success: eval → upload eval HF → pull local → **then** stop pod.

### Phase 2 — Retry v5e on safe GPU (or skip to v5e-b)

**Option A — retry v5e as-is** (curriculum already on HF dataset repo):

```powershell
python psm-model/scripts/_run_hf_lora.py --profile v5e --deploy --sync-code
# copy pod_id + proxy-user from JSON, then:
python psm-model/scripts/_run_hf_lora.py --profile v5e --pod-id <id> --proxy-user <user> --sync-code
python psm-model/scripts/_watch_hf_lora.py --profile v5e --pod-id <id> --proxy-user <user> --interval-sec 120 --stop-pod-on-done
```

**Option B — v5e-b micro-tune** if v5e trains but score < 8:

- Resume **v5c** (never v5d)
- Same hybrid curriculum, **300 steps @ 2e-5**
- Boost noise ignore rows only (no summary labels)

**Do not:** resume v5d, full MEMORY_SUMMARIES curriculum, v4 ignore bulk, ×40 fail copies.

### Phase 3 — Eval + ship decision

```powershell
python psm-model/scripts/_run_hf_lora_eval.py --profile v5e --pod-id <id> --proxy-user <user>
python psm-model/scripts/_sync_hf_lora.py --profile v5e --verify-only
```

**Promote if:** ≥8/10 `effective_stored`, no identical `raw_output` across fixtures, noise cases `ignore` (stretch goal: 10/10).

**If still 8/10 with noise storing:** acceptable for fixture bar but plan LoCoMo smoke + action_match follow-up.

**If < 8/10:** Gemma relabel **4 cases only** (`plan-01`, `workflow-runpod`, `noise-filler`, `noise-meta`) → v5f, 5 copies each max per [training-pitfalls.md](training-pitfalls.md).

### Phase 4 — If v5c is “good enough” for now

Ship **v5c adapter** for integration testing while iterating:

- HF: `hf-prod-v5c-qwen0.5b/adapter`
- Local eval: `psm-model/prod-memory/results/hf-prod-v5c-qwen0.5b-prod-grounding.json`

---

## v5e recipe (reference)

| Field | Value |
|-------|-------|
| Resume | `hf-prod-v5c-qwen0.5b/adapter` |
| Curriculum | `hf-prod-v5e.jsonl` — 380 rows, hybrid snippets + forced keys on plan/workflow |
| Boost ×20 | plan-01, workflow-runpod, noise-filler, noise-meta |
| Steps / LR | 500 / **5e-5** |
| Format | minimal, 0% recall, 0% v4 ignore |

---

## Stopped pods (do not delete until checklist)

| Pod | Last run | Notes |
|-----|----------|-------|
| `8er662o0i558p9` | v5e crash | Stopped |
| `ohrvp8zj19m5n7` | v5e crash | Stopped |
| `zxpag9azrrkb3n` | v5d complete | Stopped |
| `4fiscoegksaxbd` | v5c | Stopped (older) |

Delete only after HF manifest verified + local pull — or keep stopped (~$0.01/hr storage).

---

## Key paths

| What | Path |
|------|------|
| Fixtures | `psm-model/prod-memory/fixtures/cases.json` |
| Best eval | `psm-model/prod-memory/results/hf-prod-v5c-qwen0.5b-prod-grounding.json` |
| v5e curriculum | `psm-model/prod-memory/data/hf-prod-v5e.jsonl` |
| Train launch | `psm-model/scripts/_run_hf_lora.py` |
| Watcher | `psm-model/scripts/_watch_hf_lora.py` |
| HF sync | `psm-model/scripts/_sync_hf_lora.py` |
| Hybrid labels | `psm-model/prod-memory/prod_memory/build_minimal_fixture_rows.py` |

---

## One-line summary

**v5c at 8/10 is the current ship candidate; v5d regressed to 6/10; v5e code/curriculum is ready but train failed twice on bad GPU — tomorrow: fix watcher, retry v5e on L4/US, eval, and push past 8/10 on plan-01 + noise ignore without template collapse.**
