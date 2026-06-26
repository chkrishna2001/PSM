# HF LoRA v5k binary gate — end of day handoff (2026-06-25)

**Read first next session:** this file → `.cursor/skills/runpod-gpu-train/SKILL.md` → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on RunPod.** All pods **EXITED** (stopped). **No GPU billing.**

**Ship bar (gate):** **≥9/10 `classify_match`** on `psm-model/prod-memory/fixtures/cases.json` (10 prod fixtures, binary `ignore`/`store`, model-only — no rule pre-filters). **Not met** — best is **8/10**.

**Blocked:** `v5k-extract` LoRA train until gate ≥9/10.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Best gate run** | **`hf-prod-v5k-gate-distill-qwen0.5b`** — **8/10** `classify_match` |
| **Latest gate run** | **v5k-gate-dpo** — **2/10** (regressed; ignore on everything) |
| **Base Qwen 0.5B (no LoRA)** | **8/10** — same 2 noise failures |
| **Teacher ceiling** | Gemma 4 31b **10/10** on binary-gate pilot |
| **Model HF repo** | `krishnach7262/psm-prod-memory-hf` |
| **Dataset HF repo** | `krishnach7262/psm-prod-memory-data` |
| **HF token** | `o krishnachhftoken` → clipboard (`HF_TOKEN`) |

### Gate eval score history (`classify_match` on 10 fixtures)

| Profile | Curriculum | Train steps | Score | Failure mode |
|---------|------------|-------------|-------|--------------|
| v5k-gate | 404 rows | ~80 | **8/10** | `store` on both noise cases |
| v5k-gate-fix | 280 rows, fixture-only, 60× ignore dup | 150 | **8/10** | same |
| **v5k-gate-distill** | 788 rows, Gemma 4 31b labels + 30 noise variants | 120 | **8/10** | same — **best ship candidate** |
| v5k-gate-dpo | 880 preference pairs | 80 | **2/10** | `ignore` on all 10 (only noise correct) |
| base (no LoRA) | — | — | **8/10** | checkpoint sweep — LoRA not root cause |

### Per-fixture pattern (all 8/10 runs)

| Case | expect | model `raw_output` | classify_match |
|------|--------|-------------------|----------------|
| plan-01-handoff | store | store | ✓ |
| plan-02-chunking | store | store | ✓ |
| cursor-01-summary | store | store | ✓ |
| cursor-02-debug | store | store | ✓ |
| workflow-review-pr | store | store | ✓ |
| workflow-runpod | store | store | ✓ |
| technical-eslint | store | store | ✓ |
| technical-api | store | store | ✓ |
| **noise-filler** | **ignore** | **store** | ✗ |
| **noise-meta** | **ignore** | **store** | ✗ |

**noise-filler:** `"Okay, sure. Let me know if you need anything else."`  
**noise-meta:** `"I don't have any durable facts to store from this message..."` (explicitly says no facts; still gets `store`)

**DPO inversion (2/10):** all 8 store cases → `ignore`; both noise → `ignore` ✓.

---

## Architecture: two-pass remember()

1. **Gate (binary)** — Qwen 0.5B LoRA → `ignore` or `store` only.
2. **Extract** — separate LoRA (`v5k-extract`) → `minimal_extract` format when gate says store.

Eval metric for gate: `classify_match` in `eval_hf_grounding.py` (`output_format=binary`). Greedy decode, `max_new_tokens=384` (16 would suffice; inherited default).

---

## RunPod pods (all stopped)

| Pod ID | Profile | Status | Notes |
|--------|---------|--------|-------|
| `kvwxuy0y0uec0y` | v5k-gate-distill | EXITED | Train + eval done; watcher exited rc=1 (SSH after stop) |
| `ane3yrksku83ch` | v5k-gate-dpo | EXITED | Train 80 steps OK; watcher completed eval + stop |

Proxy users: `{pod_id}-64410f25` (distill), `{pod_id}-64410f25` (dpo) — re-fetch from dashboard if reconnecting.

---

## Artifacts

### HF (`krishnach7262/psm-prod-memory-hf`) — verified

**v5k-gate-distill:**
- `hf-prod-v5k-gate-distill-qwen0.5b/adapter/*`
- `checkpoint-40`, `checkpoint-80`, `checkpoint-120` (if synced)
- `eval/hf-prod-v5k-gate-distill-qwen0.5b-classify.json`
- `logs/hf-prod-v5k-gate-distill-qwen0.5b-train.log`

**v5k-gate-dpo:**
- `hf-prod-v5k-gate-dpo-qwen0.5b/adapter/*`
- `checkpoint-40`, `checkpoint-80`
- `eval/hf-prod-v5k-gate-dpo-qwen0.5b-classify.json`
- `logs/hf-prod-v5k-gate-dpo-qwen0.5b-train.log`

Verify: `python psm-model/scripts/_sync_hf_lora.py --profile v5k-gate-distill --verify-only`

### HF dataset (`krishnach7262/psm-prod-memory-data`)

- `prod-memory/hf-prod-v5k-gate.jsonl`
- `prod-memory/hf-prod-v5k-gate-fix.jsonl`
- `prod-memory/hf-prod-v5k-gate-distill.jsonl`
- `prod-memory/hf-prod-v5k-gate-dpo.jsonl` (880 pairs)
- `prod-memory/hf-prod-v5k-extract.jsonl` (built, not trained)

### Local paths

| Artifact | Path |
|----------|------|
| Distill eval | `psm-model/prod-memory/results/hf-prod-v5k-gate-distill-qwen0.5b-classify.json` |
| DPO eval | `psm-model/prod-memory/results/hf-prod-v5k-gate-dpo-qwen0.5b-classify.json` |
| Checkpoint sweep | `psm-model/prod-memory/data/gate-distill-checkpoint-sweep.json` |
| Distill label cache | `psm-model/prod-memory/data/v5k-gate-distill-cache.json` |
| Fixtures | `psm-model/prod-memory/fixtures/cases.json` |

**Local gap:** distill `adapter/` may be HF-cache metadata only under `checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/` — pull from HF if needed:
`python psm-model/scripts/_sync_hf_lora.py --profile v5k-gate-distill --pull-only`

---

## Teacher / label quality

| Source | Score | Notes |
|--------|-------|-------|
| `pilot_binary_gate_fixtures.py` + Gemma 4 31b | **10/10** | Best teacher for binary gate |
| Claude (binary pilot) | 9/10 | |
| Gemma 3 27b / student | 8/10 | |
| GPT-4o | 4/10 | Inverted — ignores stores |
| `compare_teacher_models.py` JSON path | Gemma 4 **8/10** | Fails `workflow-runpod`, `noise-meta` on full JSON teacher |

Distillation from Gemma 4 labels did **not** lift student above base 8/10.

---

## What we did today

### Training runs

1. **v5k-gate-distill** — pod `kvwxuy0y0uec0y` — 120 steps, 788 rows → **8/10**, artifacts on HF, pod stopped.
2. **v5k-gate-dpo** — pod `ane3yrksku83ch`:
   - First attempt: **trl import crash** (`FSDPModule` / `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES` vs torch 2.4 + transformers on pod).
   - Fix: **inline DPO trainer** in `hf_lora_train.py` (no `trl` dependency).
   - Second attempt: 80 steps completed, loss 0.98 → 0.24 (step 40) → eval **2/10**, pod stopped.

### Analysis

- **`sweep_gate_checkpoints.py`** on distill dir + base: every checkpoint **8/10**; base also **8/10** → noise failure is **pre-LoRA prior**, not insufficient SFT.
- Full DPO **over-corrects** (store-all → ignore-all).
- Eval user wrapper still says *"Extract durable memory… Choose ignore, store_episodic…"* while system prompt says binary `ignore`/`store` — may reinforce store bias.

### Code added/changed (mostly uncommitted)

| Area | Files |
|------|-------|
| Binary teacher | `prod_memory/binary_gate_teacher.py` |
| DPO curriculum | `build_binary_fixture_rows.py` → `build_v5k_gate_dpo_rows()`; `hf-prod-v5k-gate-dpo.jsonl` |
| Distill labels | `scripts/build_v5k_gate_distill_labels.py` |
| Pilots | `consult_binary_gate.py`, `pilot_teacher_fixtures.py`, `pilot_binary_gate_fixtures.py`, `pilot_hf_gate_classify.py` |
| Sweep | `scripts/sweep_gate_checkpoints.py` |
| DPO train | `src/psm_model/hf_lora_train.py` — `train_hf_lora_dpo()`, `_DpoTrainer`, `_completion_logprob` (no trl) |
| RunPod | `runpod_hf_lora_train.sh` — `HF_TRAIN_MODE=dpo`, `HF_DPO_BETA`; removed broken `pip install trl` |
| Launchers | `_run_hf_lora.py`, `_run_hf_lora_eval.py`, `_watch_hf_lora.py`, `_sync_hf_lora.py` — `v5k-gate-dpo` profile |
| Deps | `pyproject.toml` — `trl` in optional hf deps (unused on pod after inline DPO) |

---

## Next session — prioritized

### 1. Logprob gate decision (cheapest, ~30 min CPU)

Instead of greedy first-token decode, compare `log P(store|prompt)` vs `log P(ignore|prompt)` on distill adapter + base. Still model-only. May fix noise without retrain.

No script yet — add to `eval_hf_grounding.py` or small `pilot_logprob_gate.py`.

### 2. Micro-DPO from distill adapter

- Resume from `hf-prod-v5k-gate-distill-qwen0.5b/adapter`
- **10–20 steps**, **β=0.05**, noise-only preference pairs (or 880 pairs but tiny step count)
- **Do not** repeat full 80-step β=0.2 run
- Needs: `resume_from_checkpoint` or `HF_RESUME_ADAPTER` wired in `train_hf_lora_dpo()`

### 3. Larger gate model (1.5B)

If logprob + micro-DPO fail: `Qwen2.5-1.5B-Instruct` LoRA gate profile. Higher VRAM; same fixtures bar.

### 4. Prompt alignment

Train and eval share extraction-style user wrapper; consider gate-only user message for binary (`Assistant response:\n{text}`) in both curriculum and eval.

### 5. v5k-extract (after gate ≥9/10)

Profile exists in `_run_hf_lora.py`; curriculum `hf-prod-v5k-extract.jsonl` on HF dataset. Two-pass eval via `eval_hf_two_pass.py`.

### Probably skip

- More distill rows / noise duplication alone
- Full 80-step DPO at β=0.2 from scratch
- `trl` on RunPod without version pinning (use inline DPO)

---

## Commands (copy-paste)

```powershell
cd C:\Users\chkri\source\repos\PSM
o krishnachhftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'krishnach7262/psm-prod-memory-hf'
$env:PSM_HF_DATASET_REPO = 'krishnach7262/psm-prod-memory-data'
```

**Verify HF artifacts:**
```powershell
python psm-model/scripts/_sync_hf_lora.py --profile v5k-gate-distill --verify-only
python psm-model/scripts/_sync_hf_lora.py --profile v5k-gate-dpo --verify-only
```

**Pull local from HF:**
```powershell
python psm-model/scripts/_sync_hf_lora.py --profile v5k-gate-distill --pull-only
```

**Checkpoint sweep (CPU/CUDA local):**
```powershell
python psm-model/prod-memory/scripts/sweep_gate_checkpoints.py `
  --run-dir psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b `
  --include-base --device cpu `
  --out psm-model/prod-memory/data/gate-distill-checkpoint-sweep.json
```

**Eval on pod (warm):**
```powershell
python psm-model/scripts/_run_hf_lora_eval.py --profile v5k-gate-distill `
  --pod-id <id> --proxy-user <id>-<suffix>
```

**Train new gate run (deploy):**
```powershell
python psm-model/scripts/_run_hf_lora.py --profile v5k-gate-distill --deploy --sync-code
python psm-model/scripts/_watch_hf_lora.py --profile v5k-gate-distill `
  --pod-id <id> --proxy-user <user> --stop-pod-on-done
```

**Stop pod (positional id):**
```powershell
python psm-model/scripts/runpod_ctl.py stop-pod <pod_id>
```

---

## Key hyperparameters (gate profiles)

| Profile | steps | lr | save_steps | mode | notes |
|---------|-------|-----|------------|------|-------|
| v5k-gate | 80 | 2e-4 | 40 | sft | |
| v5k-gate-fix | 150 | 2e-4 | 40 | sft | |
| v5k-gate-distill | 120 | 2e-4 | 40 | sft | **best** |
| v5k-gate-dpo | 80 | 5e-6 | 40 | dpo | β=0.2 — too aggressive |

---

## Conversation / agent context

- Prior transcript: agent session `f89ef8f6-4ac3-4a4c-a02c-21b2619531bd`
- Gate goal articulated as ≥9/10 before extract; stuck at 8/10 across SFT variants; DPO proved objective can move model but regressed
