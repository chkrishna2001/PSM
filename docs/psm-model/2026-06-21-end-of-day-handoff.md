# HF LoRA prod-memory v2 — end of day handoff (2026-06-21)

**Read first next session:** this file → `.cursor/skills/runpod-gpu-train/SKILL.md` → `.cursor/rules/runpod-auto-delete.mdc` → [training-playbook.md](training-playbook.md)

**Nothing running on RunPod.** Pod `jf5j5htrfqkyc1` **deleted** after HF upload verified. **No GPU billing.**

**Prod eval: NOT run yet.** Ship bar still **unmet** until tomorrow's prod fixture eval.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **HF LoRA v2 train** | **Done** — 2400 steps, final loss **0.094**, ~2.2h runtime |
| **Adapter on HF** | `krishnach7262/psm-prod-memory-hf/hf-prod-v2-qwen0.5b/*` (42 files) |
| **Train log on HF** | `logs/hf-prod-v2-qwen0.5b-train.log` |
| **Train metrics local** | `psm-model/prod-memory/checkpoints/hf-prod-v2-qwen0.5b/train.metrics.json` |
| **Train log local** | `psm-model/prod-memory/results/hf-prod-v2-qwen0.5b-train.log` |
| **Prod eval (10 fixtures)** | **Pending** — do tomorrow |
| **v1 baseline** | 0/10 `effective_stored` (template bleed; adapter at `hf-prod-v1-qwen0.5b/`) |
| **Ship bar** | **≥8/10 (85%) `effective_stored`** on `fixtures/cases.json` |
| **RunPod pod** | **None** (deleted 2026-06-21 after upload) |

---

## v2 training summary

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Curriculum | `hf-prod-v2.jsonl` — **2289 rows** (v3 + v5, 25× fixture copies, 12% recall) |
| Steps | 2400 |
| max_length | 3072 |
| batch / grad_accum | 1 / 8 |
| save_steps | 400 → checkpoints 400, 800, 1200, 1600, 2000, 2400 on HF |
| Final adapter | `hf-prod-v2-qwen0.5b/adapter/` |
| train_loss | **0.0936** (vs v1 ~0.63) |

### v2 fixes (vs failed v1)

1. **Tokenization** — trim prompt prefix; never truncate assistant labels (`hf_lora_train.py`)
2. **Curriculum** — prod-shaped tagged labels from v3 (≥500 chars) + v5 (≥80 chars), heavy fixture oversampling
3. **Parse** — strip whitespace on `A:` / `T:` lines (`lean_format.py`)

### Watch item for eval

`train.metrics.json` still shows `input_ids_p50/p90/max: 3072` — most rows hit max length. v2 fix preserves labels when possible, but **prod eval is the real test** (v1 failed with identical template on all 10 fixtures).

---

## HF repos & tokens

| Env var | Source | Used for |
|---------|--------|----------|
| `HF_TOKEN` | `o krishnachhftoken` | Model repo **`krishnach7262/psm-prod-memory-hf`** |
| `DATASET_HF_TOKEN` | `~/.cache/huggingface/token` | Dataset repo **`krishnach7262/psm-prod-memory-data`** |

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o krishnachhftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
```

---

## Tomorrow — step-by-step

### 1. Prod fixture eval (no retrain)

Eval uses **10 cases** in `psm-model/prod-memory/fixtures/cases.json`. Adapter is pulled from HF — **no pod required for train**, but eval needs a GPU pod.

**Option A — deploy fresh eval pod (recommended):**

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o krishnachhftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()

python psm-model/scripts/_run_hf_lora_eval.py --deploy --profile v2
# Note pod_id + proxy_user from JSON output, then if deploy-only created pod:
python psm-model/scripts/_run_hf_lora_eval.py --pod-id <id> --proxy-user <user> --profile v2
```

This runs eval on pod, uploads JSON to HF, pulls locally to:
`psm-model/prod-memory/results/hf-prod-v2-qwen0.5b-prod-grounding.json`

**Option B — warm pod if one exists:**

```powershell
python psm-model/scripts/_run_hf_lora_eval.py --pod-id <id> --proxy-user <user> --profile v2
```

**Pull-only** (if eval already ran on pod/HF):

```powershell
python psm-model/scripts/_run_hf_lora_eval.py --pull-only --profile v2
```

### 2. Read results

```powershell
python -c "import json; d=json.load(open('psm-model/prod-memory/results/hf-prod-v2-qwen0.5b-prod-grounding.json')); print(json.dumps(d['aggregate'], indent=2))"
```

Primary metric: **`effective_stored`**. Gate: **≥8/10**.

Also check per-case `parse_valid`, `action_match`, `raw_output` — v1 showed identical broken template on all cases.

### 3. Decide next action

| Eval result | Action |
|-------------|--------|
| **≥8/10 effective_stored** | Promote v2; document in session-log; consider LoCoMo prep (still deferred until parse ≥95%) |
| **3–7/10** | Failure mining on misses; tune curriculum (recall mix, fixture copies, max_length) or v3 retrain |
| **0–2/10** (template bleed again) | Inspect `raw_output` + tokenization stats; may need longer context strategy or smaller prompt prefix |

### 4. Pod cleanup after eval

- **Verify** eval JSON local **and** on HF: `eval/hf-prod-v2-qwen0.5b-prod-grounding.json`
- **Stop or delete** eval pod only after artifacts verified (see `runpod-auto-delete.mdc`)

---

## Key paths

| Artifact | Location |
|----------|----------|
| Prod fixtures | `psm-model/prod-memory/fixtures/cases.json` |
| v2 curriculum | `psm-model/prod-memory/data/hf-prod-v2.jsonl` (HF dataset repo) |
| v2 adapter (HF) | `krishnach7262/psm-prod-memory-hf` → `hf-prod-v2-qwen0.5b/adapter/` |
| v2 eval (pending) | `psm-model/prod-memory/results/hf-prod-v2-qwen0.5b-prod-grounding.json` |
| v1 eval (baseline) | `psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json` |
| Upload script | `psm-model/scripts/_hf_lora_upload_all.py` |
| Eval script | `psm-model/scripts/_run_hf_lora_eval.py --profile v2` |

---

## Scripts added/updated this session

| Script | Purpose |
|--------|---------|
| `_hf_lora_upload_all.py` | Upload full v2 checkpoint tree + train log to HF; verify; pull metrics |
| `_run_hf_lora_eval.py` | `--profile v2` for eval paths |
| `_watch_hf_lora.py` | Background train watcher (was running; pod now deleted) |
| `_probe_train_progress.py` | Quick epoch/step probe |

---

## Do NOT repeat

- Do **not** delete pods before adapter + metrics + (when run) eval JSON are on HF and pulled locally.
- Do **not** use v1 eval numbers for v2 — v1 was 0/10 with template collapse.
- Do **not** promote 50M TinyDecoder for generative extract (binary classify only).

---

## Quick verify HF (optional)

```powershell
python psm-model/scripts/_hf_lora_upload_all.py --verify-only
```

Expected: all four required paths present (`adapter_config`, `adapter_model.safetensors`, `train.metrics.json`, train log).
