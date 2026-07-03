# LoCoMo checkpoint mistake + retest plan — handoff (2026-07-02)

**Read first next session:** this file → `psm-model/scripts/probe_locomo_hf_storage_cpu.py` → `psm-model/prod-memory/data/prod-teacher-cache-v4-4o.jsonl`

**Goal:** Stop retraining blind. **Probe the right checkpoints** on LoCoMo turns (send remember-shaped input → read raw model output), **then** decide train vs deploy.

---

## What we did wrong

### 1. Tested the v5k shortcut, not the big-data model

LoCoMo ingest and CPU probes used the **two-pass prod shortcut**:

| Role | Checkpoint used | Curriculum | Rows | Format |
|------|-----------------|------------|------|--------|
| Gate | `hf-prod-v5k-gate-distill-qwen0.5b` | `hf-prod-v5k-gate-distill.jsonl` | 788 | binary `ignore`/`store` |
| Extract | `hf-prod-v5k-extract-qwen0.5b` | `hf-prod-v5k-extract.jsonl` | **240** | **`minimal_extract`** |

Wired in `src/psm-core/src/config.ts` and [2026-06-26 handoff](2026-06-26-end-of-day-handoff.md).

**This is not the model trained on Codex/ChatGPT/Gemini teacher data.**

### 2. Confused “no curriculum” with “wrong curriculum deployed”

Large labeled data **does exist**:

| Artifact | Rows | Facts | Source |
|----------|------|-------|--------|
| `prod-teacher-cache-v4-4o.jsonl` | 1,474 | **78%** | Codex 725, ChatGPT 495, Gemini 254 (GPT-4o teacher) |
| `prod-extraction-v3.jsonl` | 1,504 storage | **95%** | Built from teacher cache + fixtures |
| `hf-prod-v2.jsonl` | 2,289 | 1,346 | tagged, from v3 |
| `hf-prod-v5h.jsonl` | 1,664 | 1,499 | **JSON**, from v3 |

We never LoCoMo-probed **`hf-prod-v5h`** or **`hf-prod-v2`** — the checkpoints that actually ate v3/teacher data.

### 3. `minimal_extract` cannot answer full-PSM questions

`minimal_extract` training target is one line only:

```text
store: <one grounded sentence>
```

No `F:` facts, no `TE:`/`RT:` temporal, no episodic vs semantic in output. Probing it for “does the model do facts/temporal?” was the wrong test by design.

### 4. Gate eval ≠ gate on LoCoMo

- **10/10** on fixtures ([2026-06-26](2026-06-26-end-of-day-handoff.md)) = parser fix (`store_episodic` counts as store), same adapter.
- **8/10** on fixtures before fix ([2026-06-25](2026-06-25-end-of-day-handoff.md)) = fails `noise-filler`, `noise-meta`.
- **120-turn LoCoMo gate validation** (`validate_locomo_gate_cpu.py`): distill/fix **always store**, dpo **always ignore** — **0% useful discrimination** on conversation-shaped input.

### 5. Ran full LoCoMo ingest before validating write-path on right checkpoint

~10h GPU ingest → DB with episodic only, no facts, no embeddings — then discovered format/checkpoint mismatch. Should have run **CPU probe on v5h first**.

---

## What we already probed (v5k shortcut — keep for reference)

Scripts: `psm-model/scripts/probe_locomo_hf_storage_cpu.py`, `validate_locomo_gate_cpu.py`  
Results: `benchmark/locomo/results/probe-locomo-hf-storage-*.jsonl`

| Probe | Checkpoint | Result (10 curated turns) |
|-------|------------|---------------------------|
| `minimal_extract` | v5k-extract | 100% parse, grounded one-liners, **0% facts** |
| two-pass distill+extract | v5k gate + extract | Gate always open; same as extract-only |
| tagged | v5k-extract | 0% parse (wrong format for adapter) |
| tagged | v5b | 100% parse, **30% store match**, 0% facts |
| gate validation | distill/fix/dpo | **50% accuracy** (chance) on 120 balanced turns |

**Conclusion from v5k probes:** Shortcut path works for quoting utterances; **not** a test of teacher-trained full storage.

---

## What to test now (direct probe — same style as before)

**Method:** LoCoMo remember-shaped input (`build_locomo_remember_text`), save prompt + raw output + parsed decision. **No ingest, no DB.**

### Device policy

| Where | Device | Notes |
|-------|--------|-------|
| **Local (Windows)** | `--device cpu` + `PSM_FORCE_CPU=1` | No local CUDA — quick checkpoint triage only (~4 min) |
| **RunPod pod** | `--device cuda` + `PSM_FORCE_CPU=0` | Use GPU on pod; same script, faster inference |

Script name `probe_locomo_hf_storage_cpu.py` reflects the **local default**, not a pod requirement.

**Expand cases after first pass:** use `validate_locomo_gate_cpu.py` pattern (QA-evidence = should-store) or `--per-conv-pos 6` on full 10 convs.

### Priority 1 — **Primary candidate** (trained on v3 / teacher data)

| Profile | Adapter (local) | Train curriculum | Format | Why first |
|---------|-----------------|------------------|--------|-----------|
| **v5h** | `psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter` | `hf-prod-v5h.jsonl` (1,664 rows, **90% facts**) | **json** | Closest deployable match to full `StorageDecision` from `prod-extraction-v3` |

```powershell
cd C:\Users\chkri\source\repos\PSM
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory;psm-model\scripts"
$env:PSM_FORCE_CPU = "1"
.venv\Scripts\python.exe psm-model\scripts\probe_locomo_hf_storage_cpu.py `
  --device cpu `
  --output-format json `
  --adapter-dir psm-model\prod-memory\checkpoints\hf-prod-v5h-qwen0.5b\adapter `
  --out benchmark\locomo\results\probe-locomo-hf-storage-v5h-json-cpu.jsonl
```

**Pass bar (single-pass storage):** parse_ok ≥90%, has_facts >0 on evidence turns, store/ignore reasonable on curated set, memory content grounded.

### Priority 2 — Tagged format (if JSON good)

| Profile | Adapter | Curriculum | Format |
|---------|---------|------------|--------|
| **v5b** | `hf-prod-v5b-qwen0.5b/adapter` | 195 rows, tagged, fixture-heavy | **tagged** |

```powershell
.venv\Scripts\python.exe psm-model\scripts\probe_locomo_hf_storage_cpu.py `
  --device cpu --output-format tagged `
  --adapter-dir psm-model\prod-memory\checkpoints\hf-prod-v5b-qwen0.5b\adapter `
  --out benchmark\locomo\results\probe-locomo-hf-storage-v5b-tagged-cpu.jsonl
```

Already ran once (30% store match). Re-run after v5h baseline for comparison.

### Priority 3 — Pull and probe v2 (best tagged curriculum, not local)

| Profile | Adapter | Curriculum | Format |
|---------|---------|------------|--------|
| **v2** | Pull from HF `hf-prod-v2-qwen0.5b` | 2,289 rows, **1,346 facts**, tagged | **tagged** |

```powershell
o krishnachhftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
python psm-model\scripts\_sync_hf_lora.py --profile v2 --pull-only
```

Then probe with `--output-format tagged`. **v2 is the largest tagged run on v3 data.**

### Priority 4 — Optional comparisons

| Profile | Format | Notes |
|---------|--------|-------|
| `hf-prod-v5i-qwen0.5b` | minimal | 1,789 rows from v3, has ignore rows — single-pass store+ignore |
| `hf-prod-v5j-qwen0.5b` | minimal | More ignore-heavy (40% target) |
| v5k two-pass | minimal_extract | **Baseline we already have** — do not re-run unless regressing |

### Do **not** use as primary LoCoMo storage eval

- `hf-prod-v5k-extract-qwen0.5b` — 240-row shortcut only
- `hf-prod-v5k-gate-*` alone — binary only; failed on LoCoMo-shaped input unless paired with a real storage adapter

---

## Gate (separate probe, after storage pick)

Only worth fixing gate **after** storage adapter is chosen.

If staying two-pass: probe gate with `validate_locomo_gate_cpu.py` on candidates:

```powershell
.venv\Scripts\python.exe psm-model\scripts\validate_locomo_gate_cpu.py `
  --gate-adapter psm-model\prod-memory\checkpoints\hf-prod-v5k-gate-distill-qwen0.5b\adapter `
  --per-conv-pos 6 --per-conv-neg 6 `
  --out benchmark\locomo\results\validate-locomo-gate-distill-cpu.jsonl
```

**Or** single-pass storage (v5h) where model outputs `ignore` in the JSON/tagged decision — may make separate gate unnecessary.

---

## Decision tree (after probes)

```
Run v5h JSON probe on LoCoMo turns
        │
        ├─ parse_ok ≥90% AND facts on evidence turns AND grounded content
        │     → DEPLOY v5h (or best step checkpoint-400) for LoCoMo re-ingest
        │     → Skip v5k minimal_extract path
        │     → Train only if: temporal missing (expected — teacher cache 0% temporal)
        │
        ├─ parse_ok high but weak facts / bad store-ignore
        │     → Compare v5i (minimal+v3) and v2 (tagged) probes
        │     → Train: tagged/json from prod-extraction-v3 or teacher cache (NOT minimal_extract)
        │
        └─ parse_ok low / garbage on conversations
              → Train fresh from prod-teacher-cache-v4-4o.jsonl → tagged/json curriculum
              → Do NOT train minimal_extract again
```

### Train only if probes show:

| Gap | Train target | Data source |
|-----|--------------|-------------|
| No temporal on conversations | Add `TE:`/`RT:` to teacher prompts + curriculum | v4-4o cache **re-label** or LoCoMo-shaped teacher pass |
| Gate still needed + broken | Gate on session-shaped rows | `prod-extraction-v3` ignore/store labels, **not** fixtures-only |
| v5h good on Codex-like text, bad on chat | Add conversation diversity to curriculum | Same teacher cache + optional LoCoMo evidence rows as **slice**, not benchmark overfit |

### Do **not** train (yet):

- Another `v5k-extract` / `minimal_extract` run
- LoCoMo-specific checkpoint “for the benchmark”
- Gate DPO full 80-step (proved 2/10 — [2026-06-25 handoff](2026-06-25-end-of-day-handoff.md))

---

## Key file map

| Purpose | Path |
|---------|------|
| Teacher labels (ground truth) | `psm-model/prod-memory/data/prod-teacher-cache-v4-4o.jsonl` |
| Training mix from teacher | `psm-model/prod-memory/data/prod-extraction-v3.jsonl` |
| v5h HF curriculum | `psm-model/prod-memory/data/hf-prod-v5h.jsonl` |
| v5k extract (shortcut — wrong for eval) | `psm-model/prod-memory/data/hf-prod-v5k-extract.jsonl` |
| CPU write probe | `psm-model/scripts/probe_locomo_hf_storage_cpu.py` |
| CPU gate validation | `psm-model/scripts/validate_locomo_gate_cpu.py` |
| Prior v5k probe results | `benchmark/locomo/results/probe-locomo-hf-storage-*.jsonl` |
| LoCoMo ingest DB (v5k shortcut) | `benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db` |
| CLI default adapters (wrong for full PSM) | `src/psm-core/src/config.ts` |

---

## One-liner state

**We LoCoMo-tested the 240-row `minimal_extract` shortcut (v5k), not the 1,500-row GPT-4o teacher model (v3/v5h). Next: CPU-probe `hf-prod-v5h` JSON on LoCoMo turns → if good, re-ingest; if not, train tagged/json from teacher cache — not another minimal_extract.**

---

## Next session checklist

1. [ ] Run **v5h JSON** probe (command above)
2. [ ] Read `probe-locomo-hf-storage-v5h-json-cpu.jsonl` — check `raw_output`, `facts`, `action` on `conv-26:D1:3` (temporal evidence)
3. [ ] If v5h weak: pull **v2** from HF, probe `--output-format tagged`
4. [ ] **Decision:** deploy existing vs train — use decision tree above
5. [ ] Only then: fresh LoCoMo ingest with chosen adapter (not v5k-extract)
6. [ ] Update `config.ts` defaults when checkpoint is chosen
