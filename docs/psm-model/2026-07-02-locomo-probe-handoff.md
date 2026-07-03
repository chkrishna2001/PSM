# LoCoMo probe handoff — wrong checkpoint, what to test next (2026-07-02)

**Read first next session:** this file → `psm-model/scripts/probe_locomo_hf_storage_cpu.py` → `psm-model/prod-memory/data/prod-teacher-cache-v4-4o.jsonl`

**Purpose:** Document what we tested wrong, what direct model probes to run now on LoCoMo turns, and how to decide training — without another blind GPU loop.

---

## Executive summary

We ran LoCoMo ingest and CPU write-path probes using the **v5k two-pass shortcut** (`gate-distill` + `minimal_extract`). That path was **never trained on the big teacher-labeled data** (Codex/ChatGPT/Gemini → `prod-extraction-v3` / `prod-teacher-cache-v4-4o.jsonl`).

**LoCoMo results therefore measure the shortcut, not whether the full PSM storage model is good.**

Before any new training: **direct-probe the right checkpoints** on the same LoCoMo turns (send remember-shaped input → read raw model output → parse).

---

## What we did wrong

### 1. Tested the wrong checkpoint family for the stated goal

| What we used (LoCoMo + probes) | What it was trained for |
|-------------------------------|-------------------------|
| `hf-prod-v5k-gate-distill-qwen0.5b` | Binary gate on **788 fixture rows** (plan/cursor/workflow/noise) |
| `hf-prod-v5k-extract-qwen0.5b` | **`minimal_extract`** — **240 rows**, one-line `store: <sentence>` only |

Wired in: `src/psm-core/src/config.ts` defaults, `benchmark/locomo/src/ingest-psm-model.ts`, 6/26 handoff.

### 2. Ignored the big labeled data we already built

| Artifact | Rows | Facts | Source |
|----------|------|-------|--------|
| `prod-teacher-cache-v4-4o.jsonl` | 1,474 | **78%** (GPT-4o teacher) | Codex 725, ChatGPT 495, Gemini 254 |
| `prod-extraction-v3.jsonl` | 1,504 storage | **95%** | Built from teacher cache + fixtures |
| `hf-prod-v2.jsonl` | 2,289 | 1,346 tagged | From v3 |
| `hf-prod-v5h.jsonl` | 1,664 | 1,499 JSON | From v3 |

**LoCoMo never used v2/v5h/v3-trained adapters.**

### 3. Confused `minimal_extract` with full PSM storage

- **`minimal_extract`:** gate already said store → model outputs `store: <one grounded sentence>`. No facts, no temporal, no episodic/semantic tags. Parser sets `facts: []`.
- **Full PSM (tagged/JSON):** `A:`, `T:`, `C:`, `F:`, `TE:`, `RT:`, indexables — what teacher cache labels contain.

We probed tagged format with **v5k-extract** (trained minimal only) → 0% parse. That was a format mismatch, not proof the big model failed.

### 4. Gate eval overstated quality

- **6/25 handoff:** `gate-distill` = **8/10** on 10 fixtures (fails `noise-filler`, `noise-meta`).
- **6/26 handoff:** **10/10** after **parser fix** (`store_episodic` counts as `store` in `eval_classify.py`) — same adapter, not new training.
- **LoCoMo gate validation (120 turns):** all three gates at **50% accuracy** (chance) — always-store or always-ignore on conversation-shaped input.

### 5. LoCoMo ingest DB incomplete by design

Ingested DB (`locomo-hf-prod-v5k-two-pass-nfull.db`): episodic only, **0 embeddings, 0 facts** — expected for `minimal_extract` path, not a DB bug.

---

## What we already ran (CPU direct probes)

Script: `psm-model/scripts/probe_locomo_hf_storage_cpu.py`  
Data: `benchmark/locomo/data/locomo10.json` — 10 hand-picked turns + 3 ignore cases  
Results: `benchmark/locomo/results/probe-locomo-hf-storage-*.jsonl`

| Probe | Adapter | Format | parse_ok | store_match | facts | Notes |
|-------|---------|--------|----------|-------------|-------|-------|
| v5k-extract | `hf-prod-v5k-extract-qwen0.5b` | tagged | 0% | 30% | 0% | Wrong format for adapter |
| base Qwen | none | tagged | 0% | 30% | 0% | |
| v5b | `hf-prod-v5b-qwen0.5b` | tagged | **100%** | 30% | 0% | Parses; poor store decisions on conv turns |
| minimal_extract | v5k-extract | minimal_extract | **100%** | 70% | 0% | Quotes utterances; stores everything |
| two-pass distill | gate-distill + extract | minimal_extract | 100% | 70% | 0% | Gate always opens (`store_episodic`) |
| two-pass gate-fix | gate-fix + extract | minimal_extract | 100% | 70% | 0% | Gate outputs `store`, still opens on greetings |
| two-pass gate-dpo | gate-dpo + extract | minimal_extract | 100% | 30% | 0% | Gate ignores everything |

Gate-only validation (120 balanced turns, QA-evidence labels):  
`psm-model/scripts/validate_locomo_gate_cpu.py` → `benchmark/locomo/results/validate-locomo-gate-{distill,fix,dpo}-cpu.jsonl`

Built but **not yet trained:** `hf-prod-v5l-gate.jsonl` (10,029 rows, LoCoMo evidence as conversational diversity in gate curriculum) — defer until we know which storage checkpoint to pair with.

---

## What checkpoint to direct-probe on LoCoMo NOW

**Goal:** Same as earlier probes — build LoCoMo remember text, send to model, save raw output + parsed decision. **No DB ingest, no OpenRouter answer eval yet.**

### Priority order (single-pass, full storage format)

| Priority | Checkpoint | Adapter path | `--output-format` | Why |
|----------|------------|--------------|---------------------|-----|
| **1** | **v5h** | `psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter` | `json` | Trained on **prod-extraction-v3** (1,664 rows, **90% facts**). Best match to teacher cache. |
| **2** | **v2** | `psm-model/prod-memory/checkpoints/hf-prod-v2-qwen0.5b/adapter` | `tagged` | Same v3 source, tagged format, 1,346 rows with facts. Pull from HF if missing locally. |
| **3** | **v5b** | `psm-model/prod-memory/checkpoints/hf-prod-v5b-qwen0.5b/adapter` | `tagged` | Only 195 rows (fixture-heavy). Already probed — weak on conv turns. Baseline only. |

**Do NOT use for full-spec eval:** `hf-prod-v5k-extract-qwen0.5b` (`minimal_extract`, 240 rows).

**Gate (separate):** After storage probe picks a winner, re-run `validate_locomo_gate_cpu.py` with any improved gate. Current prod gate (`gate-distill`) is broken on LoCoMo-shaped input.

### Commands

**Device:** `--device cpu` + `PSM_FORCE_CPU=1` on **local Windows only**. On **RunPod pod**, use `--device cuda` and `PSM_FORCE_CPU=0` (same script; GPU is faster).

#### Local (CPU)

```powershell
cd C:\Users\chkri\source\repos\PSM
$env:PYTHONPATH = "psm-model\src;psm-model\prod-memory;psm-model\scripts"
$env:PSM_FORCE_CPU = "1"

# Priority 1 — v5h JSON (full StorageDecision)
.venv\Scripts\python.exe psm-model\scripts\probe_locomo_hf_storage_cpu.py `
  --device cpu `
  --output-format json `
  --adapter-dir psm-model\prod-memory\checkpoints\hf-prod-v5h-qwen0.5b\adapter `
  --out benchmark\locomo\results\probe-locomo-hf-storage-v5h-json-cpu.jsonl

# Priority 2 — v2 tagged (pull adapter first if missing)
python psm-model\scripts\_sync_hf_lora.py --profile v5b --pull-only   # v5b local exists
# For v2: check HF repo krishnach7262/psm-prod-memory-hf for hf-prod-v2-qwen0.5b/adapter

.venv\Scripts\python.exe psm-model\scripts\probe_locomo_hf_storage_cpu.py `
  --device cpu `
  --output-format tagged `
  --adapter-dir psm-model\prod-memory\checkpoints\hf-prod-v2-qwen0.5b\adapter `
  --out benchmark\locomo\results\probe-locomo-hf-storage-v2-tagged-cpu.jsonl
```

#### RunPod pod (CUDA)

```bash
export PYTHONPATH=psm-model/src:psm-model/prod-memory:psm-model/scripts
export PSM_FORCE_CPU=0
python psm-model/scripts/probe_locomo_hf_storage_cpu.py \
  --device cuda \
  --output-format json \
  --adapter-dir psm-model/prod-memory/checkpoints/hf-prod-v5m-qwen0.5b/adapter \
  --out benchmark/locomo/results/probe-locomo-hf-storage-v5m-json-gpu.jsonl
```

Expand cases after first read (optional): increase coverage in `probe_locomo_hf_storage_cpu.py` `PRIORITY_QUESTIONS` / sample all QA-evidence dia_ids per conv.

### Pass criteria for “model is good enough” (write path, LoCoMo probe)

On a **larger slice** (≥50 evidence turns + matched non-evidence):

| Metric | Target (initial) |
|--------|------------------|
| `parse_ok` | ≥95% |
| `has_memory_content` (store cases) | ≥90% |
| `has_facts` (store cases) | ≥50% (stretch: ≥70%) |
| `content_grounded` | ≥90% |
| store/ignore decision vs QA-evidence proxy | F1 ≥0.6 (gate or single-pass action) |
| `has_temporal` | >0% on temporal-heavy turns (currently 0 everywhere — teacher gap) |

If **v5h/v2 pass** → deploy that adapter for LoCoMo re-ingest (tagged/json), add embeddings in ingest, skip v5k minimal_extract.

If **v5h/v2 fail** → train **one** new storage LoRA from `prod-teacher-cache-v4-4o.jsonl` or `prod-extraction-v3.jsonl` (tagged or json). Do **not** train on LoCoMo labels as primary curriculum.

---

## How to decide what to train (decision tree)

```
1. Run direct probes: v5h (json), v2 (tagged) on LoCoMo turns
        │
        ├─ PASS (parse + facts + grounded on evidence turns)
        │     → Deploy v5h or v2 for LoCoMo re-ingest
        │     → Fix gate separately (session-shaped binary curriculum from v3 ignore rows + fixtures)
        │     → Add teacher temporal labels OR accept episodic-only temporal for v1
        │
        └─ FAIL
              → One storage train from prod-extraction-v3 OR teacher cache → tagged/json
              → Eval on fixtures + LoCoMo probe before LoCoMo full ingest
              → Gate train only after storage format locked
```

**Do not train:**
- Another `minimal_extract` pass unless prod explicitly stays two-pass one-liner forever.
- LoCoMo QA-evidence as primary storage curriculum (OK as **gate** conversational diversity only).
- DPO gate at β=0.2 full 80 steps (proved 2/10 — over-corrects).

**Teacher data for new train (if needed):**
- Primary: `psm-model/prod-memory/data/prod-teacher-cache-v4-4o.jsonl` (1,474 rows, GPT-4o, facts)
- Or: `prod-extraction-v3.jsonl` (already merged + recall rows stripped for storage-only)
- Gap: **0% temporal** in teacher cache — add teacher prompt for `TE:`/`RT:` on conversation turns if temporal is required

---

## Architecture reference

**What LoCoMo used (wrong for full spec):**
```
ingest-psm-model.ts → PsmModelRuntime (hf-two-pass)
  → gate-distill (binary) → v5k-extract (minimal_extract)
  → episodic sentence only, facts=[]
```

**What we should test / deploy:**
```
remember → single HF LoRA (v5h json OR v2 tagged)
  → full StorageDecision → episodic + semantic + facts + indexables
  → optional: separate gate OR action in same tagged output
```

**Big data path (already exists):**
```
~/Downloads/training-data (codex/chatgpt/gemini)
  → ingest_training_data.py
  → GPT-4o teacher → prod-teacher-cache-v4-4o.jsonl
  → prod-extraction-v3.jsonl
  → hf-prod-v5h / hf-prod-v2 curricula → trained adapters on HF
```

---

## Key file paths

| Path | Purpose |
|------|---------|
| `psm-model/prod-memory/data/prod-teacher-cache-v4-4o.jsonl` | GPT-4o teacher labels (facts, indexables) |
| `psm-model/prod-memory/data/prod-extraction-v3.jsonl` | Merged training rows from teacher |
| `psm-model/prod-memory/data/hf-prod-v5k-extract.jsonl` | **240-row minimal_extract** (shortcut) |
| `psm-model/scripts/probe_locomo_hf_storage_cpu.py` | Direct write-path probe |
| `psm-model/scripts/validate_locomo_gate_cpu.py` | Gate-only LoCoMo validation |
| `benchmark/locomo/results/probe-locomo-hf-storage-*.jsonl` | Probe outputs |
| `benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db` | Ingest DB (minimal_extract ingest) |
| `docs/psm-model/2026-06-25-end-of-day-handoff.md` | v5k gate 8/10 context |
| `docs/psm-model/2026-06-26-end-of-day-handoff.md` | LoCoMo ingest + v5k deploy context |

---

## LoCoMo ingest status (as of 6/26 handoff)

- RunPod ingest completed later in session: ~2,651 stored, Hit@3 ~47% retrieval
- DB may have **~2× episodic rows** (double ingest append)
- Answer eval smoke ~45% with PSM recall — pre-fix path
- **Treat ingest DB as minimal_extract artifact** — do not use to judge full PSM model

---

## Next session checklist

1. [ ] Pull `hf-prod-v2-qwen0.5b/adapter` from HF if not local
2. [ ] Run **v5h json** probe on LoCoMo turns (command above)
3. [ ] Run **v2 tagged** probe
4. [ ] Compare `.summary.json` files — pick best storage adapter
5. [ ] If pass: plan LoCoMo re-ingest with that adapter + embeddings
6. [ ] If fail: build **one** tagged/json curriculum from teacher cache → single GPU train → re-probe before full ingest
7. [ ] Gate: only after storage format decided; use session-shaped labels from v3, not fixture-only

---

## One-liner

**We LoCoMo-tested the v5k minimal_extract shortcut (240 rows), not the v3/v5h model trained on 1,474 GPT-4o teacher rows. Direct-probe v5h (json) and v2 (tagged) on LoCoMo turns next; train only if those fail.**
