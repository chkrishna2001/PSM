# HF LoRA prod-memory v5 — end of day handoff (2026-06-23)

**Read first next session:** this file → `.cursor/skills/runpod-gpu-train/SKILL.md` → `.cursor/rules/runpod-auto-delete.mdc` → [training-pitfalls.md](training-pitfalls.md)

**Nothing running on RunPod.** Pod `bkbe17mgff9f0q` is **EXITED** (stopped after v4 eval). **No GPU billing.**

**Ship bar: still unmet** — need **≥8/10 `effective_stored`** on `psm-model/prod-memory/fixtures/cases.json`.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Best HF LoRA** | `hf-prod-v4-qwen0.5b` on `krishnach7262/psm-prod-memory-hf` |
| **v4 prod eval** | **3/10** `effective_stored` (parse 100%; ~0/10 real — template collapse) |
| **v4 train** | 2400 steps, loss **0.070**, 1136-row storage-only curriculum |
| **v5 train** | **Not started** — curriculum + resume-from-v4 + launcher wiring **not implemented** |
| **Teacher default** | **`google/gemma-4-31b-it`** (paid; `:free` rate-limited) — fallback `z-ai/glm-5.2` |
| **v4 teacher data** | `prod-extraction-v4.jsonl` (gpt-4o, 1474 sessions) + `hf-prod-v4.jsonl` |
| **RunPod pod** | `bkbe17mgff9f0q` stopped — can restart for v5 |

### Eval score history (prod fixtures)

| Model | `effective_stored` | Notes |
|-------|-------------------|--------|
| v1 | 0/10 | Unparseable output |
| v2 | 0/10 | Valid tags; generic template on every input |
| **v4** | **3/10** | Identical word-cloud mnemonic on all 10; 3 passes = accidental token overlap |
| Gate 058000 | 1/10 | Separate arch reference |

### v4 failure mode (why v5)

1. **Input-blind template** — same `word-cloud-word-count` mnemonic on every fixture (not in training data; format collapse).
2. **Zero ignore in curriculum** — `prod-extraction-v6-v4` has 0 ignore rows → noise cases always `store_episodic`.
3. **~99% of storage rows have `mnemonic` indexables** — model learned skeleton, not extraction.

---

## Teacher model (done today)

Switched bulk labeler after pilots:

| Model | Fixture pilot | Codex pilot (5) | Cost (est. 1474 rows) |
|-------|--------------|-----------------|----------------------|
| **gemma-4-31b-it** | **8/10** action, 6 grounded | **5/5** valid, **5/5** agree w/ gpt-4o | **~$0.15–0.50** |
| gpt-4o | 7/10 | (baseline cache) | ~$18–20 |
| glm-5.2 | 6/10 | — | ~$6–7 |
| lfm-2.5-thinking:free | 4/10 | — | $0 (broken JSON) |
| gemma-4-31b-it:free | blocked | — | 429 upstream |

**Code:** `prod_memory/openrouter_teacher.py` — `DEFAULT_MODEL = google/gemma-4-31b-it`.

**Pilot artifacts:**
- `psm-model/prod-memory/data/gemma-4-31b-fixture-pilot.json`
- `psm-model/prod-memory/data/gemma-codex-pilot.json`
- `psm-model/prod-memory/scripts/pilot_glm_teacher.py`
- `psm-model/prod-memory/scripts/pilot_codex_teacher.py`

**Optional tomorrow:** Gemma relabel **10 fixtures only** for v5 anchors (~$0.001). Skip full 1474-row relabel unless v5 train still fails.

---

## Tomorrow — priority order

### Phase 0 — Tokens & env (2 min)

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o krishnachhftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:OPENROUTER_API_KEY = "<from env or secrets>"
$env:PSM_HF_MODEL_REPO = 'krishnach7262/psm-prod-memory-hf'
$env:PSM_HF_DATASET_REPO = 'krishnach7262/psm-prod-memory-data'
```

---

### Phase 1 — Implement v5 plumbing (code, ~30–60 min)

**Not done yet.** Agent must land before train:

| File | Change |
|------|--------|
| `prod_memory/build_hf_curriculum.py` | Add **`hf-prod-v5`** profile: storage from `prod-extraction-v6-v4` + **ignore rows** from `prod-extraction-v4` (~15–20%) + **fixture anchors** (≤5 copies, Gemma or `build_minimal_fixture_rows`) + **0% recall** + strip/simplify `indexables` in HF labels (`minimal_extract` or facts-only) |
| `src/psm_model/hf_lora_train.py` | **`HF_RESUME_ADAPTER`** — load `hf-prod-v4-qwen0.5b/adapter` via `PeftModel.from_pretrained`, continue LoRA |
| `scripts/runpod_hf_lora_train.sh` | Pass resume adapter env; download v4 adapter before train |
| `scripts/_run_hf_lora.py` | **`v5` profile**: curriculum `hf-prod-v5.jsonl`, out `hf-prod-v5-qwen0.5b`, **1200–1600 steps**, resume v4 |
| `scripts/_watch_hf_lora.py` | **`v5` profile** for eval pull |

**Pitfalls (mandatory):** ≤5 fixture copies; no ×40 fail-copy; no recall in storage-fix run; read [training-pitfalls.md](training-pitfalls.md).

---

### Phase 2 — Build & upload v5 curriculum

```powershell
cd psm-model/prod-memory
$env:PYTHONPATH = "..\src;."

# Optional: Gemma fixture teacher labels (10 cases)
# python -m prod_memory.build_prod_extraction_v2 --fixtures-only ...  # if script supports; else pilot fixtures manually

python -m prod_memory.build_hf_curriculum --profile hf-prod-v5
# Verify manifest: ignore fraction ~15–20%, storage p50 sane, total rows ~900–1100

cd C:\Users\chkri\source\repos\PSM
python psm-model/scripts/_upload_hf_prod_assets.py --curriculum psm-model/prod-memory/data/hf-prod-v5.jsonl
```

**Sanity checks before upload:**
- `ignore` rows present (v4 had **0** — root cause for noise failures).
- No single content prefix dominates (v6 already deduped to max 2).
- Fixture rows grounded in actual `llmResponse` text.

---

### Phase 3 — Train v5 on RunPod (continue from v4)

**Two-phase launch** — never block on deploy; always **watch**:

```powershell
# 1) Deploy or restart stopped pod
python psm-model/scripts/runpod_ctl.py deploy --auto-gpu --name psm-hf-v5 --wait-ssh 300
python psm-model/scripts/runpod_ctl.py ssh-info <pod_id>   # proxy-user

# OR restart existing:
python psm-model/scripts/runpod_ctl.py start-pod bkbe17mgff9f0q
python psm-model/scripts/runpod_ctl.py ssh-info bkbe17mgff9f0q

# 2) Launch train (sync code first — v5 plumbing not on remote yet)
python psm-model/scripts/_run_hf_lora.py `
  --pod-id <id> --proxy-user <user> `
  --profile v5 --sync-code

# 3) Watcher (mandatory — v4 idle-billed ~2h when watcher skipped)
python psm-model/scripts/_watch_hf_lora.py `
  --pod-id <id> --proxy-user <user> `
  --profile v5 --interval-sec 600 --stop-pod-on-done
```

**Train settings (target):**

| Setting | v4 | v5 |
|---------|----|----|
| Init | base Qwen | **continue `hf-prod-v4-qwen0.5b` adapter** |
| Steps | 2400 | **1200–1600** |
| Curriculum | storage-only, 0 ignore | storage + **ignore ~15–20%**, simpler labels |
| Output prefix | `hf-prod-v4-qwen0.5b` | `hf-prod-v5-qwen0.5b` |

**Verify within 90s:** `python psm-model/scripts/runpod_ctl.py verify-pod --pod-id <id> --proxy-user <user>`

---

### Phase 4 — Eval & decide

Watcher runs eval when train finishes. Or manually:

```powershell
python psm-model/scripts/_run_hf_lora_eval.py --pod-id <id> --proxy-user <user> --profile v5
```

**Read results:**

```powershell
python -c "import json; d=json.load(open('psm-model/prod-memory/results/hf-prod-v5-qwen0.5b-prod-grounding.json')); print(json.dumps(d['aggregate'], indent=2))"
```

| Result | Action |
|--------|--------|
| **≥8/10** | Ship candidate; verify HF manifest; stop pod; document |
| **4–7/10** | Mine per-case failures; Gemma fixture relabel; v5b micro-run (800 steps) |
| **≤3/10** (same template all cases) | Try `minimal_extract` output format; reduce indexables in labels; check resume adapter actually loaded |

**Success criteria beyond score:**
- **No identical `raw_output`** across all 10 fixtures.
- **Noise suite:** model should output `ignore`, not store + grounding_reject.

---

### Phase 5 — Pod cleanup

Per `.cursor/rules/runpod-auto-delete.mdc`:

1. Local eval JSON pulled
2. HF adapter + eval JSON on `krishnach7262/psm-prod-memory-hf`
3. **Then** stop pod (not delete unless user confirms)

```powershell
python psm-model/scripts/runpod_ctl.py stop-pod <pod_id>
```

---

## Key paths

| Purpose | Path |
|---------|------|
| Prod fixtures | `psm-model/prod-memory/fixtures/cases.json` |
| v4 eval | `psm-model/prod-memory/results/hf-prod-v4-qwen0.5b-prod-grounding.json` |
| v4 teacher data | `psm-model/prod-memory/data/prod-extraction-v4.jsonl` |
| v4 storage mix | `psm-model/prod-memory/data/prod-extraction-v6-v4.jsonl` |
| HF curriculum builder | `prod_memory/build_hf_curriculum.py` |
| Train launcher | `psm-model/scripts/_run_hf_lora.py` |
| Watcher | `psm-model/scripts/_watch_hf_lora.py` |
| Teacher | `prod_memory/openrouter_teacher.py` |

## HF repos

| Repo | ID |
|------|-----|
| Model (adapters, eval) | `krishnach7262/psm-prod-memory-hf` |
| Dataset (curriculum) | `krishnach7262/psm-prod-memory-data` |

**Current HF adapters:** `hf-prod-v2-qwen0.5b`, `hf-prod-v4-qwen0.5b` (checkpoints 400–2400).

---

## Explicit non-goals tomorrow

- **Do not** relabel all 1474 rows with Gemma unless v5 eval fails and data quality is suspect.
- **Do not** delete pod until artifact checklist passes.
- **Do not** launch train without watcher (`--interval-sec 600`).
- **LoCoMo** — still deferred until prod parse/grounding stable.

---

## One-line start tomorrow

> Implement v5 curriculum + resume-from-v4 → build `hf-prod-v5.jsonl` → upload → train on RunPod with watcher → target **≥8/10 effective_stored**.
