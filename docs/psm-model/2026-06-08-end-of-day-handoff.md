# PSM 50M — end of day handoff (2026-06-08)

**Read first tomorrow:** this file → [session-log.md](session-log.md) → [runpod-ssh-ops.md](runpod-ssh-ops.md) → [training-playbook.md](training-playbook.md).

**Nothing running on RunPod** — all pods deleted. Gate 4 v4 train + expanded eval **finished** today; LoCoMo 25 **deferred**.

---

## Where we are

| Item | Status |
|------|--------|
| RunPod pods | **None** (billing stopped) |
| Gate 4 expanded @ **45600** | **FAIL** — parse **88.1%**, action **88.1%** (bar: parse/schema ≥95%, action ≥85%) |
| Best eval (lost weights) | **42000** — parse **87.4%**, action **87.1%** |
| Best **on HF** (good lineage) | **36000** — parse **85.2%** @ prior eval |
| LoCoMo 25 | **Cancelled today** — was queued in tmux; pod deleted before it ran |
| HF model repo | Sprawl pruned (**42** files, 18k–41k); **42000/45600 `.pt` still NOT on HF** |
| HF dataset repo | **45600 eval JSON + registry** uploaded |

**Action accuracy clears the bar; parse/schema do not.** Same failure mode as prior days: tagged output parse (missing/malformed `R:`, `F:`, `Q:` lines), not wrong actions.

---

## What happened today (timeline)

1. **Gate 4 v4 retrain** on pod `bllek0twl70y3j` (`psm-gate4-v4`), RTX 2000 Ada — resume **41600 → 45600**.
2. **Expanded eval @ 45600** ran ~**65–84 min** (913 budget rows, serial GPU decode, no progress logs).
3. **Eval result:** parse/schema/action all **~88.1%** → gate **FAIL** (parse/schema need 95%).
4. **HF final sync failed** — `runpod_upload_gate4.sh` missing on pod clone.
5. **Pod auto-deleted** after ctl session — local checkpoints **42000–45600 lost** again.
6. **LoCoMo 25** was queued in tmux `psm-locomo` (wait-for-eval); **never started** (pod gone).
7. **Code fixes:** delete gate blocks pod delete until target `.pt` on HF; recovery defaults 36000→42000; `runpod_locomo.sh` added (for later).

---

## Metrics @ step 45600 (CUDA, 913 rows)

```
parse_valid_rate    0.881
schema_valid_rate   0.881
action_accuracy     0.881
facts_exact_rate    0.743
memory_content      0.742
avg_generated_tokens 213
```

Local report: `psm-model/checkpoints/gate-eval/gate4-full-expanded-step-045600.json`  
HF copy: `chkrishna2001/psm-50m-action-mixed-v1` → `curriculum/gate4-full-expanded-step-45600.json`

---

## Tomorrow’s goal (unchanged ship bar)

| Metric | Today @ 45600 | Ship bar |
|--------|---------------|----------|
| Parse / schema | **88.1%** | **≥95%** |
| Action | **88.1%** | ≥85% |

**Success tomorrow:** Gate 4 expanded **PASS** on CUDA, Gate 3 direct still PASS, checkpoint **on HF**, second confirm eval, then product E2E. **LoCoMo 25+ only after parse gate passes** (deferred today).

**Do not** use 41200/41600/45600 as resume — micro/repair or lost lineage. **Resume from HF `step-036000.pt`** with **v4 curriculum** (complete-tag drills ×50, expanded ×100).

---

## First commands tomorrow

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey

# 1) Confirm HF has 36000 .pt (only good base)
python psm-model/scripts/runpod_ctl.py list-pods   # should be []

# 2) Optional: prune more HF sprawl if uploads still 403
$env:PYTHONPATH = "psm-model/src"
python -m psm_model.gate4_checkpoint_registry prune-hf-sprawl `
  --repo-id chkrishna2001/psm-50m-mixed-v1-run --min-step 18000 --max-step 41000

# 3) Train v4 recovery 36000 → 42000 (or 44000 if stable)
python psm-model/scripts/runpod_ctl.py train-gate4 `
  --deploy --gpu "NVIDIA RTX 2000 Ada Generation" `
  --name psm-gate4-recover `
  --curriculum-builder v4 `
  --resume-checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-036000.pt `
  --target-steps 42000 `
  --eval-every 0 --keep-local 2 --keep-pod

# 4) After train: upload BEFORE delete (ctl blocks delete if .pt missing on HF)
python psm-model/scripts/runpod_ctl.py upload-gate4 --pod-id <pod_id> --keep-local 2
# verify step-042000.pt on HF, then delete pod
```

**Important:** Ensure pod gets fresh `runpod_upload_gate4.sh` (dataset `psm-code/` or src sync). Use `--keep-pod` until HF upload verified.

---

## Incidents / lessons

| Issue | Mitigation |
|-------|------------|
| Pod deleted before HF upload | Delete gate in `runpod_ctl.py` — blocks delete if target/best `.pt` missing on HF |
| `runpod_upload_gate4.sh` missing on pod | Bootstrap must pull from dataset `psm-code/`; verify before train ends |
| HF private storage full (403) | Prune sprawl; batch uploads; keep only milestones + best |
| HF commit rate limit (429) | Single-commit folder uploads; avoid per-file loops |
| Expanded eval looks “stuck” | Normal — 913 × ~4–5s ≈ 75–90 min on RTX 2000 Ada; no progress until JSON dump |
| LoCoMo queued then lost | Don’t queue long secondary jobs until upload confirmed; defer LoCoMo until Gate 4 pass |

---

## Registry

`psm-model/checkpoints/gate4-checkpoint-registry.json` — best eval **42000** (lost), latest eval **45600** (lost), HF resume **36000**, LoCoMo deferred.

---

## Not doing tomorrow (unless parse passes)

- LoCoMo 25 smoke on RunPod
- `psmModel` default enable
- Repair-only micro loops (46k/50k regressed vs 42k)
