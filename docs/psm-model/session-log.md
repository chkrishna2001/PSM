# PSM Model Session Log

Rolling log for 50M training runs. See [training-playbook.md](training-playbook.md).

## Goal

50M PSM decoder: action selection, episodic/semantic memory, facts, temporal fields, schema-valid `StorageDecision`.

## Decisions

- Classifier is diagnostic/bridge only; product model is `psm-model` decoder.
- No resume from denylisted v2/repair checkpoints.
- Phase 1 action-only before full StorageDecision.
- Local GPU cap 50% VRAM; Colab after local gates.
- `nano-psm/` kept until Gate 3; classifier path not used for product training.

## 2026-06-04 (historical)

- Prior action-first run from `real-v2-50m-step-001200` — diagnostic only, not main path.
- Label audit on full curriculum: ignore ~28%, 518 high / 1697 medium risks.
- Docs consolidated under `docs/psm-model/`; restart guardrails implemented.

## 2026-06-04 — Plan execution (restart)

### Gate 0 — Data

- Filtered `psm-50m-full-storage-v1.jsonl` → `psm-50m-full-storage-v1-filtered.jsonl` (25257 kept, 510 dropped high-risk).
- Built `psm-50m-action-first-v1-filtered.jsonl` (25257 rows).
- Built `expanded-probe-v1-filtered.jsonl` (920 rows).
- `label_audit --fail-on-high-risk` on filtered set: pass.

### Gate 1 — Classifier

- Checkpoint: `psm-model/checkpoints/psm-action-classifier-v2-filtered.pt`
- Expanded probe macro: **0.959**, collapse: **0.27** — **PASS**

### Gate 2 — Phase 1 scratch 50M

- Checkpoint: `psm-model/checkpoints/real-v3-50m-action-scratch-v1.pt` (500 steps, CUDA, `--cuda-memory-fraction 0.5`)
- `phase1-action` gate: expanded macro **0.41**, manual macro **0.20** — **FAIL** (thresholds 0.85 / 0.80)
- Training probe at step 500: macro 0.41, collapse 0.42 (improved from step 100–200 store_episodic collapse)
- **Gate 3 blocked** until Phase 1 gate passes (do not resume for full StorageDecision yet).

### Gate 4 — Product

- **Deferred:** no `psm-core` wiring; `safe_generate` remains optional bridge per playbook.

## 2026-06-04 — Phase 1 continuation (500 → 3000 steps)

- Resuming `real-v3-50m-action-scratch-v1` from step 500; target 3000 steps before Colab.

## 2026-06-04 — Phase 1 continuation (2900 → 3500)

- Prior run crashed at step 2988 (overlong row 4897 tokens). Fixed: skip 31 overlong rows at load.
- Resumed from step 2900 to 3500: **success** (~33 min).
- Final `phase1-action` gate: expanded macro **0.68**, manual **0.40** — **FAIL** (need 0.85 / 0.80). Not Colab-ready yet.

### Code/docs delivered

- `docs/psm-model/` playbook, session log, denylist, archive lessons.
- Removed stale `docs/psm-model-*.md` handoffs and action-head-repair Colab notebook.
- `train.py`: `--eval-every`, `--probe`, collapse abort; save-before-probe fix.
- `gate_checkpoint.py`: `--mode phase1-action`.
- `action_diagnostics.py`: context truncation in `score_actions`, `collapse_fraction` on prefix eval.

## 2026-06-04 — Mixed curriculum pivot (direct-behavior + storage)

### Finding (nano-psm pattern reproduced)

- At step **5000** on storage-only curriculum: expanded probe macro **0.56**, but **manual smoke match_rate 0.10** (mostly `flag_and_store`).
- Step **3000** had best expanded macro **0.72**; manual smoke still **0.30** — metrics alone are not sufficient.

### Fix

- Built `psm-50m-action-mixed-v1-ctx2048.jsonl` = 25,226 storage action-first + 10,628 direct-behavior (4× copies).
- New run: `real-v3-50m-action-mixed-v1.pt` (scratch, 8000 steps target).
- Stopped storage-only 20k continuation (regressing on manual cases).

### Qualitative eval command (required every milestone)

```powershell
python -m psm_model.action_smoke psm-model/checkpoints/<ckpt>.pt psm-model/data/direct-behavior-v1/manual-probe.jsonl --device auto --output-format action --prefix-eval
```

## 2026-06-04 — End of day (training stopped)

- **Stopped** all training for the day. Active path: `real-v3-50m-action-mixed-v1` @ **step 200** (`step-000200.pt`).
- **Handoff:** [2026-06-04-end-of-day-handoff.md](2026-06-04-end-of-day-handoff.md) — resume commands, checkpoint paths, LoCoMo/REALTALK conversion notes, manual-smoke requirement.
- **External data:** LoCoMo + REALTALK already flow through `nano-psm` → `fast-mixed` / `retention-blend` → `convert_nano_dataset` → `nano-hf-storage-v1`, `real-v1`, and `psm-50m-full-storage-v1`. “Letta” in repo = benchmark name, not a training corpus. Tomorrow: consider explicit `combine_jsonl` of converted benchmark rows into mixed curriculum.
- **scratch-v1** left at ~5.2k steps for reference; manual smoke 10% @ step 5000 despite probe 0.56.

## 2026-06-06 — RunPod mixed-v1 training (primary path; Colab retired)

- **Runtime:** RunPod RTX 4090 via SSH (`runpod-psm`); HF sync `chkrishna2001/psm-50m-mixed-v1-run` + dataset `chkrishna2001/psm-50m-action-mixed-v1`.
- **Resume:** step 400 → target 8000; `tmux` session `train` on pod.
- **Best expanded probe so far:** step **4400** macro **0.648**, collapse **0.48**.
- **Manual smoke @ 4400:** `match_rate` **0.30** (need 0.80). Strong: `ignore`, `update_existing`, `flag_conflict`. Weak: `promote_semantic` vs `store_episodic` confusion.
- **Gate 2:** still **FAIL** on manual probes despite rising expanded macro — same nano-psm pattern.

### Three-stage product map (gates)

| Stage | Capability | Training gate | Current status |
|-------|------------|---------------|----------------|
| **1. Categorize** | `ignore`, `store_episodic`, `promote_semantic` | Gate 2 `phase1-action` + manual smoke | **In progress** — mixed-v1 RunPod |
| **2. Consolidate / conflict** | `update_existing`, `flag_conflict`, `flag_and_store` | Gate 2 (same) + hard-behavior probes | Partial — 3/10 manual cases pass |
| **3. Recall** | `recall_context`, full `StorageDecision`, indexable selection | Gate 3 full tagged output + direct probes | **Blocked** until Gate 2 passes |

### Data pipeline after Gate 2 (or if 8k manual still < 0.80)

1. `convert_nano_dataset` from HF: `chkrishna2001/nano-psm`, fast-mixed (LoCoMo/REALTALK), retention-blend.
2. `combine_jsonl` into mixed curriculum v2 (storage + direct-behavior + converted benchmark rows).
3. Letta: benchmark competitor only today; add adapter if export data becomes available.
4. Phase 2 (Gate 3): resume Gate-2 checkpoint on `psm-50m-full-storage-v1-filtered.jsonl` with `--output-format tagged`.

## 2026-06-06 — mixed-v1 RunPod run complete (pod stopped)

- **Training:** finished step **8000**. Best expanded probe @ 8000: macro **0.711**, collapse **0.40**. Manual smoke @ 8000: **0.40** (gate still FAIL).
- **HF:** `chkrishna2001/psm-50m-mixed-v1-run` — `step-008000.pt`, `real-v3-50m-action-mixed-v1.pt`, metrics uploaded. Full `sync_training_to_hf.py` run aborted (network); slow because it re-uploads all checkpoints.
- **Pod:** local `.pt` files deleted after upload; **stop pod in RunPod console** to avoid disk charges.

## 2026-06-06 — RunPod reusable image (ephemeral pods)

- **Goal:** delete/recreate pods without manual bootstrap; HF holds checkpoints/data.
- **Docker:** `psm-model/docker/Dockerfile` → `chkrishna2001/psm-50m-train:latest` (PyTorch + tmux + repo code + HF bootstrap on start).
- **Scripts:** `runpod_bootstrap.sh`, `runpod_entrypoint.sh`, `runpod_ctl.py` (`list-pods`, `stop-all`, `create-template`, `deploy`), `runpod_build_image.ps1`.
- **HF sync fix:** `sync_training_to_hf.py --only-new` skips already-uploaded files (manifest).
- **API key:** `o runpodkey` → `$env:RUNPOD_API_KEY = Get-Clipboard`
- **HF auth:** RunPod secret `HF_TOKEN` → template env `HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}` (no plain token in template/deploy).
