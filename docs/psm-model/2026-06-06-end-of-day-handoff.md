# PSM 50M — end of day handoff (2026-06-06)

**Read first tomorrow:** this file → [session-log.md](session-log.md) → [training-playbook.md](training-playbook.md).

Training is **stopped**. RunPod pod **deleted**. Passing weights are on **HuggingFace** only (local checkpoints may be stale or absent).

---

## What shipped today

| Artifact | HF path | Gate |
|----------|---------|------|
| Phase 1 action model | `psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.pt` | Gate 2 **PASS** |
| Full StorageDecision model | `psm-model/checkpoints/real-v3-50m-full-v2.pt` | Gate 3 **PASS** |
| Same (canonical step) | `psm-model/checkpoints/real-v3-50m-full-v2-step-022800.pt` | Gate 3 **PASS** |

**HF model repo:** `chkrishna2001/psm-50m-mixed-v1-run` (private)  
**HF dataset repo:** `chkrishna2001/psm-50m-action-mixed-v1` (curricula, `runpod/` scripts, probes)

Gate 3 was unlocked by **probe-anchor curriculum** on the pod (`direct_probes` × 500 copies mixed into full-storage). That file lived on the pod volume; rebuild locally if needed (command below).

---

## Tomorrow: step 1 — pull checkpoints locally

```powershell
cd C:\Users\chkri\source\repos\PSM
hf download chkrishna2001/psm-50m-mixed-v1-run `
  psm-model/checkpoints/real-v3-50m-full-v2.pt `
  psm-model/checkpoints/real-v3-50m-full-v2.tokenizer.json `
  psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.pt `
  psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.tokenizer.json `
  --local-dir .
```

**Local rule:** eval on **`--device cpu` only** (local GPU eval crashed the laptop). Train on RunPod if more steps are needed.

---

## Tomorrow: step 2 — re-verify gates (CPU)

```powershell
$env:PYTHONPATH = 'psm-model\src'

# Gate 3 — must pass before any psm-core wiring
.\.venv\Scripts\python.exe -m psm_model.eval_checkpoint `
  psm-model\checkpoints\real-v3-50m-full-v2.pt `
  psm-model\data\probes\direct_probes.jsonl `
  --device cpu

# Gate 2 — action model still valid
.\.venv\Scripts\python.exe -m psm_model.gate_checkpoint `
  psm-model\checkpoints\real-v3-50m-action-mixed-v2-step-009800.pt `
  --mode phase1-action --device cpu --output-format action

# Qualitative (required; do not trust probe macro alone)
.\.venv\Scripts\python.exe -m psm_model.action_smoke `
  psm-model\checkpoints\real-v3-50m-action-mixed-v2-step-009800.pt `
  psm-model\data\direct-behavior-v1\manual-probe.jsonl `
  --device cpu --output-format action --prefix-eval
```

Pass bars unchanged: Gate 3 = 100% on all `direct_probes` metrics; Gate 2 = expanded ≥ 0.85, manual ≥ 0.80, manual smoke ≥ 0.80.

---

## Tomorrow: step 3 — Gate 4 (product)

Playbook rule: **wire `psm-core` only after Gate 3 model-only generation passes** — that bar is met.

Suggested order:

1. **Broader eval** — run `eval_checkpoint` on expanded / real-v1 holdout rows (not only 5 direct probes).
2. **Sample generations** — `python -m psm_model.generate` on a few real conversation snippets; inspect tagged DSL quality.
3. **Optional:** `gate_checkpoint --mode product-safe` (needs `action-foundation-v1` probe files on disk; download from HF dataset if missing).
4. **Integrate** — point `psm-core` decoder at `real-v3-50m-full-v2.pt`, tagged output format, same tokenizer sidecar.

`safe_generate` remains an optional constrained bridge; not required now that Gate 3 passes model-only.

---

## If Gate 3 regresses or you need more training

Resume on RunPod from **step-22800** (not 23000 — that save was corrupt).

```bash
# On pod after bootstrap (stock PyTorch image + HF download)
export PYTHONPATH=psm-model/src
python3 -m psm_model.train \
  psm-model/data/curriculum/psm-50m-full-storage-v2-probe-anchor.jsonl \
  --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
  --resume psm-model/checkpoints/real-v3-50m-full-v2-step-022800.pt \
  --tokenizer psm-model/checkpoints/real-v3-50m-full-v2-step-022800.tokenizer.json \
  --steps 25000 --batch-size 1 --preset 50m \
  --output-format tagged --sampling action_balanced \
  --device cuda --save-every 200
```

**Rebuild probe-anchor curriculum locally** (if not on HF):

```powershell
$env:PYTHONPATH = 'psm-model\src'
python -m psm_model.make_action_first_curriculum `
  psm-model\data\curriculum\psm-50m-direct-anchor-v1.jsonl `
  psm-model\data\probes\direct_probes.jsonl --copies 500
# Note: make_action_first_curriculum strips memory — for full StorageDecision rows,
# duplicate probes with a small script (see session-log) or upload anchor file from HF later.
```

For Gate 3, anchor rows must keep **full** `expected` (memory + facts), not action-only. The pod used a Python loop copying full probe JSON with ids `direct-anchor:{copy}:{id}`.

**Critical:** `--steps` is **absolute** target, not “additional steps”. Example: resume at 9800 needs `--steps 12800` for +3000.

---

## RunPod ops (when needed)

- **SSH host:** `runpod-psm` in `~/.ssh/config` (update after each deploy).
- **Windows SSH:** use `ssh -tt runpod-psm` with piped stdin; bare `ssh runpod-psm "cmd"` fails with PTY error.
- **Deploy:** stock image only — `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Custom `psm-50m-train` image was never pushed (IMAGE_AUTH_ERROR).
- **API key:** `o runpodkey` → clipboard → `$env:RUNPOD_API_KEY`
- **Stop pod when idle** — exited pod still bills until deleted.

```powershell
python psm-model\scripts\runpod_ctl.py list-pods
python psm-model\scripts\runpod_ctl.py delete-pod <pod_id>
```

---

## Pitfalls (learned today)

1. **Idle GPU = wasted money** — training died silently; only `psm-sync` tmux left. Always `pgrep -af psm_model.train` + `nvidia-smi` after kickoff.
2. **Corrupt checkpoint save** — `step-023000.pt` was 134 MB (interrupted write). Resume from last good step (22800).
3. **HF commit rate limit** — 128 commits/hour; prefer single-folder `hf upload` or `sync_training_to_hf.py --only-new`.
4. **PowerShell + SSH** — `$CKPT` variables expand locally and strip paths; use literal paths in heredocs.
5. **Gate 3 needs probe anchor** — full-storage curriculum alone was not enough by step 12800; direct probe repetition (500×) fixed it by ~22800.

---

## Phase status

| Gate | Status |
|------|--------|
| 0 Data filter | PASS |
| 1 Classifier | PASS |
| 2 Phase 1 50M | **PASS** — `step-009800` mixed-v2 |
| 3 Full StorageDecision | **PASS** — `real-v3-50m-full-v2.pt` @ step 22800 |
| 4 Product / psm-core | **Start tomorrow** |

---

## Do not resume (denylist)

See `psm-model/checkpoints/DENYLIST.txt` — v2/repair/action-head checkpoints, corrupt `step-023000.pt`.
