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

## 2026-06-06 — Gate 2 repair: mixed-v2 + RunPod redeploy

- **Local rule:** training on RunPod only; **local eval `--device cpu` only** (GPU eval crashed laptop).
- **Baseline @ step-7600** (downloaded via `hf download`): expanded macro **0.785**, manual **0.70** — FAIL. Failure mode: over-predicts `store_episodic` on manual `promote_semantic` / `ignore` cases.
- **mixed-v2 curriculum** built + uploaded to HF dataset: `curriculum/psm-50m-action-mixed-v2-ctx2048.jsonl` (46,825 rows: storage + direct-behavior 8× + manual-probe 300×).
- **Training weights (v2):** `promote_semantic=6`, `store_episodic=2.5`, `ignore=4` (counter over-prediction of episodic).
- **RunPod pod:** `psm-mixed-v2` id `j9pht1oxpeomj0`, RTX 4090, stock image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (custom `psm-50m-train` image exited immediately).
- **SSH:** `ssh -p 29016 root@103.196.86.82` (update `~/.ssh/config` `runpod-psm` when pod changes).
- **Kickoff on pod:** `hf download chkrishna2001/psm-50m-action-mixed-v1 runpod/runpod_remote_kickoff.sh --local-dir /tmp && bash /tmp/runpod/runpod_remote_kickoff.sh`
- **Resume:** step-7600 → `real-v3-50m-action-mixed-v2.pt`, target 12k steps, `--manual-probe` logged every 200 steps.

## 2026-06-06 — RunPod IMAGE_AUTH_ERROR (resolved)

- **Failed pod:** `reh84ziwwkgkj3` used `chkrishna2001/psm-50m-train:latest` — image was **never pushed** to Docker Hub → RunPod stopped pod (no charge after stop).
- **Fix:** all deploys now use stock `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; bootstrap via HF dataset `runpod/` scripts.
- **Active pod:** `br27u2fbf2d73n` (`psm-mixed-v2`), template `mo1fjgnycd`, proxy SSH `br27u2fbf2d73n-644120ec@ssh.runpod.io`, `~/.ssh/config` host `runpod-psm`.

## 2026-06-06 — mixed-v2 training running (RunPod)

- **Pod:** `br27u2fbf2d73n`, RTX 4090, tmux `psm-mixed-v2`, HF sync tmux `psm-sync` (every 10 min).
- **Resume:** step-7600 → `real-v3-50m-action-mixed-v2.pt`, target 12k steps.
- **Probe @ 7800:** expanded **0.799**, manual **0.80** (manual gate threshold met), collapse **0.30**.
- **Probe @ 9800:** expanded **0.853**, manual **0.80** — **Gate 2 PASS** (`gate_checkpoint --mode phase1-action`).
- **Best checkpoint:** `real-v3-50m-action-mixed-v2-step-009800.pt`
- **Gate 3:** resumed step-9800 → `--steps 12800` (absolute), tagged full-storage curriculum, **complete @ step 12800**.
- **Output:** `real-v3-50m-full-v2.pt` — HF sync via `psm-sync` tmux.

## 2026-06-06 — Gate 3 complete (RunPod `br27u2fbf2d73n`, pod deleted)

### Gate status (end of session)

| Gate | Checkpoint | Result |
|------|------------|--------|
| **2 Phase 1 action** | `real-v3-50m-action-mixed-v2-step-009800.pt` | **PASS** — expanded 0.853, manual 0.80 |
| **3 Full StorageDecision** | `real-v3-50m-full-v2-step-022800.pt` → `real-v3-50m-full-v2.pt` | **PASS** — `direct_probes` 100% all metrics |
| **4 Product / psm-core** | — | **Next** — Gate 3 model-only pass met; wiring deferred |

### Gate 3 training arc

1. **12800** (full-storage only): `eval_checkpoint` **FAIL** — 0% parse/schema; garbage tagged output (`A:-`, wrong actions).
2. **16800** (extended +4000 steps): **FAIL** — 20% parse (`dated_event` only).
3. **Probe-anchor curriculum** built on pod: `psm-50m-full-storage-v2-probe-anchor.jsonl` = 25,257 filtered storage + **2,500** rows (`direct_probes.jsonl` × 500 copies, 6 probe cases).
4. Resumed 16800 → 25000 on probe-anchor curriculum.
5. **22800:** `eval_checkpoint` **PASS** — parse/schema/action/memory/facts all **1.0** on 5 `direct_probes`.
6. **22999:** training died mid-save; `step-023000.pt` corrupt (134 MB vs ~631 MB). GPU idle ~8+ min before noticed.
7. Promoted **step-22800** → `real-v3-50m-full-v2.pt`; stopped further training (gate already passed).

### HF + pod

- **Model repo:** `chkrishna2001/psm-50m-mixed-v1-run`
- **Uploaded & verified:** `real-v3-50m-full-v2.pt`, `real-v3-50m-full-v2-step-022800.pt` (+ tokenizers), ~631 MB each.
- **HF pitfall:** commit rate limit (128/hour) during bulk `sync_training_to_hf.py`; user completed upload manually.
- **Pod:** `psm-mixed-v2` `br27u2fbf2d73n` stopped then **deleted** (no GPU billing).

### Three-stage product map (updated)

| Stage | Gate | Status |
|-------|------|--------|
| **1. Categorize** | Gate 2 | **PASS** @ step-9800 action checkpoint |
| **2. Consolidate / full StorageDecision** | Gate 3 | **PASS** @ step-22800 full checkpoint |
| **3. Recall / product** | Gate 4 + psm-core | **Tomorrow** — broader eval, integration |

### Handoff

- [2026-06-06-end-of-day-handoff.md](2026-06-06-end-of-day-handoff.md)

## 2026-06-07 — Gate 4 wiring + RunPod SSH fixes (pod deleted)

### Gate status

| Gate | Result | Where verified |
|------|--------|----------------|
| **3 Full StorageDecision** | **PASS** | CPU local + CUDA RunPod (`gate-eval/gate3-full-direct.json`) |
| **2 Phase 1 action** | **PASS** | CUDA RunPod (`gate2-phase1-action.json`, expanded 0.853) |
| **4 Product / psm-core** | **PASS** | `psm-memory remember --psm-model` → `promote_semantic` + DB write |

### RunPod / SSH

- Fixed `runpod_ctl.py`: proxy SSH requires piped `bash -s` + chunked base64 heredoc (remote argv ignored).
- GPU eval one-shot PASS; reports in `psm-model/checkpoints/gate-eval/summary.json`.
- Eval pod `znq97fgibrg758` deleted after pull.
- Ops cheat sheet: [runpod-ssh-ops.md](runpod-ssh-ops.md).

### psm-core integration fixes

- `extractStoragePayload`: parse last JSON line in prompt (not schema example `{`).
- `PsmModelRuntime`: stdin to `remember_cli` (Windows `ENAMETOOLONG`).
- `remember_cli`: assistant-only → `User:` mapping; drop auto `source_timestamp` on non-temporal text.

### Open / minor

- `pip install numpy` in `.venv` (warning only).
- `scp` via proxy still broken; tar-pull fallback in `runpod_ctl.py` (verify on next deploy).

## 2026-06-07 — Gate 4 expanded eval (pod `xxguvuvmwbf2oz`, deleted)

After commit `942b711` (context overflow skip + token-budget filter):

| Eval | Result |
|------|--------|
| Gate 3 direct probes | **PASS** (CUDA) |
| Gate 2 action + smoke | **PASS** (CUDA) |
| Gate 4 expanded (913/920 rows, 7 dropped >1536 tok) | **FAIL** — action 0.36, parse 0.58, facts_exact 0.25 |

Expanded eval completed without crash. Strict direct-probe thresholds (1.0) were applied in that run; product path on 5 probes still passes.

**2026-06-07 (later):** Codified Gate 4 bar in `psm_model.gates.EXPANDED_PROBE_THRESHOLDS` (action ≥0.85, parse/schema ≥0.95, content/facts ≥0.50). `eval_checkpoint --gate-mode expanded` and `analyze_eval_report` bucket failures (parse / action / content). Current full model still **FAIL** vs expanded bar — resume training from step-22800 with expanded + ignore-heavy curriculum until Gate 4 passes.

## 2026-06-07 — Gate 4 training started (pod `jk2bodseapigvi`)

- **Proxy SSH:** `jk2bodseapigvi-644115c2@ssh.runpod.io`
- **Direct TCP:** `root@213.173.110.93 -p 12473`
- **Command:** `runpod_ctl.py train-gate4 --pod-id jk2bodseapigvi --proxy-user jk2bodseapigvi-644115c2 --target-steps 28000`
- **Resume:** `real-v3-50m-full-v2-step-022800.pt` → absolute target **28000** (+5200 steps)
- **Curriculum:** `psm-50m-full-storage-v4-gate4.jsonl` — 35,901 rows (25,257 base + 2,500 direct anchor + 7,360 expanded + 784 ignore extra)
- **tmux:** session `psm-gate4`, log `/tmp/psm-gate4-train.log`, metrics `real-v3-50m-full-v2-gate4.metrics.jsonl`
- **Probe during train:** expanded every 400 steps; direct probes as manual-probe sanity check
- **Note:** first deploy failed on stale `/workspace/PSM` clone; fixed bootstrap + restarted successfully

### Post-train eval @ step-28000 (CUDA, pod `jk2bodseapigvi`)

| Gate | Result |
|------|--------|
| Gate 3 direct probes | **PASS** (100% all metrics) |
| Gate 2 action + smoke | **PASS** |
| Gate 4 expanded (913 rows) | **FAIL** — action **0.49**, parse **0.62**, facts_exact **0.34**, memory_content **0.33** |

Improved vs pre-train baseline (action 0.36, parse 0.58) but still far from ship bar (action ≥0.85, parse ≥0.95). Reports pull failed (SCP/tar); full JSON in eval terminal log.

### Round 2 complete @ step-32000

- Training finished; uploaded step-032000 + promoted `full-v2.pt` to HF.
- Post-train eval (CUDA): Gate 3 **PASS**, Gate 4 expanded **FAIL** — action **0.51**, parse **0.60** (vs 0.49/0.62 @28k; still below ship bar).
- Pod `jk2bodseapigvi` deleted after eval.

### Round 2 training (resume @28000 → 32000)

- Upload: `runpod_upload_gate4.sh` / `train-gate4 --upload-first`
- Heavier curriculum: expanded ×12, ignore extra ×6
- Pod: `jk2bodseapigvi` / proxy `jk2bodseapigvi-644115c2`

## 2026-06-07 — Gate 4 production curriculum (`gate4-train-v1`)

**Decision:** Stop diluting with 25k full-storage base. Train eval-aligned full DSL first.

- **New builder:** `psm_model.build_gate4_train_v1` — expanded-probe ×40 + parse drills (promote/store) ×25 + stratified promote/store from full-storage (max 2500) + direct anchors ×500.
- **Resume:** `real-v3-50m-full-v2-step-022800.pt` (Gate 3 pass), target **36000** (+13200 steps).
- **Train flags:** `--output-format tagged`, promote/store span weight 8, eval every 200 steps (action-prefix probe).
- **Ship bar unchanged:** Gate 4 expanded — parse/schema ≥0.95, action ≥0.85.

```powershell
python psm-model\scripts\runpod_ctl.py train-gate4 --deploy --target-steps 36000 --proxy-user <pod>-<suffix>
```

## 2026-06-07 — Gate 4 v1 training live (pod `zya02byfyqquyr`)

- **Proxy SSH:** `zya02byfyqquyr-644114c3@ssh.runpod.io`
- **Direct TCP:** `root@103.196.86.55 -p 33382` (when exposed)
- **Command:** `train-gate4 --pod-id zya02byfyqquyr --proxy-user zya02byfyqquyr-644114c3 --target-steps 36000`
- **Resume:** `real-v3-50m-full-v2-step-022800.pt` → absolute **36000** (+13200)
- **Curriculum:** `psm-50m-gate4-train-v1.jsonl` — **53,800 rows**
  - expanded full ×40: 36,800 (68.4%)
  - parse drills ×25: 12,000 (22.3%)
  - stratified promote/store: 2,500 (4.7%)
  - direct anchors ×500: 2,500 (4.7%)
- **tmux:** `psm-gate4`, log `/tmp/psm-gate4-train.log`, metrics `real-v3-50m-full-v2-gate4.metrics.jsonl`
- **GPU:** RTX 4090, torch 2.4.1+cu124
- **Note:** first kickoff hit broken `/workspace/PSM` clone; fixed bootstrap in `runpod_train_gate4.sh` (fresh clone when `psm-model/src` missing).
