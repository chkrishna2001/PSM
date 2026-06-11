# PSM 50M — end of day handoff (2026-06-10)

**Read first tomorrow:** this file → [2026-06-10-parse-recovery-plan.md](2026-06-10-parse-recovery-plan.md) → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on RunPod** — pod `l5g83xswkat3dt` (`psm-gate4-batchfix`, RTX 3090 $0.46/hr, ~2h40m total ≈ $1.25) deleted after HF verify passed.

**One thing matters tomorrow: run the expanded eval on step-045000.** Training finished; the eval never ran (warm-path launch script has no eval-after step). Until that eval, we don't know if today's run worked.

> **RESOLVED 2026-06-11: parse 94.30% (861/913) — 45000 promoted to best, HF pruned to best-only, pod deleted.** See the [recovery plan](2026-06-10-parse-recovery-plan.md#phase-1-result--expanded-eval--45000-run-2026-06-11-pod-7o9sjibuetgf82-l4) for full results. Eval-only pods now have a dedicated script: `psm-model/scripts/runpod_eval_gate4_expanded.sh` (run via `run_pod_script.py`).

---

## Where we are

| Item | Status |
|------|--------|
| RunPod pods | **None** (billing stopped) |
| Best (frozen production candidate) | **42000 @ 88.2% parse** — unchanged, on HF |
| New checkpoint | **45000 trained, NOT evaluated** — all triples 042200→045000 on HF |
| Today's run | batchfix: resume 42k → 45k, v4 curriculum, **batch 16** (vs 1), constant **LR 1e-4** (vs 3e-4), structural loss 1.0 (neutral/off) |
| Ship bar | parse/schema ≥95%, action ≥85% (action already passes) |
| HF model repo | `subbu83/psm-50m-mixed-v1-run` (chinna token) |
| HF dataset repo | `chkrishna2001/psm-50m-action-mixed-v1` (chkrishna token — **tokens are disjoint**, see gotchas) |
| Registry | `pending_eval` = 45000, promote only if parse > 0.882 |
| LoCoMo | still deferred until parse ≥95% |

---

## Why today's run is different from micro v1–v3 (key findings)

Full forensics in [the recovery plan](2026-06-10-parse-recovery-plan.md). Summary:

1. **The bottleneck was the optimizer, not model capacity.** All prior runs used `--batch-size 1` (hardcoded), constant LR 3e-4, no schedule. 42k steps = ~42k samples = <0.5 epoch of the 104k-row v4 curriculum. Micro runs (400 steps × batch 1 = 400 samples) were gradient noise — that's the 86–88% churn.
2. **Failures concentrate in 5 messy probe families** (retention_decay_5k 43%, codex_sessions 50%, incremental_5k 31%, reviewed_v2 29%, hard-local-semantic 24%). **All `direct-*` families: 0% fail.** The fact-format drills from the 06-09 Proposal A target the already-solved direct style — don't weave in more of them.
3. **Raw outputs show content collapse, not format slips.** GPU re-eval of the 108 fails (all reproduce deterministically): the model emits word salad on these inputs (`F:User prefers cans.`, `Q:0.86,0.02,sesnt_ging_...`) with intact scaffolding. 28 runaways never reach `END`. The training set contains all 913 eval probes ×100 copies and the model still can't reproduce them → under-trained, not at capacity.
4. **Eval reports now capture `raw_output`** for parse-fail rows (`eval_generation.py` patched) — never design an intervention blind again.
5. Honesty note: the v4 curriculum anchors the eval probes ×100, so Gate 4 expanded is effectively a memorization test. That's the chosen bar; just don't read 95% as generalization.

---

## Tomorrow — first commands

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey   # API key (runpod_ctl auto-reads via opener)
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()   # model repo token
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'

# 1) Confirm no pods
python psm-model/scripts/runpod_ctl.py list-pods

# 2) Deploy + expanded eval of step-045000 (913 probes, ~1.5-2h on GPU)
#    Cold-deploy a pod (A5000 preferred; 3090 worked today), then on pod:
#    - fetch triple: hf download subbu83/psm-50m-mixed-v1-run psm-model/checkpoints/real-v3-50m-full-v2-step-045000.{pt,tokenizer.json,meta.json} --local-dir .
#    - fetch probes (chkrishna token!): hf download chkrishna2001/psm-50m-action-mixed-v1 probes/expanded-probe-v1-budget.jsonl --repo-type dataset --local-dir psm-model/data
#    - tmux with PSM_RUNPOD=1:
#      python3 -m psm_model.eval_checkpoint psm-model/checkpoints/real-v3-50m-full-v2-step-045000.pt \
#        psm-model/data/probes/expanded-probe-v1-budget.jsonl \
#        --output-format tagged --gate-mode expanded --device cuda > .../gate4-full-expanded-step-045000.json

# 3) Optionally also eval 44000/44600 if 45000 regressed (all triples are on HF)

# 4) Decision (pre-committed):
#    parse >= 90%        -> big win; consider 45k -> 48k continuation, same config
#    88.2% < parse < 90% -> promote as new best; plan Phase 1b (family-targeted curriculum)
#    parse <= 88.2%      -> discard weights, 42000 stays best; go Phase 1b or Phase 2 (see plan doc)

# 5) Update registry (best/pending_eval) + delete pod only after verify (now actually works)
```

### Decision rules (carried from plan, do not relitigate)

- 42000 stays the frozen production candidate until something **beats 88.2% on expanded eval**.
- Never resume from 42400/42800/43400 (micro perturbations).
- One intervention class per run. Today = optimizer. Tomorrow's Phase 1b (if needed) = curriculum reweight toward the 5 failing families, **not** more direct-style drills.
- Phase 2 (decode-boundary repair pass) recovers only single-field slips (~30–40 rows); it cannot fix word-salad content. Worth building in parallel, zero GPU.

---

## What happened today (timeline)

1. **Forensics (local, free):** analyzed `gate4-full-expanded-step-042000.json` → findings 1–3 above; wrote [recovery plan](2026-06-10-parse-recovery-plan.md).
2. **Tooling:** `raw_output` in eval reports; `build_parse_fail_subset.py`; `classify_parse_failures.py`; `push_pod_files.py`; `BATCH_SIZE`/`LEARNING_RATE`/`MIN_LEARNING_RATE` knobs wired through `runpod_ctl.py train-gate4` → both launch scripts → `psm_model.train`.
3. **Local CPU eval attempt:** killed after ~1h (runaway 1200-token generations are brutal on CPU). Lesson: generation-heavy evals on pod only.
4. **Pod deploy:** A5000 out of stock → RTX 3090 (`l5g83xswkat3dt`). Cold bootstrap failed on dataset fetches (token split, see gotchas); staged files via HF dataset repo instead.
5. **Raw eval on pod GPU (13 min):** 108/108 fails reproduce; classification = 60 Q-numeric / 53 F-pipe / 42 missing-R / 28 runaway; content = word salad.
6. **Train launch failures then success:**
   - tokenizer default pointed at non-existent 036000 file → pass `--tokenizer ...042000.tokenizer.json`
   - `--structural-loss-weight 0` crashes (trainer rejects 0; **1.0 = off**) → relaunched with 1.0
   - 42k→45k at batch 16: ~1.5h, GPU 55–100%, VRAM 23.9/24 GB (3090 is at the ceiling at batch 16; batch 8 is the OOM fallback)
7. **Shutdown:** HF sync tmux had died (uploads stopped at 43200) → fixed CRLF on pod scripts, ran final upload (51 files), verified all triples 042200–045000 + 042000 on HF, deleted pod with working verify.

---

## Gotchas fixed / discovered today (respect these tomorrow)

| Gotcha | Status |
|---|---|
| `huggingface_hub.file_exists(repo_id, filename)` args were swapped in `runpod_ctl.py` → **every** HF verify was a false negative (this is what forced yesterday's `--force-delete-pod`) | **Fixed in repo** |
| HF token split: chinna token = model repo only; chkrishna2001 token (`~/.cache/huggingface/token`) = dataset repo only. Cold bootstrap with one token fails dataset fetches with "Repository Not Found" | Workaround: stage via HF with the right token per repo. Consider one token with access to both |
| `expanded-probe-v1-budget.jsonl` + `parse-fails-step-042000.jsonl` now in dataset repo under `probes/` | Done |
| Proxy tar-push >10 MB times out (900s); direct TCP port unreachable from this network | Stage big files via HF |
| tmux without `PSM_RUNPOD=1` silently runs on pod CPU (eval did this once) | Always export it |
| `runpod_upload_gate4.sh` CRLF after Windows scripts-sync → `set: pipefail: invalid option` | `sed -i 's/\r$//' psm-model/scripts/*.sh` on pod after sync (still needs a permanent fix, e.g. `.gitattributes` or sync-side conversion) |
| Launcher's 15s `_verify_pod_job` fires during the multi-minute v4 curriculum load → false "FAILED" | Verify manually ~3 min after launch |
| Warm-path launch script has **no eval-after step** (cold script only) | That's why 45000 is unevaluated |
| HF sync tmux can die silently — uploads today stopped at 43200 until manual final sync | Check `tmux ls` on pod mid-run |

---

## Files changed today

| File | Change |
|---|---|
| `psm-model/src/psm_model/eval_generation.py` | `raw_output` (≤4000 chars) in reports for parse-fail rows |
| `psm-model/scripts/runpod_ctl.py` | `train-gate4 --batch-size/--learning-rate/--min-learning-rate`; **fixed swapped `file_exists` args** |
| `psm-model/scripts/runpod_start_gate4_train_only.sh`, `runpod_train_gate4.sh` | `BATCH_SIZE`/`LEARNING_RATE`/`MIN_LEARNING_RATE` env knobs |
| `psm-model/scripts/build_parse_fail_subset.py` | new — extract parse-fail probes from an eval report |
| `psm-model/scripts/classify_parse_failures.py` | new — bucket raw failure outputs (Q-slip / F-slip / missing-R / runaway) |
| `psm-model/scripts/push_pod_files.py` | new — tar-push helper (use only for <10 MB; HF for bigger) |
| `psm-model/checkpoints/gate4-checkpoint-registry.json` | `pending_eval` = 45000 with promote rule |
| `docs/psm-model/2026-06-10-parse-recovery-plan.md` | full diagnosis + plan + live-run log |
| `psm-model/data/direct-behavior-v1/parse-fails-step-042000.jsonl` | 108 failing probes |
| `psm-model/checkpoints/gate-eval/parse-fails-042000-raw.json` | raw outputs of the 108 fails @ 42k |

## Not doing until parse ≥95%

- LoCoMo ingest
- Product E2E wiring
