# PSM 50M — end of day handoff (2026-06-11)

**Read first tomorrow:** this file → [2026-06-10-parse-recovery-plan.md](2026-06-10-parse-recovery-plan.md) → `.cursor/rules/runpod-auto-delete.mdc`

**Nothing running on RunPod** — all LoCoMo / eval pods deleted. **No GPU billing.**

**Gate 4 ship bar met @ step 48000.** LoCoMo n=25 **not completed** (local runs killed; no valid `*-results.json`). Product integration (daemon + async hook) **not ship-ready**.

---

## Where we are (snapshot)

| Item | Status |
|------|--------|
| **Best checkpoint** | **48000** — parse **95.40%** (871/913), action **95.2%**, `gate_passed: true` |
| Prior best | 45000 @ 94.3% |
| Registry | `psm-model/checkpoints/gate4-checkpoint-registry.json` |
| Eval report (48000) | On pod logs; pull to `psm-model/checkpoints/gate-eval/gate4-full-expanded-step-048000.json` if missing locally |
| HF model repo | `subbu83/psm-50m-mixed-v1-run` (chinna token via `o chinnahftoken`) |
| HF dataset repo | `chkrishna2001/psm-50m-action-mixed-v1` (v5 curriculum + repair pack uploaded) |
| Local 48000 triple | `psm-model/checkpoints/real-v3-50m-full-v2-step-048000.{pt,tokenizer.json,meta.json}` (~601 MB `.pt`) |
| RunPod pods | **None** |
| LoCoMo n=25 | **Incomplete** — partial dirty DB only (see below) |
| CLI default checkpoint | `step-048000` in `config.ts`, LoCoMo ingest scripts |
| `psmModel.enabled` | Still **false** — opt-in via `--psm-model` |

### Metrics (do not conflate)

| Metric | Value | What it measures |
|--------|-------|------------------|
| Gate 4 `parse_valid_rate` | **95.4%** | Tagged parse on **913 curated eval probes** |
| Product repair (offline) | **~99.89%** on 45000 report failures | `storage_decision_repair.py` on eval rows — **not** live until fully wired in daemon |
| LoCoMo ingest | **No score** | 24/25 failed on first RunPod attempt (pre-repair); local attempts killed |

---

## What we accomplished today

### Training & eval (RunPod)

1. **Phase 1b r2 succeeded** — pod `cv75h94udtjn31`, v5 curriculum (52 failing probes ×400 boost), batch 8, L4, 45000→48000.
2. Post-train expanded eval ran; **48000 promoted** to registry best.
3. HF triple for 48000 verified; pod deleted per policy.
4. **Run 1 (45000→48000 on `4u7blgwnqle1u5`) lost weights** — CRLF broke `runpod_upload_gate4.sh`; fixed with `.gitattributes`, LF-normalized `*.sh`, `sed` strip in train script.

### Phase 2 repair (local, zero GPU)

- Built `psm-model/src/psm_model/storage_decision_repair.py` — deterministic tagged repair + fail-safe `ignore`.
- Measured **99.89% product parse** on 45000 eval failures (51 repaired, 1 fail-safe).

### Product integration (this session — **local, uncommitted**)

| Change | File(s) | Notes |
|--------|---------|-------|
| Repair wired into remember path | `remember_cli.py` | `apply_product_boundary()` after decode; handles `repair_remember_json` without re-decode |
| Repair prompt fix | `psm-model-runtime.ts` | Accepts `repair_remember_json` payloads |
| Ignore canonicalization | `storage_decision_repair.py` | `action: ignore` → `memory: null` + reasoning |
| Warm remember server | `remember_server.py`, `remember-server.ts` | Load 50M **once**; default on (`PSM_REMEMBER_SERVER=0` to disable) |
| `max_new_tokens` | remember path | **1200 → 384** (tagged outputs are short) |
| Device policy | `device_policy.py` | **GPU only on RunPod** (`PSM_RUNPOD=1`) or `PSM_ALLOW_LOCAL_GPU=1`; local `auto` → CPU |
| Local LoCoMo script | `run-ingest-psm-model.ps1` | **CPU only**, `PSM_FORCE_CPU=1`, blocks `-Device cuda` |
| LoCoMo npm fix | `runpod_locomo.sh` | `--ignore-scripts` (skip GGUF postinstall); Node 22; absolute python path |
| Python path fix | `ingest-psm-model.ts` | Bare `python3` not resolved under repo root |
| CRLF / shell | `.gitattributes`, all `psm-model/scripts/*.sh` | LF enforced |

**Tests:** `tests/test_remember_cli.py`, `tests/test_device_policy.py` pass. `npm run build` passes after `npx tsc -b --force` (incremental build can leave stale `psm-model-runtime.js` — **force rebuild psm-core after TS changes**).

---

## LoCoMo — what happened (all runs failed / killed)

### RunPod (`ac6g14kxg1ckyq`, deleted)

| Attempt | Issue | Outcome |
|---------|-------|---------|
| 1 | Node 20 — no `node:sqlite` | Failed before ingest |
| 2 | npm postinstall GGUF — `Invalid gpuLayers value: 0` | Fixed with `--ignore-scripts` |
| 3 | `python3` → `/workspace/PSM/python3` ENOENT | Fixed absolute `/usr/bin/python3` |
| 4 | `PSM_RUNPOD` missing → **CPU at 100%**, 0% GPU | Fixed `export PSM_RUNPOD=1` |
| 5 | GPU run with repair not wired | **0 stored, 1 ignored, 24 failed** (strict parse) |
| Pod | Deleted | Billing stopped |

### Local (laptop — **do not repeat full n=25 on CPU**)

Four background runs (task IDs 368767, 641953, 529370, 858106) — all exit **4294967295** (killed). Causes:

1. First attempts used `auto` → **CUDA on laptop** → crash / thermal.
2. Cold **subprocess per turn** reloaded **~601 MB checkpoint** each time → **~6–7 min/turn**.
3. Warm `remember_server` added but Windows ready handshake initially broken (stderr → stdout JSON `{"ready":true}`).
4. Stale `src/psm-core/dist/psm-model-runtime.js` — server not used until `tsc -b --force`.

**Partial artifact:** `benchmark/locomo/results/locomo-psm-model-step-048000-n25.db` — **45 decision rows** from overlapping runs (21 `error`, 24 `ignore`). **Delete before clean rerun.** No `*-results.json`.

**Bench after warm server (short prompts, CPU):** ~**22 s/turn** → n=25 ≈ **10 min** ideal; LoCoMo prompts are much longer.

---

## Product architecture — what we ship vs what LoCoMo tested

### Intended product

```
Main LLM responds → hook fires remember() → 50M decides store/ignore → SQLite
```

- **Hook:** `runHookRemember` in `psm-cli` — after assistant response (not blocking user chat in ideal design).
- **Daemon:** `psm-cli daemon` — keeps runtime warm; hook prefers daemon when `daemon.enabled` + `autostart`.
- **50M path:** `PsmModelRuntime` → `remember_cli` / `remember_server` (not Qwen GGUF).

### Gaps before ship (priority order)

| # | Gap | Impact |
|---|-----|--------|
| **P0** | **Daemon uses `NodeLlamaRuntime` (Qwen GGUF)** — not `PsmModelRuntime` | Hook does not use 48000 50M today |
| **P0** | **Hook is synchronous** — `await service.remember()` blocks | Bad UX if decode takes 20s+ |
| **P1** | LoCoMo n=25 never scored | No retrieval hit@k for 48000 |
| **P1** | Eval report 48000 not pulled locally | Missing `gate4-full-expanded-step-048000.json` locally |
| **P2** | HF intermediate steps still on HF (45200–48000 sprawl) | `GATE4_KEEP_BEST_ONLY` incomplete |
| **P2** | `psmModel.enabled` still false | Opt-in only |
| **P3** | Daemon lacks `--psm-model` flags | Document / wire |

### Latency targets (ship)

| Environment | Target |
|-------------|--------|
| RunPod GPU | Sub-second to few seconds per remember |
| Local CPU (warm daemon/server) | **1–3 s** typical turn; LoCoMo-sized prompts longer |
| Local laptop | **Smoke n=3–5 only** — not full benchmarks |

**50M is the right CPU model** — slowness was **integration** (cold reload × 25), not model size.

---

## Tomorrow — prioritized steps

### 1. Ship path (P0) — ~2–4 h

```text
[ ] Wire daemon createRuntime() to PsmModelRuntime when config.psmModel.enabled / env
[ ] Daemon loads remember_server once at startup (same as PsmModelRuntime pool)
[ ] Hook remember: fire-and-forget to daemon (return immediately; audit log async result)
[ ] Smoke: psm hook remember on 3 turns, confirm <5s warm CPU
[ ] Commit integration changes (repair + server + device policy + daemon)
```

**Files:** `src/psm-cli/src/daemon.ts`, `src/psm-cli/src/index.ts` (`runHookRemember`), `src/psm-core/src/psm-model-runtime.ts`.

### 2. LoCoMo score (P1) — RunPod only

```powershell
cd C:\Users\chkri\source\repos\PSM
o runpodkey
o chinnahftoken; $env:HF_TOKEN = (Get-Clipboard -Raw).Trim()
$env:PSM_HF_MODEL_REPO = 'subbu83/psm-50m-mixed-v1-run'

# Deploy L4 pod (cheap smoke) or use existing warm pod policy from runpod_ctl
# Sync scripts: psm-model/scripts/runpod_locomo.sh (has ignore-scripts, PSM_RUNPOD=1, Node 22)

python psm-model/scripts/run_pod_script.py --proxy-user <pod>-<suffix> --timeout-sec 7200 `
  --env LOCOMO_WAIT_FOR_EVAL=0 --env LOCOMO_LIMIT=25 `
  psm-model/scripts/runpod_locomo.sh

# After completion: pull results
python psm-model/scripts/pull_pod_dir.py --proxy-user <pod>-<suffix> `
  benchmark/locomo/results/locomo-psm-model-step-048000-n25-results.json

# Delete pod after verify
```

**Before rerun locally (smoke only):**

```powershell
Remove-Item benchmark\locomo\results\locomo-psm-model-step-048000-n25.db -ErrorAction SilentlyContinue
.\benchmark\locomo\run-ingest-psm-model.ps1 -Limit 5 -Device cpu
npx tsc -b src/psm-core --force   # if TS changed
```

### 3. Registry / HF hygiene (P2)

```powershell
# Pull 48000 eval report if missing
# HF prune intermediate steps (keep 48000 best only) per gate4_checkpoint_registry prune-hf-keep-best
```

### 4. Optional training

- Only if LoCoMo or product metrics regress.
- 48000 @ 95.4% already passes gate; **do not train before LoCoMo + daemon ship** unless blocked.

---

## Decision rules (unchanged)

- **Promote** only if expanded parse > prior best (48000 is current bar: **95.4%**).
- **LoCoMo** unblocked at ≥95% parse — run benchmark, don't defer further.
- **Local laptop:** CPU only, no CUDA (`PSM_FORCE_CPU=1`, `-Device cpu`).
- **RunPod:** always `PSM_RUNPOD=1`, `--proxy-user`, verify GPU util, delete pod after HF verify.
- **Never delete pod** if registry best missing HF triple.

---

## Credentials & ops gotchas

| Secret | How |
|--------|-----|
| RunPod API | `o runpodkey` |
| HF model (subbu83) | `o chinnahftoken` → `$env:HF_TOKEN` |
| HF dataset (chkrishna) | `~/.cache/huggingface/token` or `DATASET_HF_TOKEN` |
| SSH | Always `--proxy-user <pod_id>-<suffix>@ssh.runpod.io` — never SCP to proxy; use tar-push |
| `PSM_RUNPOD=1` | Required on pod for CUDA; omit locally |
| `PSM_FORCE_CPU=1` | Local dev / LoCoMo script sets this |
| `PSM_REMEMBER_SERVER=0` | Disable warm server (fallback to cold spawn per call) |

---

## Key paths

```text
psm-model/checkpoints/gate4-checkpoint-registry.json     # best=48000
psm-model/checkpoints/real-v3-50m-full-v2-step-048000.pt
psm-model/src/psm_model/remember_cli.py                  # repair boundary
psm-model/src/psm_model/remember_server.py               # warm inference
psm-model/src/psm_model/storage_decision_repair.py
psm-model/scripts/runpod_locomo.sh
psm-model/scripts/runpod_eval_gate4_expanded.sh
benchmark/locomo/run-ingest-psm-model.ps1                # local CPU smoke
src/psm-core/src/psm-model-runtime.ts
src/psm-core/src/remember-server.ts
src/psm-cli/src/daemon.ts                                  # still GGUF — fix tomorrow
docs/psm-model/2026-06-10-parse-recovery-plan.md
```

---

## Timeline (2026-06-11)

1. Expanded eval 45000 → 94.3%, promoted.
2. Phase 1b train 48000 → 95.4%, promoted, gate passed.
3. CRLF/upload fixes; HF v5 curriculum uploaded.
4. psm-cli defaults → 048000; repair pass built (offline).
5. LoCoMo RunPod — multiple infra failures, then 24/25 strict-parse fails; pod deleted.
6. Device policy + repair wired into `remember_cli`; `remember_server` added.
7. Local LoCoMo n=25 — laptop GPU crash, then hour-long CPU marathon; all runs killed.
8. Identified ship gap: daemon/hook not on 50M; sync blocking; LoCoMo ≠ product path.

---

## One-line summary for tomorrow

**48000 passes Gate 4; wire daemon + async hook to warm 50M, then LoCoMo n=25 on RunPod for the retrieval score — do not benchmark on laptop CPU.**
