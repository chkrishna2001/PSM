# PSM Project Memory (facts, state, and current best)

Last updated: 2026-07-02

## Goal
Fine-tune a **Qwen 0.5B / 0.6B** LoRA adapter so the model can act as the **PSM storage model** and emit **PSM-compatible StorageDecision memory/facts** for real assistant conversations via **PSM CLI JSON** (`operation=remember_llm_response`, `conversation=[assistant]`).

**Deploy bar:** fixtures **≥7/10** `effective_stored` with a healthy full case table **AND** holdout retrieval ≥ baseline (`v5n-dpo`). `effective_stored` alone is never sufficient. LoCoMo is eval/holdout only — not a training target.

**Eval gate:** every train run must follow `.cursor/rules/psm-train-eval-gate.mdc` — full storage case table + holdout ingestion + holdout retrieval on non-training data, before promoting a checkpoint. No blind training.

## Canonical I/O (locked for v5o+)
- **Train / prod eval / CLI probe:** flatten assistant text via `remember_target_from_input()` → `PROD_STORAGE_USER_PREFIX` + text (`hf_prompts.row_messages` / `storage_inference_messages`).
- **Do not** use `to_model_input()` for probes/eval (rewrites assistant-only as `User:` — train skew).
- Helpers: `storage_llm_response_from_input()`, `storage_inference_messages_from_input()`.

## What we did (high signal)
1. **v5n** (v3 soup + ~200 Gemma conversation anchors): prod fixtures **5/10** `effective_stored` — best so far, not deploy-ready. HF: `hf-prod-v5n-qwen0.5b`.
2. Diagnosed stall: curriculum dilution (~90% v3), train/probe I/O skew, no eval-gated training, JSON truncation at inference.
3. **v5o path implemented (2026-07-02):**
   - Merged **545** Gemma teacher rows → `hf-prod-conversation-gemma-all.jsonl` (193 ignore / 288 promote / 64 episodic). Codex+gemini labeling complete: **347** rows in `hf-prod-conversation-gemma-codex-gemini.jsonl` (4 skipped).
   - **Conversation-only SFT:** `hf-prod-v5o-sft.jsonl` — **1217** rows (no v3), heavy fixture boost.
   - **StorageDecision DPO:** `hf-prod-v5o-dpo.jsonl` — **4109** pairs (wrong_action, truncated, malformed, ungrounded) with rule rewards (`storage_rewards.py`).
   - Profiles: `v5o-sft` (300 steps, resume v5n) → `v5o-dpo` (100 steps DPO, resume v5o-sft).
   - Probe aligned: `probe_locomo_hf_storage_cli_shape.py` uses `remember_target` path.
4. **v5o-sft trained/evaluated:** RunPod pod `nrcz4u4kf72raf`, 300 steps from v5n. Prod fixtures regressed to **3/10** — **do not run v5o-dpo from this adapter**.
5. **v5p** (v5n + expanded conversation, no simplify): pod `4z287r5sh66ag4`, 200 steps → **0/10** (parse_valid 0.3). Longer JSON hits 384-token ceiling.
6. **v5n-dpo** (fixture-heavy DPO, 172 pairs, 80 steps, resume v5n): pod `bkznph58fvswql` → **5/10** at 384 tokens (tie v5n). **6/10** at **768 tokens** (`technical-api` fixed; parse_valid 1.0). HF: `hf-prod-v5n-dpo-qwen0.5b`.
7. **max_new_tokens sweep (2026-07-02):** v5n stays **5/10** at 768 and 1024. v5n-dpo **6/10** at 768 only. Remaining failures: `plan-01-handoff` (guard bleed), `workflow-runpod` (parsed but not grounded).

## Current best state
| Checkpoint | Prod fixtures (10) | Notes |
|------------|---------------------|-------|
| **v5n-dpo @ 768 tok** | **6/10** effective_stored | **Current prod default** adapter + `PROD_STORAGE_MAX_NEW_TOKENS=768` |
| v5n / v5n-dpo @ 384 | 5/10 effective_stored | Default eval ceiling truncates JSON |
| v5n @ 768/1024 | 5/10 | Token bump alone does not help v5n |
| v5o-sft | 3/10 effective_stored | Worse than v5n; DPO gated off |
| v5p | 0/10 effective_stored (parse_valid 0.3) | Conversation soup + no simplify |
| v5h | 0/10 (local harness) | JSON parse OK, session-metadata bias |
| v5o-dpo | not trained | needs a better SFT base |

## RunPod / artifact rule
Pods must not be deleted until adapter + metrics + prod eval are on HF and verified locally. **No pods running** (2026-07-02). Before `--deploy`, pass `--delete-pod-id` or reuse `--pod-id` to avoid duplicate `psm-hf-lora` billing.

## Key scripts
- `psm-model/prod-memory/scripts/build_v5o_prod_curriculum.py` — build SFT + DPO JSONL
- `psm-model/prod-memory/prod_memory/storage_rewards.py` — rule reward for StorageDecision JSON
- `psm-model/prod-memory/prod_memory/build_v5o_storage_dpo_rows.py` — DPO pair builder
- `psm-model/scripts/_run_hf_lora.py --profile v5o-sft` / `v5o-dpo`
- `psm-model/scripts/probe_locomo_hf_storage_cli_shape.py` — aligned CLI probe

## Next action (infra before more training — see psm-train-eval-gate.mdc)
1. **Build holdout smoke** (no GPU train yet): frozen holdout split (LoCoMo convs never in v5n+ curriculum, or synthetic unseen turns) + single-adapter ingest/retrieval smoke pointed at `v5n-dpo`.
2. **Wire v5n-dpo single-adapter JSON into end-to-end path** — `buildHfPsmRuntime`/`answer-evaluate.ts`/`ingest-psm-model.ts` currently default to old v5k two-pass adapters. Add single JSON adapter runtime option.
3. **Establish v5n-dpo retrieval baseline** on holdout; hook smoke into `_watch_hf_lora.py` `_finish()`.
4. **Then** targeted DPO on `plan-01-handoff` / `workflow-runpod` (guard/grounding) + fact quality — only after real signal exists.
5. **`--delete-pod-id` before `--deploy`** — duplicate deploy billing risk.
