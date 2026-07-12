"""Storage v6: a much larger, hand-labeled real-content SFT set, mined from the same
575-turn candidate pool (`coding-agent-candidate-turns.jsonl`, real Claude Code + Codex
sessions) that earlier rounds only sampled ~44 rows from for small DPO patches.

Rationale: v5n-dpo3 (0.65 action_match, current prod default) has been patched 6 times via
small (12-60 example) DPO rounds resumed on top of each other (v5n-dpo -> dpo2 -> dpo3 ->
dpo4/5/6). The last three attempts (dpo4/5/6) all net-regressed or broke even -- pushing the
over-store/ignore boundary in isolation costs store-side accuracy at roughly a 1-for-1 rate
regardless of how the round is scoped. That is a sign this is at a ceiling for further DPO
nudges on a chain of small patches, not a sign the underlying task is unsolvable.

This file is a full-scale ground-up SFT set instead: ~50 new real, hand-labeled examples
(29 store, ~21 more ignore beyond the existing pool) sourced from two genuinely distinct real
domains in the candidate pool (Claude Code / RunPod-ML session, and a separate Codex / C#
research-experiment session) -- deliberately diverse phrasing and topic, not templated. Intent
is to retrain fresh (no resume-adapter from the DPO chain) mixing this with the existing real
v5n curriculum, giving the model many more genuine examples of the exact judgment call
(concrete finding/diagnosis/result worth remembering vs. transient status/process narration)
rather than another small contrastive nudge.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# STORE-worthy: concrete findings, diagnoses, verified results, decisions with specifics.
# Real text, verbatim (source-cleaned of markdown/mojibake only), from unused turns in
# coding-agent-candidate-turns.jsonl (Claude Code session 79c7a744.../e8bce25e...,
# Codex session rollout-2026-07-07T04-40-06-019f3bbb...). None overlap with
# holdout-coding-agent-cases.json or any existing seed bank.
# ---------------------------------------------------------------------------
STORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("parse-failure-token-inflation",
     "I now have the full picture — the report caps `raw_output` at 500 chars for display "
     "(`eval_hf_grounding.py:188`), so the true generation length for the two failures is "
     "unknown; the fact schema also redundantly triples `subject`/`value`/`value_text` per "
     "fact, which is a plausible token-inflation culprit distinct from truncation being "
     "unfixable."),
    ("plan-agent-file-checks-out",
     "The Plan agent's file checks out against direct reads — `_make_fact()` does hardcode "
     "`value_text=value` at label_from_assistant.py:80-96, `generate()` already computes "
     "`new_tokens` making token-count instrumentation trivial, and `complete_json` in "
     "storage_rewards.py:45 is ready to reuse."),
    ("call-sites-break-keep-signature",
     "That breaks 11 other call sites across the codebase that expect `generate()` to return "
     "a plain string. I need to keep `generate()`'s signature unchanged and add a separate "
     "method for the instrumented path instead."),
    ("workflow-runpod-not-truncation",
     "This overturns the original hypothesis — `workflow-runpod` only generated 222 tokens "
     "(not anywhere near the 768 ceiling) and technical-eslint only 286."),
    ("not-truncation-bug-hallucination",
     "Found it — this is not a truncation bug at all. The model appears to be hallucinating "
     "malformed structure for `tags`/`product`/`range`, not running out of budget."),
    ("variants-wired-48-rows",
     "Both new variants are wired correctly and get the `failure_copies=6` boost for both "
     "target fixtures (48 rows each = 8 variants × 6 copies)."),
    ("curriculum-regenerated-900-1248",
     "Curriculum regenerated (900 → 1248 rows), manifest confirms both new variants and "
     "`workflow-runpod` now included in the failure-fixture boost list."),
    ("false-alarm-race-condition-gpu87",
     "Good news — the initial exit code 1 was a false alarm from a race condition (verify-pod "
     "ran before the tmux session was fully up). Checking directly now shows training **is** "
     "actively running: `job_state: \"training\"`, GPU util 87%, CUDA healthy."),
    ("training-completed-tmux-exited",
     "Training actually **completed successfully** — the pod's tmux session naturally exited "
     "after finishing, which is why `verify-pod`'s health-check reported it as \"failed\". The "
     "log shows checkpoint-80 fully uploaded and \"uploaded adapter to "
     "krishnach7262/psm-prod-memory-hf\"."),
    ("commits-confirmed-timestamp-match",
     "Confirmed — commits from 2026-07-05 09:54 UTC (today, minutes ago), matching this exact "
     "retrain run. The adapter is safely verified on HF hub."),
    ("pod-deleted-status-204",
     "Pod `2kk9a1eymp5d72` deleted (status 204 — confirmed terminated, no more billing)."),
    ("regression-6-10-to-3-10",
     "This is a clear regression — `effective_stored` dropped from 6/10 to **3/10**, and "
     "previously-passing fixtures (`plan-01-handoff`, `cursor-01-summary`, "
     "`workflow-review-pr`) now fail with *new* errors."),
    ("pre-retrain-adapter-intact",
     "Good, the pre-retrain adapter is intact locally. Now let's restore it to HF hub."),
    ("curriculum-scoped-978-rows",
     "Tests pass. Curriculum is scoped down (978 rows, ~9% growth over the ~900 baseline vs. "
     "the previous 39% growth), `dpo_beta` raised to 0.3 for a more conservative update, and "
     "the two new variants restricted to the `promote_semantic`/technical-workflow rows where "
     "the real bugs occurred."),
    ("new-commits-training-finished",
     "New commits landed at 10:54-10:55 UTC — training attempt 2 finished and synced."),
    ("pod-created-json-parsing-failed",
     "A pod actually got created (`k0ik4lvjbu0tqp`, billing started) but the script's own "
     "JSON-parsing logic failed to extract the pod ID/proxy user from the deploy response."),
    ("190-turns-burned-failures",
     "The main ingest process is alive but already burned 190 turns as failures before my fix "
     "landed (`ingested 190 | stored=0 ignored=0 failed=190`). Model loading is now succeeding "
     "post-fix, but this run's data is already corrupted."),
    ("process-auto-restarted-retry",
     "The killed process auto-restarted via the shell script's own retry wrapper — and since I "
     "deleted the DB right before, it's now starting fresh with the working code."),
    ("rate-estimate-zero-failures",
     "Based on the observed rate (~11-12s/turn, one progress line per 10 turns), it's "
     "genuinely working, not stuck, and producing clean data (zero failures since the fix)."),
    ("ingested-880-stored-92",
     "The actual job on the pod is fine — it's still healthy and progressing: `ingested 880 | "
     "stored=92 ignored=788 failed=0`, same processes alive, zero failures, DB actively being "
     "written."),
    ("ingestion-finished-1032-104",
     "Ingestion itself finished — `seen: 1032, stored: 104, ignored: 928, failed: 0` (matches "
     "the full LoCoMo turn count, confirming apples-to-apples comparison with v5n-dpo's known "
     "797/1032)."),
    ("real-finding-parse-failures-not-ignore",
     "This is the real finding — most \"ignore\" decisions aren't the model choosing to "
     "ignore, they're **parse failures being fail-safed into ignore**."),
    ("918-of-927-parse-failures",
     "This is a major reframe: **918 of 927 \"ignore\" decisions (99%) are actually parse "
     "failures being safely defaulted to ignore — not genuine model decisions.** Only 9 turns "
     "were genuinely judged \"nothing to store.\""),
    ("6-10-matches-baseline-parse-climbed",
     "6/10 effective_stored — matches the v5n-dpo baseline, up from 3/10 on v5q, and "
     "parse_valid climbed from 0.4 to 0.6. And notably this ran entirely on CPU locally, no "
     "pod needed."),
    ("cat-dataset-compresses-5-into-1",
     "Now the cat dataset compresses 5 experiences into 1 trace while the sudden shift still "
     "forms 2 traces, which is exactly the pressure test we wanted."),
    ("invariant-experiment-succeeds",
     "With only the threshold adjusted, the invariant experiment now does the thing we wanted: "
     "`111100`, `111010`, and `111001` collapse into `111???`."),
    ("prediction-held-collapsed-families",
     "The prediction held: it collapsed both families into one `111???` trace, then attached "
     "both futures equally. That is a clean falsification of \"invariants alone are enough\"."),
    ("interaction-history-didnt-split",
     "The result is useful in a very specific way: interaction history did not split the "
     "trace; it only biased the single trace toward the more frequent future."),
    ("second-trace-appears-rejection-works",
     "This is the clean result: the original `111???` trace stays unresolved, and a second "
     "`111???` trace appears from later rejected observations, with a consistent future. We "
     "also learned a side constraint: trace identity cannot be the same thing as perception "
     "hash once rejection exists."),
)

# ---------------------------------------------------------------------------
# IGNORE-worthy: transient status/process narration, forward-looking "let me check X next",
# acknowledgments. Real text, verbatim, from the same two sessions, different turns than
# above and than the existing 31-row ignore-seed bank.
# ---------------------------------------------------------------------------
IGNORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("read-core-docs-first",
     "Let me read the core PSM docs to understand the overall system, not just the "
     "model-training memory."),
    ("check-recall-side-scoped",
     "Let me check what's already built for the recall/retrieval side of the adapter, since "
     "v5n-dpo/v5q-dpo only train the storage half — I want to confirm whether recall-plan "
     "training data or curriculum exists before scoping this."),
    ("research-first-before-cost",
     "This involves spinning up RunPod training runs (real cost) and multiple steps, so let "
     "me research the specific failure mode first and put together a concrete plan before "
     "executing anything."),
    ("waiting-both-agents",
     "Both explore agents are running in the background — one diagnosing the exact truncation "
     "failure mode, the other mapping the train → eval → holdout-gate → promote command "
     "mechanics. I'll wait for both before designing the fix plan."),
    ("verify-ambiguity-before-fix",
     "Let me verify one critical ambiguity myself — whether the model actually hit the "
     "768-token ceiling or stopped early — before finalizing the fix approach, since that "
     "changes what the fix needs to be."),
    ("plan-agent-designing",
     "The Plan agent is designing the concrete fix sequence now — I'll review it and finalize "
     "the plan file once it's back."),
    ("extract-helper-next",
     "Now I'll extract a reusable `is_json_complete` helper in `storage_rewards.py` and wire "
     "everything into `eval_hf_grounding.py`, `eval_grounding.py`, and `_print_case_table.py`."),
    ("verify-full-file-tests",
     "Now let's verify the full file and check nothing else broke, then run the existing test "
     "suite for this module if one exists."),
    ("move-to-task-2",
     "This looks correct and non-breaking now. Let's move to task #2 — wire the new metrics "
     "into `aggregate_metrics()` and `_print_case_table.py`."),
    ("running-diagnostic-eval",
     "Everything's in place. Running the diagnostic eval at 768 tokens first (this will take "
     "a few minutes on CPU for a 0.5B model across 10 fixtures)."),
    ("check-case-rows-ceiling",
     "The `hit_token_ceiling_rate` is 0.0 across the board — let me check the actual case "
     "rows to see what happened with `workflow-runpod`/`technical-eslint` specifically."),
    ("syntax-check-curriculum-builder",
     "Now let's syntax-check and actually run the curriculum builder to verify the new "
     "variants generate valid pairs."),
    ("scheduled-checkin-safety-net",
     "I've scheduled a check-in shortly as a safety net, but I'll primarily wait for the "
     "automatic task-completion notification."),
    ("check-pod-status-directly",
     "Let me check the current pod status directly to see if training actually finished "
     "successfully despite the script's exit code."),
    ("verify-cmd-exists",
     "This has a `cmd_verify` that's exactly what I need first — checks HF for the adapter "
     "without needing pod SSH. Let's run that."),
    ("run-local-cpu-eval",
     "24 files pulled. Now let's run the local CPU eval against the retrained adapter."),
    ("stop-pod-no-longer-needed",
     "Right — we've already verified the adapter is safely on HF and pulled it locally, so the "
     "pod is no longer needed. Let me stop/delete it now before continuing."),
    ("verify-revert-worked",
     "Restored. Let's verify the revert worked by re-pulling and re-running the eval to "
     "confirm we're back to 6/10."),
    ("build-restore-dir-reeval",
     "Now let's build the full restore directory (adding the tokenizer files which are "
     "unchanged since they're base-model artifacts) and re-eval locally to confirm this is "
     "genuinely the 6/10 checkpoint before re-uploading to HF."),
    ("regenerate-reduced-failure-copies",
     "Now let's regenerate the curriculum with reduced `failure-copies` and check the "
     "resulting row count and scoping."),
    ("wait-not-check-prematurely",
     "The retrain job is deploying now with the scoped curriculum and higher `dpo_beta`. I'll "
     "wait for it to finish rather than checking prematurely (last time an early check "
     "produced a false-failure signal from a race condition)."),
    ("check-hf-directly-instead-of-waiting",
     "Good, training is actively running (`job_state: \"training\"`, GPU util 65%). Since this "
     "task-notification woke me up already (10 min elapsed since deploy), let me check HF hub "
     "directly for completion instead of waiting further."),
    ("stale-data-wait-longer",
     "That's stale data from the reverted run — the new commit hasn't landed yet. Training is "
     "still genuinely in progress (GPU at 65%). Let me wait a bit longer before checking "
     "again."),
    ("waiting-final-restore-verification",
     "Waiting for the final restore verification to complete before summarizing where things "
     "stand."),
    ("need-cloudflare-credentials",
     "I need Cloudflare credentials (`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`) for the "
     "answer-accuracy step of the holdout gate, and I don't see how they're normally sourced "
     "in this codebase."),
    ("get-fp16-diagnostic-plan",
     "Let me test whether float16 precision alone (independent of GPU vs CPU) causes the "
     "degradation, by forcing fp16 weights on CPU for the same fixture eval — this avoids "
     "needing another paid GPU pod for this specific diagnostic."),
    ("scope-substantial-task-first",
     "This is a substantial task (data sourcing + labeling design), so let me scope it "
     "properly before building anything."),
    ("verify-no-overlap-before-eval",
     "Now let's verify the schema/JSON validity and confirm zero overlap with any training "
     "curriculum data before running the eval."),
    ("check-pod-really-gone",
     "Confirmed — no pods running, nothing to check on or bill for there. Let's check the "
     "background early-stop agent."),
    ("sanity-check-sample-row",
     "It works and produced 900 DPO pairs. Let me sanity-check a sample row and confirm the "
     "jsonl was written correctly."),
    ("wire-training-profile-next",
     "Data looks correct — chosen rows have valid enum values, rejected rows carry the exact "
     "hallucinated variants v5q actually produced. Now let me wire the training profile into "
     "`_run_hf_lora.py`, following the exact pattern of `v5n-dpo`/`v5o-dpo`."),
    ("kicked-off-background-training",
     "I've kicked off the Phase 3 v5q-dpo training launch in the background — it's deploying "
     "a new RunPod GPU pod, syncing the enum-DPO curriculum/build scripts ahead of git, and "
     "starting an 80-step DPO run. I'll let you know when it completes or if it hits an "
     "error."),
    ("truncated-output-find-pod-id",
     "The `tail -100` in my launch command truncated the earlier deploy output (pod ID, GPU "
     "selection) and the final verify-pod step. Let me find the pod ID so I can check on it "
     "properly."),
    ("net10-installed-building-probe",
     "`.NET 10.0.301` is installed, so we can use the exact runtime you wanted. I'm making "
     "Commit-1-ish code with a tiny extra prediction probe."),
    ("build-run-check-line-count",
     "The first version is now in place and still small enough to hold in your head. I'm "
     "going to build and run it, then I'll check the line count and git diff so we know "
     "exactly what landed."),
    ("bin-obj-appeared-gitignore",
     "`bin/` and `obj/` appeared from the run, as expected. I'm going to ignore and remove "
     "those generated artifacts so the repository still reads like the experiment, not the "
     "tooling residue."),
    ("committing-first-experiment",
     "Since you explicitly framed each experiment as a commit, I'm going to commit this as "
     "the first runnable experiment with an honest message: local trace associations over "
     "unlabeled frames."),
    ("staging-needs-git-access",
     "Staging needed `.git` write access and is now done. I'm committing next; the "
     "line-ending warnings are Git's normal Windows CRLF note, not a code issue."),
    ("final-status-check-repro",
     "The first experiment commit exists now. I'm doing one final status check so I can tell "
     "you whether the tree is clean and give you the exact command/output shape to reproduce "
     "it."),
    ("patch-blunt-running-to-see",
     "The patch is intentionally blunt: no confidence, no hierarchy, no controller. Now I'm "
     "running it to see whether the cat variations compress to one trace and whether the A/B "
     "shift creates a second structure instead of overwriting the first."),
    ("build-clean-committing-next",
     "Build is clean and we're still at 216 C# lines. I'm committing this as the next "
     "experimental step: equality replaced with similarity, plus the first forgetting "
     "benchmark."),
    ("updating-ledger-adding-experiment",
     "Yes, that contradiction is sharper than the last one: averaging compressed the "
     "examples, but it may have invented a thing that was never observed. I'll update the "
     "ledger first, then add the smallest invariant-discovery experiment without expanding "
     "the architecture."),
    ("running-all-experiments-check-regressions",
     "The trace now has exactly one new self-description: `Resolved` or `Unresolved`. I'm "
     "running all experiments to make sure old successes stay intact and Experiment 7 marks "
     "only the contradictory structural trace."),
    ("updating-ledger-provisional-capability",
     "The detector behaves like a warning light, not a repair tool: the ambiguous `111???` "
     "trace becomes `Unresolved`, while the ordinary traces remain `Resolved`. I'm updating "
     "the ledger to capture that as a provisional capability and the next open question."),
    ("no-recovery-fragmentation",
     "Experiment 9 gives the expected but important answer: no recovery. Permanent rejection "
     "causes fragmentation, including duplicate outcome traces, so we've found the next "
     "contradiction without adding another mechanism."),
)


def _store_expected(content: str) -> dict[str, Any]:
    return {
        "action": "store_episodic",
        "memory": {"content": content, "type": "episodic"},
        "facts": [],
        "indexables": [],
        "reasoning": "A concrete verified finding, diagnosis, or result worth remembering, not "
        "transient status narration.",
    }


def _ignore_expected() -> dict[str, Any]:
    return {
        "action": "ignore",
        "memory": None,
        "facts": [],
        "indexables": [],
        "reasoning": "Transient status/next-step narration with no durable project fact, "
        "decision, or preference worth remembering.",
    }


def build_storage_v6_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, text in STORE_SEED_TURNS:
        rows.append({
            "id": f"v6-store-{row_id}",
            "task": "remember_llm_response",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_v6",
            },
            "expected": _store_expected(text),
            "source": "storage_v6",
        })
    for row_id, text in IGNORE_SEED_TURNS:
        rows.append({
            "id": f"v6-ignore-{row_id}",
            "task": "remember_llm_response",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_v6",
            },
            "expected": _ignore_expected(),
            "source": "storage_v6",
        })
    return rows
