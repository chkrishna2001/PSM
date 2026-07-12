"""v5n-dpo2 calibration DPO pairs: fix v5n-dpo's over-storing bias.

Corrected 2026-07-09 eval (--output-format json) showed hf-prod-v5n-dpo-qwen0.5b correctly
catches nearly every true store-case but also wrongly stores every ignore-worthy filler turn
on holdout-coding-agent-cases.json (0/6 correct on that gate's ignore cases). These 12 seed
rows are real, hand-labeled ignore-worthy turns sourced from the SAME candidate pool the gate
was built from (psm-model/prod-memory/results/coding-agent-candidate-turns.jsonl) but are
NOT any of the 17 cases in the gate itself -- the gate stays untouched for verification.
"""
from __future__ import annotations

from typing import Any

from prod_memory.build_v5o_storage_dpo_rows import _dpo_pair, _wrong_action_rejected
from prod_memory.row_validation import remember_target_from_input

# Each turn_text is cleaned of source mojibake (a display artifact in the raw Claude Code
# transcript, not meaningful content) but otherwise verbatim from a real candidate turn.
_IGNORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("codex-final-status-no-ambiguity",
     "Commit created. I'm checking the final status and latest commit so there's no ambiguity this time."),
    ("codex-staged-not-project-state",
     "Staged. Git also complained it cannot read the user-level ignore file, but repository staging "
     "succeeded; it's not part of this project state."),
    ("codex-checking-diff-mechanical-split",
     "I'm checking the final diff, then I'll commit the split. This should be a mechanical separation "
     "plus a methodology note, not a behavioral change."),
    ("claude-syntax-check-curriculum",
     "Now let's syntax-check and actually run the curriculum builder to verify the new variants "
     "generate valid pairs."),
    ("claude-check-consolidation-scoped",
     "Let me check whether memory consolidation/conflict detection is already scoped anywhere in the "
     "docs or code before we design that third adapter from scratch."),
    ("codex-staged-post17-sequence",
     "Staged. I'm committing this as the post-17 comparison/re-derivation experiment sequence."),
    ("codex-verification-passed-diff-shape",
     "Verification passed. I'm checking the diff shape and then I'll commit Experiment 27 and the "
     "Experiment 26 wording correction."),
    ("claude-cleaned-up-resume-pipeline",
     "Cleaned up. Now let's resume the pipeline at the retrieval step, reusing the already-complete DB."),
    ("claude-check-gpu-or-cpu-fallback",
     "Good catch - let me check whether the ingest process on the pod is actually using the GPU or "
     "silently falling back to CPU."),
    ("codex-staged-docs-only",
     "Staged. This is documentation only, so I'm not rebuilding the console app; I'll commit and then "
     "verify the tree is clean."),
    ("codex-checking-wording-duplicate",
     "That refinement is in. I'm checking the final wording for duplicate or contradictory "
     "prediction/falsification text, and making sure the admission-rule table is easy to find."),
    ("claude-holdout-gate-running-background",
     "The holdout gate is running in the background - this involves fresh ingestion of two full "
     "conversations plus retrieval and Cloudflare-based answer evaluation, likely longer than the "
     "training runs. I'll check back in a while."),
    # Second batch (2026-07-10, v5n-dpo3 iteration) -- expanding the seed pool from 12 to 28
    # after the first calibration attempt (v5n-dpo2) showed a small, generic set duplicated
    # 4x overcorrected the store/ignore threshold broadly rather than fixing the target bias.
    ("codex-staged-ignore-warning-external",
     "Staged. The usual global ignore warning is external to this repository and didn't affect staging."),
    ("codex-staged-ignore-warning-still-outside",
     "Staged. The global ignore warning is still outside the repo and not blocking anything."),
    ("codex-verification-passed-committing-experiment25",
     "Verification passed. I'm checking the final diff shape, then I'll commit Experiment 26 and "
     "the Experiment 25 conclusion correction together."),
    ("codex-cleaner-experimental-spine",
     "Good, the note has a cleaner experimental spine now: question, prediction, falsification, "
     "interpretation. I'll run one last status/phrase check and then summarize."),
    ("codex-build-passes-checking-status",
     "Build passes. The diff shows this branch already had broader uncommitted experiment-history "
     "changes relative to Git, so I'm checking status now and I'll avoid claiming anything I didn't "
     "actually add in this turn."),
    ("claude-retrain2-running-check-back",
     "Retrain #2 (scoped curriculum, `dpo_beta=0.3`) is running - I'll check back in about 10 "
     "minutes to verify completion, pull the adapter, delete the pod, and re-run the eval."),
    ("claude-check-callsites-generate-returntype",
     "Now let's check for other call sites of `HfGenerationSession.generate()` that need updating "
     "since I changed its return type."),
    ("claude-no-regressions-regenerate-curriculum",
     "Good, no regressions. Now let's check `build_v5q_dpo_curriculum.py` and regenerate the "
     "curriculum file."),
    ("claude-diagnostic-tasks-done-run-eval",
     "Both diagnostic-instrumentation tasks (#1, #2) are done and verified non-breaking. Now let's "
     "run the actual diagnostic eval - task #3."),
    ("codex-verification-passed-staging-experiment25",
     "Verification passed. I'm staging just these Experiment 25 changes and committing them now."),
    ("claude-module-imports-cleanly-check-callsites",
     "Module imports cleanly. Now let's check the other call sites I found (`eval_hf_two_pass.py`, "
     "`eval_hf_locomo.py`, etc.) still import fine since they reference `HfGenerationSession`/"
     "`open_hf_session`."),
    ("codex-commit-is-in-checking-tree",
     "Commit is in. I'm checking the working tree and latest commit so we end cleanly."),
    ("claude-check-ingest-process-alive",
     "Now let's check if the main ingest process is still alive and whether it'll benefit from "
     "this fix, or if it already gave up."),
    ("claude-hit-ceiling-zero-check-case-rows",
     "The `hit_token_ceiling_rate` is 0.0 across the board - let me check the actual case rows to "
     "see what happened with `workflow-runpod`/`technical-eslint` specifically."),
    ("claude-no-overlap-run-eval-stable-checkpoint",
     "No overlap confirmed. Now let's run the eval against the stable v5q-dpo checkpoint."),
    ("claude-final-sanity-check-checkpoint-score",
     "Let's do a final sanity check that the primary local checkpoint dir now scores 6/10."),
    # Third batch (2026-07-10, v5n-dpo4 round targeting the 2 remaining over-store misses
    # on the coding-agent gate: tests-pass-nextstep, docs-agree-sanity-check)
    ("claude-no-pods-agent-still-working",
     "Confirmed: no pods running (billing stopped). The background agent (early-stop generation "
     "fix, task #10) hasn't sent a completion notification yet - it's still working. I won't read "
     "its raw transcript directly per guidance; I'll report when it finishes."),
    ("claude-mapped-mechanics-waiting-agent",
     "Got the training/eval mechanics mapped out. Still waiting on the truncation-diagnosis agent "
     "before I draft the fix plan."),
    ("claude-new-commit-confirmed-cleanup",
     "New commit confirmed. Now let's replace the local corrupted copies and clean up temp restore "
     "directories."),
)


def _ignore_expected(reasoning: str) -> dict[str, Any]:
    return {
        "action": "ignore",
        "memory": None,
        "facts": [],
        "indexables": [],
        "reasoning": reasoning,
    }


def build_v5n_dpo2_ignore_rows() -> list[dict[str, Any]]:
    """Seed rows: real transient/filler coding-agent turns labeled ignore."""
    rows: list[dict[str, Any]] = []
    for row_id, text in _IGNORE_SEED_TURNS:
        rows.append({
            "id": f"v5n-dpo2-ignore-{row_id}",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_calibration",
            },
            "expected": _ignore_expected(
                "Transient status/next-step narration with no durable project fact, decision, or "
                "preference worth remembering."
            ),
            "source": "v5n_dpo2_calibration",
        })
    return rows


def build_v5n_dpo2_calibration_pairs() -> list[dict[str, Any]]:
    """DPO pairs: chosen=ignore (correct), rejected=synthetic wrong store (over-storing fix)."""
    pairs: list[dict[str, Any]] = []
    for row in build_v5n_dpo2_ignore_rows():
        remember_target = remember_target_from_input(row["input"])
        if not remember_target:
            continue
        expected = row["expected"]
        rejected_raw = _wrong_action_rejected(expected)
        pair = _dpo_pair(
            str(row["id"]),
            remember_target,
            expected,
            rejected_raw,
            source="v5n_dpo2_calibration",
            variant="wrong_action_overstore",
        )
        if pair:
            pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# v5n-dpo4 (2026-07-10): under-storing fix -- real genuine-finding turns the
# checkpoint currently misclassifies as ignore (parse-failure-finding,
# run-complete-hf, hallucination-not-truncation, powershell-path-resolution on
# the coding-agent gate). Chosen=store_episodic (correct), rejected=ignore (the
# model's actual wrong behavior) -- the inverse direction of the over-store fix
# above, using _wrong_action_rejected's existing store-side branch.
# ---------------------------------------------------------------------------
_STORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("confirmed-6-10-baseline",
     "Confirmed - `6/10 effective_stored`, matching the true baseline. This is genuinely the "
     "pre-retrain checkpoint. Now let's properly upload it to HF and replace the corrupted local "
     "copies."),
    ("confirmed-hf-local-back-baseline",
     "Confirmed - HF and local state are both back to the safe 6/10 baseline. Both retrain "
     "attempts made real progress on the specific bugs, but destabilized the model overall each "
     "time: effective_stored dropped to 3/10 then 4/10, from the original 6/10."),
    ("revert-fully-verified",
     "Revert fully verified - HF hub and both local checkpoint dirs are back to the genuine 6/10 "
     "pre-retrain weights (new commit `c040d81` on HF, confirmed via direct eval)."),
    ("verify-pod-race-condition-pattern",
     "Confirmed same pattern as before - training started fine, the \"failed\" status is the same "
     "verify-pod race condition."),
    ("real-infra-bug-3-pids",
     "Confirmed - this is a real infra bug, not GPU/scheduling flakiness. Three different PIDs "
     "(15545, 15615, 15908) appeared across just 32 seconds, each living only ~1-2 seconds, and "
     "GPU memory stayed at 1MiB the entire time despite `--device cuda` being passed everywhere."),
    ("proxy-host-id-stale-cache",
     "The API still reports the original proxy host id (`64411220`), not the one you pasted "
     "(`64410f20`) - that's likely just a stale/cached suffix from the RunPod dashboard."),
    ("zero-hits-recall-plan-qwen",
     "Confirmed by grep - zero hits for `recall_plan`/`context_plan` across every training JSONL "
     "that's ever fed a Qwen HF profile (`v5b` through `v5q-dpo`). Qwen0.5B/0.6B has never been "
     "trained on recall/retrieval at all."),
    ("pod-already-up-billing",
     "Found it - pod `k0ik4lvjbu0tqp` / proxy user `k0ik4lvjbu0tqp-64411536` is already up and "
     "billing. I'll reuse it directly rather than deploying a duplicate."),
    ("experiment-6-result",
     "Added and committed Experiment 6. Commit: `8ac8a5a Experiment 6: interaction history "
     "contradiction`. Result: Frames observed: 18, Traces formed: 3. So the answer is no: "
     "interaction history alone does not let the trace separate itself."),
    ("pod-running-tmux-sync-started",
     "Found it - pod `udk4kfi83tx5st` (\"psm-hf-lora\"), RUNNING, $0.39/hr. The training tmux "
     "session (`psm-hf-lora`) and HF sync loop (`psm-hf-sync`, uploads every 120s) both started "
     "successfully on the pod."),
)


_V5N_DPO5_ROW_IDS = frozenset({
    "claude-no-pods-agent-still-working",
    "claude-mapped-mechanics-waiting-agent",
    "claude-new-commit-confirmed-cleanup",
})


def build_v5n_dpo5_calibration_pairs() -> list[dict[str, Any]]:
    """DPO pairs: ONLY the 3rd-batch ignore seeds (the ones v5n-dpo3's frozen curriculum never
    saw), isolated from the understore direction this time -- v5n-dpo4 proved combining both
    directions in one round overcorrects. This round tests the over-store fix alone, resumed
    from v5n-dpo3."""
    pairs: list[dict[str, Any]] = []
    for row in build_v5n_dpo2_ignore_rows():
        if not any(row["id"].endswith(row_id) for row_id in _V5N_DPO5_ROW_IDS):
            continue
        remember_target = remember_target_from_input(row["input"])
        if not remember_target:
            continue
        expected = row["expected"]
        rejected_raw = _wrong_action_rejected(expected)
        pair = _dpo_pair(
            str(row["id"]),
            remember_target,
            expected,
            rejected_raw,
            source="v5n_dpo5_calibration",
            variant="wrong_action_overstore",
        )
        if pair:
            pairs.append(pair)
    return pairs


def _store_expected(content: str, reasoning: str) -> dict[str, Any]:
    return {
        "action": "store_episodic",
        "memory": {"content": content, "type": "episodic"},
        "facts": [],
        "indexables": [],
        "reasoning": reasoning,
    }


def build_v5n_dpo4_store_rows() -> list[dict[str, Any]]:
    """Seed rows: real genuine-finding coding-agent turns labeled store_episodic."""
    rows: list[dict[str, Any]] = []
    for row_id, text in _STORE_SEED_TURNS:
        rows.append({
            "id": f"v5n-dpo4-store-{row_id}",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_calibration",
            },
            "expected": _store_expected(
                text,
                "A concrete verified finding, diagnosis, or infra state check worth remembering, "
                "not transient status narration.",
            ),
            "source": "v5n_dpo4_calibration",
        })
    return rows


def build_v5n_dpo4_calibration_pairs() -> list[dict[str, Any]]:
    """DPO pairs: chosen=store_episodic (correct), rejected=synthetic wrong ignore (under-storing fix)."""
    pairs: list[dict[str, Any]] = []
    for row in build_v5n_dpo4_store_rows():
        remember_target = remember_target_from_input(row["input"])
        if not remember_target:
            continue
        expected = row["expected"]
        rejected_raw = _wrong_action_rejected(expected)
        pair = _dpo_pair(
            str(row["id"]),
            remember_target,
            expected,
            rejected_raw,
            source="v5n_dpo4_calibration",
            variant="wrong_action_understore",
        )
        if pair:
            pairs.append(pair)
    return pairs
