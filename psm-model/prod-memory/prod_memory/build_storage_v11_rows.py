"""Storage v11 data expansion: targeted fix for the residual under-storing pattern found in
v10's eval (0.7059, 12/17 -- all 5 misses were false negatives: expect store, model said
ignore). Per-case inspection of the coding-agent gate showed a consistent pattern: genuine
findings/decisions/results phrased tersely (no explicit "found"/"confirmed" framing keyword)
get miscategorized as transient status narration.

Mined from the same provenance-checked candidate pool (`coding-agent-candidate-turns.jsonl`)
used to build the eval gate and v6/v7 training data, after excluding (via fuzzy text match,
not exact string match -- an encoding difference between the pool and the fixture caused 3
false negatives on a first exact-match pass) every turn already used as an eval case or
already-used training row. All STORE examples below are genuine findings/decisions/results
stated compactly; all IGNORE examples are genuinely vague/redundant/thin turns matched for
similar terse phrasing, to avoid re-teaching a blunt "always store terse sentences" bias (the
mirror-image mistake of the original over-storing bug this project already fixed once).
"""
from __future__ import annotations

from typing import Any

STORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("diagnostic-tasks-verified-nonbreaking",
     "Both diagnostic-instrumentation tasks (#1, #2) are done and verified non-breaking. "
     "Now let's run the actual diagnostic eval -- task #3."),
    ("confirmed-6-10-pretrain-baseline",
     "Confirmed -- `6/10 effective_stored`, matching the true baseline. This is genuinely the "
     "pre-retrain checkpoint. Now let's properly upload it to HF and replace the corrupted "
     "local copies."),
    ("revert-verified-6-10-weights",
     "Revert fully verified -- HF hub and both local checkpoint dirs are back to the genuine "
     "6/10 pre-retrain weights (new commit `c040d81` on HF, confirmed via direct eval)."),
    ("pod-stopped-billing-halted",
     "Pod `udk4kfi83tx5st` is now stopped (`EXITED` as of 16:01:53 UTC) -- billing on it has "
     "stopped. Let me check what the earlier SSH session actually captured before responding "
     "further."),
    ("eval-report-on-hf-artifact-rule-satisfied",
     "Eval report is now on HF alongside the adapter -- artifact rule satisfied retroactively "
     "(adapter + metrics + prod eval all on HF, verified locally). Let me update the project "
     "memory with the real Phase 3 outcome."),
    ("ledger-two-findings-rejection-and-identity",
     "I've updated the ledger with both findings: rejection works provisionally, and perception "
     "identity stopped being enough for trace identity. Now I'll build, inspect the final diff, "
     "and commit the experiment."),
    ("architecture-contract-added-no-impl-details",
     "The architecture contract is now added and contains no implementation details, just the "
     "baseline ontology and constraints. I'm reviewing it and the diff before committing so we "
     "don't accidentally smuggle in algorithm language."),
    ("runtime-split-status-matches-intended-move",
     "The status looks exactly like the intended move: old root runtime files deleted, new "
     "`Runtime/` and `Experiments/` directories added, and the ledger updated. The diff stat "
     "undercounts the new files because they're untracked, so I'll check line counts and rerun "
     "the suite before committing."),
    ("ledger-records-falsified-claim-limited-result",
     "Ledger now records the falsified claim and the accepted-but-limited result. I'm updating "
     "the roadmap so the next step is not \"add arrangement,\" but test whether missing "
     "arrangement causes explanatory damage."),
    ("experiment16-recorded-arrangement-necessary",
     "I recorded Experiment 16 as the moment arrangement crosses from independent to necessary, "
     "while still avoiding the leap to a specific arrangement algorithm. Next I'll update the "
     "ledger with the rejected claim and provisional acceptance."),
    ("h21-recorded-wording-changed",
     "The new hidden assumption is recorded as H21, and I changed the wording from \"noise\" to "
     "\"task-irrelevant independent evidence\" where it matters. I'm doing a quick consistency "
     "check so the note doesn't accidentally imply a mechanism."),
    ("h22-rejected-question-moved",
     "The ledger now records H22 as rejected for the diagnostic and moves the active question to "
     "whether relevance is really about interactions. I'm doing one more build/run check plus a "
     "diff review for mistakes or accidental overreach."),
    ("ledger-records-27-composition-break",
     "The ledger now records 27 as the composition break: individually successful distinctions "
     "stopped being sufficient when composed. I'm aligning roadmap and matcher notes with the "
     "same non-decomposability language."),
    ("experiment13-wired-row-column-transitions",
     "Experiment 13 is wired in now: row transitions plus column transitions, encoded as an "
     "immutable formed observation. I'm building and running it to see whether the result "
     "matches the prediction or gives us a nastier contradiction."),
    ("experiment-wired-distinguishes-two-cases",
     "The experiment is wired in, and it distinguishes two cases: cross-set transfer using the "
     "old selected identity, and target-local prior success using the new evidence set's own "
     "outcomes. I'm adding the tiny constructor for that independent evidence set now."),
)

IGNORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("confirmed-same-pattern-already-known",
     "Confirmed same pattern as before -- training started fine, the \"failed\" status is the "
     "same verify-pod race condition. Let's check current live status directly."),
    ("no-overlap-confirmed-thin",
     "No overlap confirmed. Now let's run the eval against the stable v5q-dpo checkpoint."),
    ("stale-proxy-host-id-transient",
     "The API still reports the original proxy host id (`64411220`), not the one you pasted "
     "(`64410f20`) -- that's likely just a stale/cached suffix from the RunPod dashboard. Let "
     "me use the confirmed one."),
    ("committed-final-check-transient",
     "Committed. I'll do a final clean-tree/history check and then I'll answer your question "
     "directly."),
    ("ledger-result-recorded-vague",
     "Ledger result is recorded. I'm rerunning the build and then I'll check the final diff "
     "surface."),
    ("implementation-docs-aligned-vague",
     "The implementation and docs are aligned now. I'm doing the final build/run check so the "
     "recorded numbers match the code after the wording changes."),
    ("everything-named-new-frame-vague",
     "Everything is now named in the new frame. I'm doing the final verification pass: build, "
     "focused run output, and a quick search for stale \"transferable\" links."),
)


def _store_expected(content: str) -> dict[str, Any]:
    return {
        "action": "store_episodic",
        "memory": {"content": content, "type": "episodic"},
        "facts": [],
        "indexables": [],
        "reasoning": "A concrete verified finding, diagnosis, decision, or result worth "
        "remembering, even though it is stated compactly without an explicit "
        "found/confirmed framing keyword.",
    }


def _ignore_expected() -> dict[str, Any]:
    return {
        "action": "ignore",
        "memory": None,
        "facts": [],
        "indexables": [],
        "reasoning": "Vague, redundant, or already-known status narration with no new durable "
        "fact, decision, or result worth remembering.",
    }


def build_storage_v11_new_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, text in STORE_SEED_TURNS:
        rows.append({
            "id": f"v11-store-{row_id}",
            "task": "remember_llm_response",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_v11",
            },
            "expected": _store_expected(text),
            "source": "storage_v11",
        })
    for row_id, text in IGNORE_SEED_TURNS:
        rows.append({
            "id": f"v11-ignore-{row_id}",
            "task": "remember_llm_response",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_v11",
            },
            "expected": _ignore_expected(),
            "source": "storage_v11",
        })
    return rows
