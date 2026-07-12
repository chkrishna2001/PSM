"""Storage v7 data expansion: round 2 of hand-labeling the real candidate pool
(`coding-agent-candidate-turns.jsonl`), after user feedback that storage -- the foundational
adapter everything else depends on -- deserves real, quality-first data investment rather than
another small patch or a capacity-ceiling test outside the project's model-size budget.

This batch adds ~35 more real, hand-labeled examples on top of the ~90 from build_storage_v6_rows.py
(44 pre-existing + ~50 mined this session), sourced from:
  - The remainder of the same Codex research-experiment session (rollout-...019f3bbb...,
    ...019f3d01..., ...019f3d37...) -- genuine quantified findings ("Experiment N: X/Y result")
    vs. genuine process narration ("build clean, committing"), a different but analogous
    judgment call to the ML/infra domain already covered.
  - A newly-discovered untouched Codex session (rollout-2026-04-19T07-51-08-019da594...,
    PDF-extraction workflow) -- adds a third genuinely distinct real-content domain.

Deliberately capped at a modest addition (not hundreds) to avoid over-indexing on one narrow
research topic's phrasing patterns, per the lesson from earlier DPO rounds that low-diversity
seed sets cause overfitting to surface features rather than the underlying judgment.
"""
from __future__ import annotations

from typing import Any

STORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("experiment10-fragmentation-reduced",
     "Experiment 10 did reduce active fragmentation: 6 active hypotheses became 3 by moving "
     "only assignment edges. The uncomfortable part is that it may have over-collapsed "
     "structural observations into one resolved hypothesis, so the earned conclusion is "
     "narrow: relationship revision reduces fragmentation, not that it preserves the right "
     "explanations."),
    ("raw-attack-broke-mechanism",
     "The attack did what we wanted: it broke the current mechanism. Vertical and horizontal "
     "raw-grid observations got mixed into one unresolved hypothesis because shared empty "
     "pixels overwhelmed the actual shape. I'll record this as a falsification of "
     "position-wise similarity for perception-like input, not as a failure of the ontology yet."),
    ("experiment12-partial-success",
     "Experiment 12 partially succeeded: formed observations separate V and H, but they "
     "collapse dense checkerboard outcomes because the formation rule only sees adjacent "
     "same-direction pairs. That is useful: transient observation formation can help, but the "
     "naive local structure rule discards too much information. Also, we've crossed the "
     "500-line ceiling."),
    ("mechanical-split-verified",
     "The mechanical split is verified: runtime is 150 lines, experiment harness is 517 lines. "
     "I'm adding a short methodology note to the ledger so this commit is clearly about "
     "reproducibility and core-runtime inspectability, not a new mechanism."),
    ("experiment13-alternation-preserved",
     "The result is better than the prediction in the useful way: transition counts preserve "
     "alternation, but they collapse vertical and horizontal sparse lines into one hypothesis "
     "under the existing similarity rule. So Experiment 13 didn't give us \"better observation "
     "formation\"; it gave us a clean complementary measurement: alternation survives, "
     "orientation/local continuity is not preserved enough."),
    ("experiment13-numbers-verified",
     "Verified. The key Experiment 13 numbers are: mixed sparse/alternating = 0, mixed V/H = "
     "1. That's the clean result: alternation preserved, orientation not preserved."),
    ("experiment15-collisions-found",
     "Experiment 15 found three same-basis/different-arrangement collisions: shifted vertical "
     "lines, shifted horizontal lines, and phase-inverted checkerboards. That proves the "
     "current basis cannot reconstruct arrangement, though it also raises the important "
     "nuance: some arrangement loss may be desirable invariance unless future explanations "
     "need it."),
    ("experiment16-meaningful-result",
     "Now the result is meaningful: the identical-basis A/B source becomes contradictory, "
     "then rejection creates another mixed A/B source trace. That is explanatory damage: the "
     "system cannot route different futures because the basis erased the needed distinction."),
    ("experiment17-cost-vs-influence",
     "This is a better failure than expected. Even the 1-bit phase offset makes A and B "
     "different as observations, but the hypothesis matcher still assimilates them because "
     "that bit is drowned by the 56 existing basis bits. So Experiment 17 exposed a new "
     "distinction: information cost is not the same as influence under similarity."),
    ("experiment18-broke-useful-direction",
     "Experiment 18 broke in the useful direction: flat comparison reproduced Experiment 17's "
     "A/B mixing, while identity-preserving comparison got mixed A/B source hypotheses down to "
     "0 and correct predictions up to 7/7."),
    ("experiment19-survived-attack",
     "Experiment 19 survived the first generalization attack: synthetic context/signal "
     "benchmark, same comparison conditions, same outcome pattern. I'm recording this "
     "narrowly: second supporting benchmark, still not an accepted principle because we do "
     "not yet have a benchmark designed to break it."),
    ("experiment20-adversarial-worked",
     "Good, the adversarial benchmark did its job: separated comparison over-separated noisy "
     "same-family observations and lost to flat comparison. I'm recording this as a failure of "
     "the experimental condition, not as a reason to add weights or patch the mechanism."),
    ("experiment21-clean-flip",
     "Experiment 21 produced the clean flip we wanted: in one conflict M1-only predicts "
     "perfectly while M2-only fails, and in the other conflict the roles reverse. I'm "
     "recording that as a narrow result: relevance was inferable from explanatory success in "
     "this constructed diagnostic, not solved in architecture."),
    ("experiment22-clean-numbers",
     "The run is clean, and the interesting bit held: inferred comparison gets 7/7 in both "
     "transfers, equal separated comparison collapses to 0/3, and the deletion test drops to "
     "2/7. I'm going to write that into the research docs so the result is reproducible as an "
     "experiment, not just console output."),
    ("experiment23-partial-failure",
     "The run gives the expected partial failure: the old selected identity cannot transfer "
     "into a renamed independent set, while target-local prior success still produces 7/7. "
     "I'm recording that as a failure of name-bound transfer, not a failure of Experiment 22."),
    ("experiment24-reuse-vs-rederivation",
     "That's the result we wanted to test: reuse fails at 2/7, re-derivation gets 7/7. I'm "
     "going to update the docs to make 23/24 say \"re-derived relationship\" instead of "
     "\"transferable quantity.\""),
    ("experiment25-structural-break",
     "Experiment 25 did break the structural analogy: source/position reuse is 1/15, "
     "single-channel local reconstruction is 1/15 and 3/17, while full interaction comparison "
     "is the only improved condition at 6/15. I'm documenting that as a partial failure, not a "
     "new mechanism."),
    ("experiment26-cross-family-signal",
     "This is a useful outside-family result: single locations mix A/B and get 3/11, paired "
     "spatial relation gets 5/9 with no A/B mixing. It is not spectacular; it is a first "
     "cross-family survival signal, not an architectural promotion."),
    ("commit-post17-sequence-confirmed",
     "You were right. I committed the accumulated post-17 work.\n\nCommit:\n\n```text\n"
     "4f63166 Record comparison experiments after experiment 17\n```\n\nIncluded the "
     "experiment harness changes and docs for the comparison/relevance/re-derivation sequence "
     "through Experiment 24. I also ran `dotnet build` before committing; it passed cleanly. "
     "Working tree is clean now."),
)

IGNORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("build-clean-check-diff-then-commit",
     "Build is clean. I already ran the harness after adding Experiment 15; I'll check the "
     "diff shape, then commit."),
    ("experiment16-patched-in-running",
     "Experiment 16 is patched in. It uses two phase-inverted checkerboards that have "
     "identical current basis values but lead to different 56-bit futures, repeated enough "
     "times to test whether the shared source hypothesis becomes contradictory."),
    ("experiment17-comparing-candidates",
     "Experiment 17 is now comparing candidates by cost and whether they resolve the "
     "Experiment 16 failure. I'll build and run it; I expect phase offset to win on this "
     "benchmark, but the run gets the vote."),
    ("docs-agree-pause-final-check",
     "The docs now agree on the pause: Phase 1A is closed as originally conceived, and "
     "Experiment 18 is not implementation-ready until the matcher assumptions are named. I'll "
     "do one final diff/status check so I can give you a clean summary."),
    ("build-clean-final-status-targeted-diff",
     "Build is clean after the ledger update. I'm doing a final status and targeted diff "
     "check so I can give you the exact result without drowning you in the full harness "
     "output."),
    ("output-wording-aligned-final-check",
     "That last output wording is now aligned with the caution. I'll do one final build and "
     "status check."),
    ("lab-notebook-precise-final-check",
     "The lab notebook now says the important thing precisely: two supporting benchmarks, "
     "zero break-focused tests. I'm doing one final build/status check and then I'll "
     "summarize the Experiment 19 result."),
    ("ledger-reflects-contradiction-final-check",
     "The ledger now reflects the new contradiction: flat comparison ignores small decisive "
     "evidence; veto-style separated comparison overreacts to noisy evidence. I'll do a final "
     "build/status check and then summarize the result."),
    ("filtered-run-grabbing-context",
     "The filtered run confirms the new sections execute, but it hid the numeric lines too "
     "aggressively. I'm grabbing just the Experiment 22/23 block with context so I can verify "
     "the reported scores exactly."),
    ("staging-whole-tree-single-commit",
     "Build is clean. I'm staging the whole working tree now and making a single commit for "
     "the accumulated experiment history."),
    ("staged-committing-post17-sequence",
     "Staged. I'm committing this as the post-17 comparison/re-derivation experiment sequence."),
    ("commit-created-checking-status-clean",
     "Commit created. I'm checking status so we know the tree is clean and there isn't "
     "anything left dangling."),
    ("pulling-pdf-to-text-first",
     "I'm pulling the PDF into text first so I can review the proposal directly, then I'll "
     "summarize the idea and turn it into concrete options we can execute in this repo."),
    ("pdftotext-not-installed-switching",
     "`pdftotext` isn't installed, so I'm switching to a quick local PDF extraction path and "
     "then I'll read through the paper content."),
    ("python-launcher-unusable-checking-runtimes",
     "The default `python` launcher isn't usable in this shell context, so I'm checking the "
     "available local runtimes and will use whichever can extract the PDF without leaving the "
     "machine."),
)


def _store_expected(content: str) -> dict[str, Any]:
    return {
        "action": "store_episodic",
        "memory": {"content": content, "type": "episodic"},
        "facts": [],
        "indexables": [],
        "reasoning": "A concrete verified finding, diagnosis, or quantified result worth "
        "remembering, not transient status narration.",
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


def build_storage_v7_new_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, text in STORE_SEED_TURNS:
        rows.append({
            "id": f"v7-store-{row_id}",
            "task": "remember_llm_response",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_v7",
            },
            "expected": _store_expected(text),
            "source": "storage_v7",
        })
    for row_id, text in IGNORE_SEED_TURNS:
        rows.append({
            "id": f"v7-ignore-{row_id}",
            "task": "remember_llm_response",
            "input": {
                "conversation": text,
                "operation": "remember_llm_response",
                "source_kind": "coding_agent_v7",
            },
            "expected": _ignore_expected(),
            "source": "storage_v7",
        })
    return rows
