# PSM coding-domain benchmark ("coding LoCoMo") — design + build plan

**Status as of 2026-07-29: harness built and run end-to-end for real on RunPod (v16b vs v5 storage
candidates, both domain=coding, real `PsmMemory.Cli` product path).** Results:
`benchmark/coding-locomo/data/coding-locomo-answer-v16b.json` and `-v5.json`. This is a new
benchmark, not part of the existing `docs/plans/psm-context-aware-storage-training.md` /
`psm-conversational-v9-handoff.md` tracks. Read `project_psm_failure_taxonomy` memory first — this
benchmark exists specifically to give the coding domain the same kind of real-pipeline evidence
LoCoMo already gives conversational, since coding currently has none.

## First real result (2026-07-29, v16b vs v5 storage candidates)

Both models scored **identically**: 20 questions, 25% overall answer accuracy, by category:
single-hop 2/10 (20%), multi-hop 0/4, temporal 0/3, adversarial (no-hallucination) 3/3 (100%).

The exact match across two different storage adapter candidates is itself the finding: swapping the
storage adapter made zero measurable difference on this benchmark. That's strong evidence the
bottleneck for real coding-domain QA is **retrieval, not storage quality** — consistent with
`project_psm_failure_taxonomy` item #8 (retrieval/ranking-miss architecture gap, shared across
domains, still open — task #10). Multi-hop and temporal both scoring 0% (which require combining or
ordering facts across sessions) while adversarial scores perfectly (which only requires not
hallucinating, no retrieval needed) fits the same story: whatever's being stored isn't being surfaced
at answer time.

**Implication for shippability:** further storage-side curriculum fixes (task #8) are unlikely to move
this benchmark's score until the retrieval-miss gap (task #10) is investigated and fixed first. Task
#10 should be reprioritized ahead of task #8 based on this evidence.

## Why this is needed

The coding storage/retrieval-plan/consolidation adapters have never had a benchmark analogous to
LoCoMo. Every coding-domain check so far has been a small, one-off, hand-built probe (25 turns, 5
questions, a single real transcript) — useful for finding bugs, but not a repeatable, scalable
evidence source the way LoCoMo is for conversational. The user asked directly: coding needs its own
real benchmark before the coding adapters can be judged shippable.

## What we looked at and ruled out

Surveyed HF broadly for an existing fit:
- `SWE-bench/SWE-smith-trajectories`, `Inferact/codex_swebenchpro_traces` — real, high-quality
  multi-turn coding-agent trajectories, but each is a **single session resolving one isolated GitHub
  issue**. No multi-session narrative on one project over time, so no natural "what did we decide 3
  sessions ago" structure. Good as supplementary raw material if more volume is ever needed, not a
  ready-made benchmark.
- InfiniteBench `Code.Debug` (found via a memory-agent eval repo on HF) — a completely different task
  shape: given a huge raw code dump, find which function has an injected bug. That's long-context code
  *comprehension*, not remembering facts from agent *conversation* history. Not a fit for what PSM's
  coding adapters actually do.
- Nothing on HF is shaped like a genuine "coding LoCoMo" (multi-session, same project, facts
  established early and tested later).

## The data source: real multi-agent history on this actual project

Rather than adapt a mismatched external dataset, build this from real local session history — better
fit than anything external, because it's the literal real use case (an agent remembering *this*
project's facts across *real* sessions), and ground truth is verifiable against actual git history and
the project's own memory files (which already track a huge amount of verified "what was true when").

**Survey result (2026-07-29):** the PSM project itself has by far the richest real multi-agent,
multi-session coverage of anything available:
- **Claude Code**: 6 sessions, 94MB total, spanning 2026-07-04 through 2026-07-28 (24 days). One of
  these (`ad347195-...`, 24,993 lines) is enormous and covers a huge amount of ground alone. The
  in-progress current session (`c5b97e67-...`) should be EXCLUDED from the benchmark itself (still
  live, not a settled historical record) but is useful context for authoring ground truth.
- **pi**: 39 sessions under `~/.pi/agent/sessions/--C--Users-chkri-source-repos-PSM--/`.
- **Codex**: 23-43 sessions under `~/.codex/sessions/**/*.jsonl` mentioning PSM (exact count needs
  final dedup pass — the `Downloads/training-data/codex-sessions` copy is a strict subset of
  `~/.codex/sessions`, so treat `~/.codex/sessions` as the canonical source and skip the Downloads copy
  for this project).
- **Gemini**: 0 sessions under `Downloads/training-data/gemini-sessions` mention PSM — gemini worked on
  other projects, not usable for this benchmark.

Other projects with decent multi-agent coverage if more scale is needed later: AIInferenceRouter (36 pi
sessions), Organism (30 pi sessions), agent-context-card (18 pi sessions) — same three-agent-source
pattern likely applies, not yet surveyed in detail.

## Benchmark design (mirrors LoCoMo's structure)

- **"Conversation" = one project's real history.** PSM is conversation #1. Each real session (from any
  of the 3 agents) is one "session," analogous to LoCoMo's multi-session-per-persona-pair structure,
  ordered by real timestamp across ~24 days.
- **QA categories, matching LoCoMo's own category split** (so results are comparable in kind, not just
  in name):
  - **Single-hop**: answerable from one session alone (e.g. "what GPU did the v9 training run use?").
  - **Multi-hop**: requires combining facts from 2+ sessions (e.g. "how did the resume-checkpoint for
    conversational training change between v2 and v7, and why?").
  - **Temporal**: requires correct ordering/dating across sessions (e.g. "was the schema bug found
    before or after the epoch-count bug?").
  - **Adversarial / no-answer**: a plausible-sounding question with no real answer in the history,
    testing that the system doesn't hallucinate one (LoCoMo has this category; it's been a real PSM
    failure mode all session — worth testing directly).
- **Ground truth authoring**: cross-reference candidate QA pairs against real git history (`git log`)
  and this project's own memory files (which already record verified "what was true when" — a
  significant, unusual advantage over building LoCoMo-style ground truth from scratch), not just the
  raw transcript text alone.
- **Turn extraction**: reuse the same substantive-turn filtering approach already validated this
  session (assistant turns with real content, length-thresholded, normalized across the 3 different
  raw JSONL/JSON formats — Claude Code, pi, Codex each have their own schema, need one shared parser
  producing a common `{speaker, text, timestamp, session_id}` shape).

## Build plan

1. **DONE — normalizers for all 3 agent formats.** `scripts-scratch/extract_psm_sessions.py` handles
   Claude Code (`type:"assistant"`, `message.content[].type:"text"`), pi (`type:"message"`,
   `message.role:"assistant"`, `content[].type:"text"`), and Codex (`type:"response_item"`,
   `payload.type:"message"`, `payload.role:"assistant"`, `content[].type:"output_text"`) — normalized
   to a common `{text, timestamp, session_id, agent, source_path}` shape.
2. **DONE — extraction + windowing.** Extracted 6,549 substantive turns (80+ chars) across all real PSM
   sessions April–July; user chose to scope the first version to the 2026-07-04 → 2026-07-29 window
   (3,558 turns, 43 sessions, agents: claude-code 2,214 / codex 1,315 / pi 29) — the window with the
   best memory-file cross-reference coverage for verifying ground truth. The live/in-progress session
   at extraction time (`c5b97e67-...`) was deliberately excluded (not yet a settled historical record).
   Full turn pool saved to scratchpad (`psm_window_turns.json`) — not committed to the repo (large,
   derived, regeneratable from the script).
3. **DONE — first QA batch.** 20 QA pairs written to `benchmark/coding-locomo/data/psm-project-v1.json`
   (10 single-hop, 4 multi-hop, 3 temporal, 3 adversarial/no-answer) — every answer is either a direct,
   verified quote-backed fact from the extracted turn pool, or (for the adversarial category) an
   explicit, checked absence, never a fabricated or hedged answer. Two early drafts (a placeholder
   "[a second cause...]" and a hedged temporal answer) were caught and rewritten before finalizing —
   worth double-checking newly-authored ground truth for this exact failure mode before adding more.
4. **NOT STARTED — ingest+eval harness.** Analogous to
   `dist/benchmark/locomo/src/locomo-dotnet-benchmark.js` but for this new format — ingest the
   extracted turns via the real .NET CLI (`--domain coding`), then run the 20 questions through
   `recall`/`context` and score against ground truth. This is the next concrete step.
5. **NOT STARTED — run against the current coding candidates** (v5 storage, plus retrieval-plan/
   consolidation once those get their own fix rounds) as the first real, repeatable coding-domain
   benchmark result.

## Possible future expansion (not started, not blocking step 4/5)

- Scale beyond 20 QA pairs if the first run's signal is promising.
- Expand to the other rich projects (AIInferenceRouter: 36 pi sessions, Organism: 30, agent-context-
  card: 18) as additional "conversations," each its own project-scoped benchmark file.
