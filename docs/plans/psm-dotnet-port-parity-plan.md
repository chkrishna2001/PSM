# PSM .NET port parity plan

**Status:** Draft — not started. The conversational-storage DPO round this plan was gated on is now
**done and promoted to production** (2026-07-21 — see
[[project_psm_storage_detail_probe_findings]]: 35.7%→17.9% "X values Y" template rate, action-match
held steady, live in `gguf-runtime/v1`). This plan is unblocked; re-read it fresh, confirm it's still
accurate against the current codebase, and resolve the open questions below with the project owner
before starting any implementation.
**Owner:** PSM team
**Origin:** During the conversational-storage DPO fix work, two real TS→C# porting gaps surfaced ad
hoc (embeddings, chunking). Rather than keep finding gaps one at a time, a full audit was run
against the original TypeScript implementation (`src/psm-core/src/*.ts`, `src/psm-cli/src/*.ts`) to
produce a complete inventory before deciding what to build next.

---

## How to read this document

This is an **inventory + prioritization framework**, not a committed implementation plan. Several
items need an explicit decision from the project owner before scoping (see "Open questions"). Do
not start building anything here without re-reading this doc fresh and confirming it's still
accurate — the codebase moves fast in this project.

**Important caveat on the audit itself:** the audit agent ran in an isolated git worktree, which
only sees *committed* code. It flagged the embedding pipeline as "not found" — that is a false
alarm caused by the isolation, not a real regression. The embedding fix (`EmbeddingRuntime.cs`,
`MemoryStore` embedding methods, `PsmService` wiring) and both consolidation fixes
(`ConsolidationMergeRetentionMinRatio`, `ConsolidationSemanticMinScore`) are real, done, and present
in the actual working tree as of 2026-07-21 — just uncommitted. Lesson for next time: don't isolate
an audit agent in a session with significant uncommitted work, or commit first.

---

## Confirmed gaps (from the full audit)

Ordered roughly by how much they'd matter to a real user of PSM as a memory hook, not by ease of
implementation.

### 1. The CLI's agent-integration layer is entirely unported

`src/psm-cli/src/index.ts` has `hook recall|remember|session-start|session-end`, `install-agent`
(Claude Code/Codex/Gemini), daemon mode, `setup`, `config`, `export`/`import`/`backup`, `review`.
`dotnet/src/PsmMemory.Cli` only has the raw `remember`/`recall`/`context`/`show`/`conflicts`/`init`
primitives (plus a new `serve` NDJSON mode with no TS equivalent). **This is the layer that would
make PSM a working drop-in hook for a real coding agent today — right now only the engine exists,
not the integration.** Whether this matters depends on whether MCP (`PsmMemory.Mcp`) is considered a
sufficient integration surface on its own, or whether CLI-based hook installation is still needed
for some agents — **open question, see below.**

### 2. `context()` is not a distinct operation in C# — it's `recall()` with a lower threshold

The real TS `context()` (`service.ts:46-84`) does something meaningfully different from `recall()`:
it ranks facts alongside memories, then makes a **second LLM call** (`renderContext`, via
`buildContextRenderPrompt`/`parseContextRender`) that lets the model select and rewrite grounded
context into a single text block (`agent_context`), with a deterministic
(`fallbackAgentContextItems`) fallback if the render step fails to parse. That's the actual string a
real hook integration would inject into an agent's system/context. C#'s `ContextAsync` is literally
`PlanAndRankAsync` called with a different prompt and a lower min-score constant — no facts, no
render step, no `agent_context` field in the output at all. `renderAgentMemoryContext`
(`context.ts`) — the function that formats ranked items into that text block — also has zero C#
footprint.

### 3. `recall()` output has no facts/indexables/workflows

TS: `{ memories, facts, indexables, workflows }`. C#: `RecallResult { UserId, Query, Plan,
PlanFallback, Memories }` only. Facts and indexables *are* persisted to the store on write (`
MemoryStore.InsertMemoryFact`/`UpsertIndexable`), they're just invisible at read time — there's no
`Ranking`-equivalent scoring for them (`rankFacts`/`rankIndexables` in TS have no C# port at all).

### 4. Indexables are never auto-synthesized

`indexables.ts`'s `buildIndexablesForRemember` (workflow detection via header-pattern regex +
numbered-step extraction, mnemonic key/salience synthesis, fact-anchor synthesis) has zero C#
footprint. Confirmed via the phase-3 doc below: this was a deliberately designed, tested feature in
TS (`recall("review-pr")` returning a 5-step procedure was an actual passing exit criterion), not a
minor convenience. In C#, an indexable can only ever exist if the storage adapter spontaneously
emits a well-formed `indexables[]` array unprompted — and `PromptBuilder.BuildStoragePrompt`'s
system instruction only mentions `indexables[]` in one trailing clause, with no schema or examples,
unlike TS's more explicit prompt. In practice this likely means indexables/workflows barely exist in
the C# store today.

### 5. Temporal expression resolution is never run

`temporal.ts`'s `normalizeMemoryTemporalFields`/`normalizeFactTemporalFields` — TS calls these
unconditionally in `remember()` before storage guards run, resolving relative-time phrases
("yesterday", "last week") against `source_timestamp` into a concrete `resolved_time`. C# never
calls anything equivalent; `resolved_time`/`resolved_time_confidence` are populated only if the
model happens to put them directly in its JSON output, with no deterministic fallback. This quietly
undermines `Ranking.cs`'s own `temporalScore` term, which assumes these fields are reliably present.

### 6. `prompts.ts` was deliberately, substantially rewritten (not a bug, but a real behavior change worth knowing)

None of the TS prompt-builders are reproduced byte-for-byte — the C# adapters were trained on a
different (Qwen2 ChatML, terser) format, which is a legitimate and already-justified reason. But the
practical effect: **the storage prompt in C# never includes existing-memory context, source
metadata, or the user message** — TS's `buildStoragePrompt` embedded a `memory_store` array of up
to 20 existing memories directly in the prompt; C# defers *all* existing-memory awareness to the new
consolidation step, which only ever sees the single nearest memory. This is an architecture
trade-off already made (two-step consolidation vs. one-shot with full context), not something to
undo, but worth knowing when reasoning about why storage decisions might miss cross-memory context
that TS would have had.

---

## Prior art that changes the shape of this plan

Before scoping chunking, it's worth knowing the original TS implementation of chunking (and
indexables) was **already designed, implemented, tested, and marked Complete** —
[`docs/plans/psm-production-memory/phase-2-chunking-pipeline.md`](psm-production-memory/phase-2-chunking-pipeline.md)
and
[`phase-3-indexables-workflows.md`](psm-production-memory/phase-3-indexables-workflows.md).
This matters directly for the "does naive splitting lose meaning?" question raised when this plan
was requested:

- The segmenter is **not** naive character/token-count splitting. It's structure-aware, in priority
  order: markdown headers → numbered step lists → paragraph boundaries → hard token-count fallback
  only as a last resort. It has a passing, real test: a `review-pr` 5-step workflow procedure stayed
  in **one chunk, all 5 steps preserved** at a 1200-token budget, and a 3-section plan produced ≥3
  chunks (one per header section) at a 40-token budget. Explicitly designed to avoid the failure
  mode the plan-requester was worried about (splitting mid-procedure).
- A faithful C# port of this exact algorithm already exists, uncommitted, at
  `dotnet/src/PsmMemory.Core/TextSegmenter.cs` (written this session, builds clean, not yet wired
  into anything).
- The **only** narrower, legitimate open question: the hard-max fallback path (used only when a
  single structural unit is *itself* still too long) does plain sentence-by-sentence greedy packing,
  which could in principle sever a long unstructured paragraph mid-thought. This is a much smaller,
  rarer problem than "do we need ML-based semantic chunking in general" — it was not something the
  original TS implementation solved with a classifier either, and there's no evidence yet that it
  needs one. Recommend: ship the structure-aware port as-is, and only revisit the hard-max fallback
  if real data shows it's actually causing problems (same evidence-first approach used throughout
  this session's other fixes).

---

## Revised fire-and-forget `remember()` design (per 2026-07-21 discussion)

Not yet implemented (a `pending_remember_requests` table + `EnqueueRememberAsync`/queue-drain
scaffolding was drafted then paused when this plan was requested). Design as agreed:

- **The enqueue path does zero work.** `remember()`, from a real hook/agent's perspective, must
  return instantly — the caller doesn't need the result, and what's being remembered isn't needed
  for the current turn. The enqueue call writes exactly one row (raw `llmResponse` + user id +
  source metadata + domain) to a durable queue table and returns. No segmentation, no LLM calls, no
  embedding — nothing computed synchronously.
- **All logic — segmentation, per-chunk storage decisions, consolidation, embedding — happens in a
  background worker** that drains the queue within a long-running process (`PsmMemory.Mcp`'s host
  process is the natural home; the CLI's single-shot commands and `serve` NDJSON mode should stay
  exactly as they are today, synchronous, since benchmark/eval harnesses depend on getting the
  decision back immediately to tally stats).
- Mirrors the existing `ignored_decisions` table pattern already in this codebase (log now, process
  later) rather than an in-memory fire-and-forget `Task`, which would be fragile — a short-lived CLI
  invocation exits right after returning and could kill an in-flight background `Task` before it
  finishes.
- Known trade-off to accept: a `recall()` called immediately after an enqueued `remember()` may not
  find that memory yet (it's still pending). This is fine for the target use case (memory is for
  future turns) but is a real behavior change from today's synchronous guarantee.

---

## Open questions — RESOLVED 2026-07-21

1. **CLI hook/install-agent/daemon layer: YES, still needed.** Not a replacement for MCP — a
   complementary automatic layer. MCP requires the agent's own reasoning to decide to call a memory
   tool (on-demand). A CLI hook route (`psm-memory hook session-start`/`hook remember`, wired into an
   agent harness's own hook system — e.g. Claude Code's `SessionStart`/`UserPromptSubmit`/`Stop`
   hooks) makes recall/remember happen automatically at turn boundaries, independent of whether the
   agent thinks to invoke a tool. Closer to PSM's "memory as a cognitive skill" thesis than an
   optional tool call. Build both: CLI hooks for the automatic baseline, MCP for explicit on-demand
   queries.
2. **`context()`'s LLM-render step: YES, build it.** Deliberate design choice, not wasted effort —
   raw structured JSON (field names, nesting, IDs, scores, timestamps) is token-inefficient and not
   the format an LLM parses most naturally; the render step can also prioritize/condense in a way a
   deterministic dump can't. Keep `recall()` as the structured/programmatic output and `context()` as
   the separate "ready to inject into a prompt" rendered output for different consumers. Keep the
   deterministic `fallbackAgentContextItems` fallback for when the render step fails to parse
   (grounding safety net).
3. **Recall-time facts/indexables/workflows: HIGH PRIORITY.** Coding-agent workflow recall (e.g. a
   `review-pr` procedure) depends on this — scope it now, not deferred.
4. **Priority ordering, resolved:** fire-and-forget queue + chunking **first** (already scoped,
   ready to build) → facts/indexables at recall-time + auto-synthesis (high priority, and a hard
   dependency of context-render below) → context-render step (depends on facts/indexables existing)
   → CLI hook layer (needs the queue's async delivery to actually have something new to hook into)
   → temporal resolution (small, fits in wherever convenient, e.g. alongside facts since facts also
   carry temporal fields).

---

## Explicit sequencing

**UPDATE 2026-07-21: the conversational-storage DPO round this section referred to is done, verified,
and promoted to production, AND the open questions above are now resolved with the project owner.**
Confirmed order: (1) fire-and-forget queue + chunking, (2) facts/indexables at recall-time +
auto-synthesis, (3) context-render step, (4) CLI hook layer, (5) temporal resolution.

**(1) fire-and-forget queue + chunking: DONE 2026-07-21.** `MemoryStore` gained WAL + busy_timeout
pragmas and full CRUD for `pending_remember_requests`.

**Corrected mid-implementation (important):** the first pass had CLI/`serve` call
`PsmService.RememberAsync` directly while MCP called a separate enqueue path — two divergent
processing routes for what should be one operation. Fixed by making `RememberQueueDrainer` (Core) the
single entry point every host goes through, with the only difference being whether a caller waits:
- `Enqueue` (static, store-only, no model needed) — fire-and-forget, durably logs the request.
- `RememberAndWaitAsync` — enqueues via the exact same `Enqueue`, then immediately drains that one
  row and returns the result(s). Used by CLI's `remember` and `serve`'s NDJSON remember command.
- `DrainOnceAsync`/`DrainRowAsync` — batch background draining, used by `RememberQueueWorker`
  (`BackgroundService` in the MCP host, its own second `MemoryStore` connection) and manual CLI
  testing (`drain-queue`).
- MCP's `remember` tool calls `Enqueue` only (fire-and-forget); CLI/`serve` call
  `RememberAndWaitAsync` (same queue, same chunking, same everything — just also waits).

**Known behavior change**: CLI `remember`/`serve`'s remember result is now a JSON **array** (one
`RememberResult` per chunk — one element in the common case) instead of a single object, since a
`remember()` call can now legitimately produce more than one decision. Any external consumer of the
CLI/serve wire format (e.g. `benchmark/locomo/src/locomo-dotnet-benchmark.ts`, if it parses this
directly) will need a small update to expect an array — not yet checked/fixed as part of this phase.

Verified via 3 new unit tests (chunking + per-chunk source-ids, single-segment passthrough, one-bad-
row isolation — all passing, full suite 39/39 non-skipped green) AND real end-to-end smoke tests
against the production GGUF runtime: (a) a genuine 3-topic response correctly split into 2 independent
stored memories with `sourceId` values `chunk-conv-2:chunk-0`/`chunk-conv-2:chunk-1`, and (b) after the
unification fix, confirmed the CLI's synchronous `remember` command still works correctly through the
new shared path (array output, memory correctly stored).

**(2) facts/indexables at recall-time + auto-synthesis: DONE 2026-07-21/22.** Implemented by a
delegated background agent, reviewed line-by-line against the real TS source
(`src/psm-core/src/{service.ts,indexables.ts}`) before acceptance — every ported algorithm (RankFacts,
RankIndexables, BuildIndexablesForRemember + all its helpers) matched the original exactly, including
details the agent found by reading the source directly rather than following the brief verbatim (e.g.
the `.slice(0, 6)` tag cap on mnemonic/fact_anchor rows).
- New `dotnet/src/PsmMemory.Core/Indexables.cs`: faithful port of `buildIndexablesForRemember`/
  `rankIndexables`/`normalizeRecallKey` and all private helpers, styled after `TextSegmenter.cs`.
- `Ranking.cs` gained `RankFacts`/`FactScore` (ported from `service.ts:413-443`, was never factored
  into `ranking.ts` in the original either).
- `PsmService.RememberAsync` now calls `BuildIndexablesForRemember` right before `ApplyDecision`,
  gated exactly like the TS call site (`WouldStoreDecision(decision) && decision.Indexables.Count == 0`).
- `RecallResult` gained `Facts`/`Indexables`/`Workflows`. Facts rank on both `RecallAsync` and
  `ContextAsync` (matching TS); indexables/workflows rank ONLY on `RecallAsync` — `ContextAsync` never
  surfaces them, matching the real TS source exactly (confirmed by grep, not assumption).
- Verified: full suite 53/53 non-skipped passing (10 new tests, incl. the canonical review-pr fixture
  from the original `tests/indexables.test.ts`), full solution builds clean, AND a real end-to-end
  smoke test against the production GGUF runtime confirmed live synthesis (mnemonic path: real model
  output correctly triggered `BuildIndexablesForRemember`, producing `fixed-auth-bug-documenting` with
  salience 0.76 matching the exact formula). The workflow-detection path is proven correct via the
  fake-runtime tests (including the literal review-pr fixture) but not re-verified live — a separate,
  unrelated model repetition-loop issue on backtick-heavy structured input blocked that specific live
  check (flagged as a follow-up, not a defect in this phase's code).

**(5) temporal expression resolution: DONE 2026-07-22.** Implemented by a delegated background agent
(run in parallel with Phase 4a below — Core-only vs Cli-only file scope, no overlap), reviewed against
the real `src/psm-core/src/temporal.ts` line-by-line before acceptance — exact match, including a
non-obvious detail (month/year rollover arithmetic) the agent got right without prompting.
- New `dotnet/src/PsmMemory.Core/TemporalNormalizer.cs`: ports `normalizeMemoryTemporalFields`/
  `normalizeFactTemporalFields`/`detectRelativeExpression`/`resolveRelativeTime` faithfully, including
  the exact non-ISO output string formats TS uses (`"14 June 2026"`, `"week before ..."`, `"May 2026"`,
  bare year strings) — deliberately not "improved" into ISO dates.
- Wired into `PsmService.RememberAsync` at the exact TS-equivalent position: between
  `ApplySourceOverrides` and `GroundingGuards.ApplyStorageGuards`.
- No schema/model changes needed — the three temporal fields already existed on the payload/record
  types; `Ranking.cs`'s `TemporalSignal` scoring term, previously dormant, now gets real signal.
- Verified: 74/74 non-skipped Core tests passing (17 new), full solution clean, AND a real end-to-end
  smoke test against the production GGUF runtime: `"...fixed the bug yesterday..."` with
  `source-timestamp="21 July 2026"` correctly resolved to `resolvedTime: "20 July 2026"`, confidence
  0.9 — composing cleanly with Phase 2's indexables synthesis in the same live call.

**(4a) CLI hook commands + install-agent (build-only): DONE 2026-07-22.** The foundational slice of
Phase 4 — `hook recall/remember/session-start/session-end` and `install-agent` — implemented by a
second delegated agent in parallel with Phase 5 above (Cli-only file scope, confirmed zero overlap
with `PsmService.cs`). Reviewed against the real `src/psm-cli/src/index.ts` line-by-line; found and
fixed one real fidelity gap myself (the old-hook-cleanup regex was missing the legacy
`psm-codex-hook.ps1` pattern TS also cleans up on reinstall) — everything else matched exactly.
- New `dotnet/src/PsmMemory.Cli/{HookIo.cs,HookContextRenderer.cs,HookCommands.cs,InstallAgentCommand.cs}`
  plus a new `dotnet/tests/PsmMemory.Cli.Tests/` project (69 tests).
- `hook remember`/`session-start`/`session-end` correctly reuse `RememberQueueDrainer.Enqueue` (the
  Phase-1 single-path primitive) — fire-and-forget by design, no model load needed, matching TS's own
  fire-and-forget philosophy for these hooks better than a bespoke daemon would have.
- `hook recall` uses a new deterministic `HookContextRenderer` as an interim stand-in for Phase 3's
  not-yet-built `agent_context` field — explicitly documented as forward-compatible (Phase 3 should
  prefer a real rendered field over this fallback once it exists, exact code comment left in place).
- `install-agent` writes to real global agent configs (`~/.claude/settings.json`,
  `~/.gemini/settings.json`, `~/.codex/config.toml`+`hooks.json`) only when NOT run with `--dry-run`
  and NOT given a `--config-dir` override — confirmed via direct code read that every path resolver
  gates on `configDir is not null`, and every test in the new project passes `--config-dir` pointing
  at a temp directory. **This command has never been run for real against any actual home directory**
  — per explicit user decision, build it but never auto-execute it without an explicit in-the-moment
  ask.
- Verified: 69/69 new Cli tests passing, full solution clean (143 total non-skipped across both test
  projects), AND real end-to-end smoke tests against the production GGUF runtime: `hook remember`
  correctly enqueued (verified via direct DB inspection) and audit-logged; `drain-queue` processed it;
  `hook recall` correctly retrieved and rendered it to stdout; `install-agent all --dry-run
  --config-dir <temp>` produced exactly the expected JSON/TOML shapes for all 3 agents and created
  zero files.

**Deferred from Phase 4** (per explicit user decision, not urgent): daemon mode (optional/off-by-
default even in TS) — **done, see below**; `setup`/`config`, `export`/`import`/`backup`, `review` (the
last of which depends on the hook audit log this round already built, so it's now unblocked whenever
picked up) remain deferred.

Next up: (3) context-render step — confirmed via research to need NO new trained adapter (reuses the
retrieval-plan adapter with a new prompt, same trick `ContextAsync` already uses), so this is
implementable now. Sequential (not parallel with anything else), since it's the last remaining
`PsmService.cs`-touching phase.

## Warm-host consolidation (bootstrap/domain/model-dir duplication + daemon mode)

Not one of the original numbered phases above — surfaced mid-session from a direct user complaint:
CLI and MCP each independently reimplemented model+store+service bootstrapping, domain-string
parsing existed as 3 separate implementations, and CLI/MCP resolved the model directory with
genuinely different (and in MCP's case, buggy) logic. Full architecture in
`docs/plans/psm-core-consolidation.md`; summary of what landed:

- **`PsmDomainParser`** (`Core/Runtime/PsmDomainParser.cs`) replaces 3 independent domain-string
  parsers (CLI direct commands, CLI hooks, MCP) with one `Parse(raw, mode)` — `Strict` throws
  `PsmDomainParseException` (CLI/MCP direct commands rewrap it into their own exception types to
  preserve exact prior external behavior), `LenientDefaultCoding` never throws (hooks).
- **`ModelDirResolver`** (`Core/Runtime/ModelDirResolver.cs`) extracts CLI's existing walk-up-from-
  executable logic into a pure, testable function, reused by both CLI's `Defaults.ResolveModelDir()`
  and a **real bug fix in MCP's `Program.cs`**: MCP previously did a flat `Path.Combine` with no
  walk-up at all, so it only found the model when launched with the repo root as its cwd.
- **Warm-host subsystem** (`Core/Runtime/WarmHost/`) — loopback-HTTP (`System.Net.HttpListener`,
  zero new package deps) server that loads the GGUF model + both runtimes once and serves
  `IPsmRuntime`/`IEmbeddingRuntime` calls to short-lived CLI processes over `POST /v1`, with sliding-
  expiration idle shutdown (default 15 min) and an atomic exclusive-create lock file
  (`WarmHostLock`, `FileMode.CreateNew`) around spawn-vs-spawn races — the one real gap in the
  original TS `daemon.ts` design (no such lock there) that this port fixes rather than carries over.
  `PsmService` needed **zero changes**: confirmed its constructor depends only on the
  `IPsmRuntime`/`IEmbeddingRuntime` interfaces, never the concrete `LlamaSharp*` types.
- **`PsmRuntimeAcquisition.AcquireAsync`** is the single decision point every CLI bootstrap call site
  now goes through: tries the warm host first when `--daemon`/`PSM_MEMORY_DAEMON=on` is set, catches
  any failure (unreachable, spawn failed, timed out) and logs+falls back to a direct local
  `LlamaSharpPsmRuntime`/`LlamaSharpEmbeddingRuntime.CreateAsync` load — matches this codebase's
  existing never-hard-fail-the-caller bias. Wired into all 5 `Commands.cs` bootstrap sites
  (`remember`/`recall`/`context`/`drain-queue`/`serve`) plus `HookCommands.cs`'s `hook recall` (the
  highest-value site — fires every agent turn). New `daemon-run` CLI command wraps
  `WarmHostServer.RunAsync` directly (spawned automatically via `WarmHostClient.SpawnDetached`, not
  meant to be typed by hand). Default is **off** (opt-in via `--daemon` or the env var) since a
  subsystem that spawns detached processes should be proven before being on by default.
- **MCP explicitly untouched** by the warm-host mechanism itself (only the unrelated model-dir bug
  fix applies to it) — MCP is already a long-lived process that loads once and stays warm; routing
  it through a daemon would mean two processes holding a loaded model for zero benefit.
- Verified: 189/189 non-skipped tests passing across both test projects (120 Core + 69 Cli) after
  each round; new `[Fact(Skip = "manual -- ...")]` `WarmHostSmokeTests.cs` (matching
  `LlamaSharpSmokeTests.cs`'s convention) covers the real end-to-end path and the idle-shutdown path
  against the real GGUF model, left skipped by default pending a manual run.
- **Not yet done**: the live manual smoke tests in `WarmHostSmokeTests.cs` (real end-to-end call
  through the proxy; idle-timeout actually firing; concurrent-spawn race; detached-process lifetime
  on Windows) have not been executed for real yet — deferred pending explicit go-ahead.
