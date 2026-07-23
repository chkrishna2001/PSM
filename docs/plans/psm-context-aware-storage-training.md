# PSM context-aware storage training round

**Status as of 2026-07-23: curriculum format designed + verified, 3 real example rows built, full
curriculum not yet mined, no training run started.** Read this whole doc before touching anything —
it's written to hand off cold to a fresh agent/session.

## What we're working on

PSM gives AI coding agents persistent memory: a small Qwen 0.5B model, fine-tuned as separate LoRA
adapters per domain (`conversational`, `coding`), decides what's worth storing from an agent's
responses and later retrieves it. The user has spent **months** iterating on these adapters —
every round looks good on isolated evals, then falls apart against realistic end-to-end testing.
This session found the actual root cause (see below) and is now fixing it properly instead of
retraining blind again. **The user's own words: "make sure we train models properly, from months I
am waiting to release PSM with proper models."** Treat that as the bar — do not promote anything
without the verification steps in "What we expect from it" below.

## What we achieved (this session)

- Built/verified a full RunPod workflow for testing the real .NET CLI product end-to-end (deploy →
  push source via base64/tar over proxy SSH → install .NET SDK on pod → publish self-contained →
  fix a known HF-repo adapter-layout bug → ingest/recall → pull results → delete pod). See
  `project_psm_locomo_dotnet_runpod_dryrun.md` memory for the exact commands.
- Ran the real LoCoMo benchmark end-to-end: **16.6% answer accuracy**, retrieval misses evidence
  entirely **58%** of the time. Found and fixed 3 real bugs along the way (temporal-date surfacing,
  `TemporalNormalizer` missing "this month/week/year", a dataset schema bug for adversarial
  questions) — all landed, see `benchmark/locomo/src/locomo-dotnet-benchmark.ts` and
  `dotnet/src/PsmMemory.Core/TemporalNormalizer.cs`.
- Ran an equivalent coding-domain probe (real Claude Code session transcript, ingest + recall) and
  found the **same failure patterns**: stored-but-not-retrieved, and self-reflective content not
  stored at all. Fixed a real production bug found in the process: `HookContextRenderer.cs` (the
  ACTUAL `psm-memory hook recall` output real agents see) never surfaced `SourceTimestamp`/
  `ResolvedTime` — fixed, tested, 0 regressions.
- Restored `psm-model/prod-memory/fixtures/holdout-coding-agent-cases.json` (100 cases), accidentally
  deleted in an unrelated commit.
- **Root cause confirmed**: storage decisions are made on ONE isolated turn with zero surrounding
  conversational context. Short/reactive turns that only make sense with context (e.g. "It's Shia
  Labeouf!" answering "who said that quote?") get correctly-in-isolation judged as unstorable.
  **Tried the naive fix** (just concatenate the last 2 turns into the prompt) and verified live
  against the real model: it triples storage volume but the model picks facts from ANYWHERE in the
  window and misattributes them to the wrong turn's sourceId — confirmed via manual comparison of 30
  real stored memories against source text, not just the aggregate count. **Reverted.** This is the
  key lesson: an aggregate metric going up is not evidence of a real fix — always read actual content.
- Designed the real fix: `PromptBuilder.BuildStoragePrompt` (`dotnet/src/PsmMemory.Core/Prompts/PromptBuilder.cs`)
  now takes an optional `contextTurns` parameter. **Byte-identical output when omitted** (locked down
  by `PromptBuilderTests.cs`) — zero risk to the currently-running production adapters. When context
  is passed, it's labeled with an explicit instruction: *"for understanding only — do NOT extract a
  memory from this section"* — directly targeting the misattribution failure just found.
- Built 3 real training example rows (schema below) with a passing test proving the JSONL's `user`
  field is byte-identical to what `PromptBuilder.BuildStoragePrompt` actually emits for the same
  inputs — this specific consistency check is what prevents training data and inference code from
  silently drifting apart.

## What's next

1. **Mine a full curriculum, both domains.** 3 examples exist; past curricula in this project ran
   40-150+ hand-verified rows. Need real, verified rows for 3 categories per domain:
   - **Positive**: current turn is unstorable alone, context makes it storable (e.g. the Luna-age
     example below).
   - **Discipline**: context is rich/storable, but the CURRENT turn adds nothing new — correct label
     is `ignore`, resisting the temptation to re-extract the context's fact. This category is what
     the reverted naive experiment lacked, and is the most important one to get right.
   - **Control**: context present but not needed — current turn is fully self-contained. Prevents the
     adapter from learning to always lean on context.
   - Conversational source: LoCoMo **training-split** conversations only — `conv-26`, `conv-47`,
     `conv-48`, `conv-49`, `conv-50` in `benchmark/locomo/data/locomo10.json`. **Never** mine from
     `conv-30/41/42/43/44` — those are held out for eval; using them for training contaminates every
     future benchmark run.
   - Coding source: real Claude Code/Codex session transcripts. Note found this session: coding-agent
     turns tend to be more self-contained/verbose than casual conversational banter, so the exact
     "short turn meaningless alone" pattern is rarer — may need more sessions or a different framing
     for coding's discipline/positive examples. Check `~/.claude/projects/C--Users-chkri-source-repos-PSM/`
     for session transcripts not already used in `psm-model/prod-memory/scripts/extract_coding_agent_candidates.py`'s
     source list (avoid reusing exactly the same turns already in the 100-case gate where avoidable).
2. **Thread context through the real production ingest path**, not just training data. Today's work
   only touched `PromptBuilder.cs` (the format) and training-data mining. `RememberRequest`
   (`dotnet/src/PsmMemory.Core/Models/Requests.cs`) needs a new optional field for recent context
   turns, threaded through `PsmService.RememberAsync` → `IPsmRuntime.GenerateStorageDecisionAsync` →
   `PromptBuilder.BuildStoragePrompt`. The actual hook/CLI/MCP callers (`HookCommands.cs`,
   `RememberQueueDrainer.cs`) need to actually supply a small window of recent turns — for hooks this
   likely means tracking the last N turns per session somewhere (does not exist yet).
3. **Train** via the existing `_run_hf_lora.py` profile pattern (see "How to do it").
4. **Verify rigorously** before promoting anything (see "What we expect from it").

## How to do it

- Training row schema (SFT, confirmed real via `_run_hf_lora.py`/`hf_lora_train.py`): JSONL, one row
  per line, `{"id": "...", "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."},
  {"role":"assistant","content":"<compact JSON StorageDecision>"}]}`. See
  `dotnet/tests/PsmMemory.Core.Tests/PromptBuilderTests.cs`'s
  `PromptBuilderCurriculumConsistencyTests` for the exact expected `user` string shape.
  **3 complete real seed rows (positive/discipline/control) are committed at
  `psm-model/prod-memory/fixtures/context-aware-storage-seed.jsonl`** — read these first, they're the
  template for every new row: same schema, same `note`/`category` fields for traceability, same
  verified-consistent `user` field shape.
- **Every new row's `user` field must be produced by literally calling `PromptBuilder.BuildStoragePrompt`
  with the intended `llmResponse`/`contextTurns`, then copying that exact string** — never hand-type
  it separately from the code, or training/inference will silently drift apart again.
  Add a test per new row (or a bulk test iterating the whole curriculum file) asserting this.
- Launch training: `python psm-model/scripts/_run_hf_lora.py --profile <new-profile-name> --deploy --sync-code`.
  Add the new profile to `HF_PROFILES` in `_run_hf_lora.py` first — model on:
  - Conversational: resume from `hf-prod-conversational-storage-v2-qwen0.5b/adapter` (current
    production base), DPO or SFT depending on how much the new curriculum overlaps with existing
    correct behavior (existing `conversational-storage-dpo1` profile is a DPO example: 60 steps, lr
    3e-6, beta 0.15 — mirror this shape if doing DPO; use SFT-boost like `storage-v16b` — 800 steps,
    lr 1e-4 — if doing a fuller retrain).
  - Coding: resume from current production `hf-prod-storage-v16b-qwen0.5b/checkpoint-800`, not a
    fresh base retrain, since existing detail-fidelity there is already good (62% verbatim) — this
    round should ADD context-discipline, not relearn everything.
  - RunPod mechanics (deploy, proxy SSH quirks, GHCR image, GPU preference A5000>L4): see the
    `runpod-gpu-train` skill and `feedback_runpod_ssh_pty_technique.md`/
    `feedback_runpod_custom_docker_image.md` memory files.

## What we expect from it — the promotion bar

**Do not promote a new checkpoint on aggregate metrics alone.** Required before promotion, both
domains:
1. Existing isolated gate (100-case coding, 419-case conversational) — no regression in
   `action_match_rate`.
2. Re-run the SAME LoCoMo probe methodology used this session (ingest a held-out conversation,
   `--skip-answer`, dump the store, cross-reference specific previously-known-missed evidence IDs)
   — confirm previously-misattributed cases now attribute correctly, not just that storage volume
   went up.
3. Re-run the coding-transcript probe similarly.
4. **Manually read a real sample (~20-30) of newly-stored memories against their actual source
   turns** for both domains — this exact check is what caught the misattribution problem in the
   reverted experiment; an aggregate score would have hidden it completely.
5. Only after all four pass: update `convert_adapters_gguf.py`'s `DOMAIN_ADAPTERS` mapping, regenerate
   the production GGUF adapter file, smoke-test via the real CLI, then update
   `project_psm_locomo_dotnet_runpod_dryrun.md` / `project_psm_coding_domain_probe_2026_07_23.md` /
   `PSM-MEMORY.md` with the real before/after numbers.

## Reference: the 3 example rows already built

`psm-model/prod-memory/fixtures/context-aware-storage-seed.jsonl` — 3 real, complete rows (positive:
Luna's age from conv-48; discipline: meteor-shower reaction from conv-26; control: Max-the-cat from
conv-48), all real LoCoMo training-split turns, never used for eval. This is the template for the
full curriculum: same schema, same field names, same verified-consistent `user` string shape (see
`PromptBuilderTests.cs`). The next agent's mining work is to produce many more rows in this exact
shape, for both domains, per the 3 categories in "What's next" above.
