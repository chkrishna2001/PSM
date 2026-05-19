# Session Hooks and Lean Context Injection

## Summary

Add session-level hooks for developer continuity and reduce hook recall payload size. Session hooks should store compact project/work state, not raw transcripts. Hook recall should inject only concise, high-signal context so PSM stops sending heavy repeated blocks.

## Key Changes

- Add new CLI hook commands:
  - `psm-memory hook session-start`
  - `psm-memory hook session-end`
  - Gemini variants use `--agent gemini` and suppress output like existing hooks.

- Install session hooks where supported:
  - Codex/Claude: keep existing prompt/stop hooks and add session start/end event hooks if the agent accepts those event names.
  - Gemini: add matching start/end hooks only if its settings schema supports those events; otherwise expose commands but do not install unsupported event names.
  - Existing `recall` and `remember` hooks remain compatible.

- Session start behavior:
  - Capture compact developer state: `cwd`, git repo root, branch name, dirty file summary, package/project name, agent name, and transcript/session path if available.
  - Store as one compact memory through `PsmService.remember()`.
  - Use `source_kind=session_start`, `source_timestamp=<now>`, and `source_label=agent session start`.

- Session end behavior:
  - Capture compact end-of-session state: repo/branch, changed file summary, recent test/build result when inferable, and unresolved next steps when inferable.
  - Do not store full transcript.
  - Store one compact memory through `PsmService.remember()`.
  - Use `source_kind=session_end`, `source_timestamp=<now>`, and `source_label=agent session end`.

- Fix heavy hook context:
  - Add a lean renderer for hook recall output.
  - Default limits: max 3 context items, max 300 chars per item, max 1,200 chars total injected context.
  - Prefer `memory_fact` rows over long narrative memories when both are relevant.
  - Strip repeated metadata prefixes when they are not needed.
  - Keep only useful provenance fields: source id/date when available.
  - Do not include raw `memory_context`, `fact_context`, ranking metadata, or full JSON in normal hook injection.
  - Preserve full JSON only for `--pretty` / debug output.

- Update PI plugin renderer similarly:
  - Apply the same max item and max character behavior.
  - Keep API compatibility for `memoryContext`.

## Test Plan

- CLI installer tests:
  - Existing Codex/Claude/Gemini recall/remember hooks remain installed.
  - Session hook commands are included only for supported agent event names.
  - Old PSM hooks are still removed cleanly before adding new ones.

- Hook behavior tests:
  - `hook recall` injects no more than the configured max items and total chars.
  - Long memory content is truncated cleanly.
  - Fact context renders before long memory prose.
  - Gemini output remains valid `hookSpecificOutput.additionalContext`.
  - Missing prompt/session payload skips safely.

- Session command tests:
  - `hook session-start` stores one compact memory.
  - `hook session-end` stores one compact memory.
  - No raw transcript content is stored by default.
  - Source metadata uses `session_start` / `session_end`.

## Assumptions

- We should not ingest full session transcripts in session hooks.
- Hook context should optimize for usefulness, not exhaustiveness.
- If an agent does not support a session event name, the CLI command can still exist but installer should avoid writing invalid hook config.
- Normal hook output should be small by default; debug/full output belongs behind `--pretty`.
