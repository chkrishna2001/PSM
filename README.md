# PSM Memory

PSM Memory is a local-first shared memory layer for AI agents.

The project is based on the idea described in [The Personal Small Model (PSM): Memory as a Learned Cognitive Primitive](https://dev.to/chkrishna2001/the-personal-small-model-psm-memory-as-a-learned-cognitive-primitive-324f): memory should not be treated as a database problem alone. A dedicated small model should learn memory operations such as relevance gating, storage decisions, consolidation, contradiction detection, decay, and recall weighting.

In this repo, the Personal Small Model (PSM) is used as a memory specialist around a shared per-user SQLite store. The primary LLM keeps doing the main reasoning work. PSM decides what should be remembered, how it should be stored, and what should be recalled later.

## Why This Exists

Most agent memory systems follow this loop:

```text
store text -> vector search -> inject retrieved chunks -> hope the LLM uses them correctly
```

That helps with retrieval, but it does not solve the harder memory questions:

- Is this worth remembering?
- Is it episodic, semantic, archival, or a conflict?
- Is it a durable user preference or just transient task noise?
- Does it contradict older memory?
- Should it decay, strengthen, or be promoted?
- What should be shown to the agent right now?

PSM Memory separates those responsibilities:

```text
PSM weights    -> shared memory skill, trained once
Memory store   -> per-user private content
Vector index   -> semantic candidate retrieval
Agent LLM      -> receives concise private context
```

The PSM weights do not store user content. User memory stays in the local SQLite database.

## What Works Today

This repo currently ships three public packages:

- `@psm-memory/sdk`: memory store, PSM service orchestration, local model runtime, embeddings, ranking, and parsing.
- `@psm-memory/cli`: `psm-memory` command for setup, remember, recall, review, export, and agent installation.
- `@psm-memory/pi-plugin`: helper APIs for agent/plugin runtimes.

Implemented capabilities:

- Local GGUF PSM runtime through `node-llama-cpp`.
- Automatic model setup from Hugging Face.
- SQLite memory tables for episodic, semantic, archival, conflicts, decay schedule, and decisions.
- Text embeddings with Hugging Face Transformers.
- Vector-backed candidate retrieval with lexical fallback.
- PSM-authored recall for humans and agents.
- Grounded agent context rendering; PSM can turn retrieved rows into concise context notes, and SDK validation falls back to exact stored statements if rendering is invalid.
- JSON repair retry when PSM returns malformed remember output.
- Codex and Claude hook installers.
- Local hook audit logs, review reports, and opt-in full PSM model I/O traces.

## Install

After publishing, install the CLI globally:

```bash
npm install -g @psm-memory/cli
```

Install only the CLI for normal use. The CLI declares `@psm-memory/sdk` as a dependency, so npm installs the SDK automatically. You do not need to install both packages unless you are testing unpublished local tarballs.

Global install runs setup. When run from an interactive terminal, setup asks for the shared memory directory, local user id, recall count, embedding settings, and daemon settings. Press Enter to accept the defaults. The same setup can be rerun later:

```bash
psm-memory setup
```

The CLI installs:

- PSM GGUF model: `chkrishna2001/psm-memory-qwen-1.5b-gguf`
- default text embedding model: `Xenova/all-MiniLM-L6-v2`
- editable config: `config.json` in the PSM app-data directory

When daemon autostart is enabled, normal `remember` and `recall` commands start the daemon on first use. The daemon binds to a dynamic local port, writes runtime discovery data to `daemon.json` in the memory directory, and exits after the configured idle timeout.

To skip model download during package install:

```bash
PSM_MEMORY_SKIP_MODEL_DOWNLOAD=1 npm install -g @psm-memory/cli
psm-memory setup
```

To skip install-time setup entirely:

```bash
PSM_MEMORY_SKIP_SETUP=1 npm install -g @psm-memory/cli
```

Package roles:

- `@psm-memory/cli`: install this for the `psm-memory` command and agent hooks. It pulls the SDK transitively.
- `@psm-memory/pi-plugin`: install this when embedding PSM Memory in another agent/plugin runtime. It pulls the SDK transitively.
- `@psm-memory/sdk`: install this directly only when building a custom integration against the SDK APIs.

## Basic Usage

Store a memory:

```bash
psm-memory remember "User prefers SQLite for local-first tools."
```

Recall relevant memory:

```bash
psm-memory recall "What database should I use?"
```

Recall returns readable text by default so humans and agents can use the same output. Use JSON only when a tool needs structured data:

```bash
psm-memory recall "What database should I use?" --json
```

Choose a custom shared memory directory during setup:

```bash
psm-memory setup --memory-dir C:\psm-memory
```

Find or inspect the editable config file:

```bash
psm-memory config --path
psm-memory config
```

## Agent Hooks

PSM Memory can install hooks for supported local agents.

Codex:

```bash
psm-memory install-agent codex
```

Claude Code:

```bash
psm-memory install-agent claude
```

Gemini CLI:

```bash
psm-memory install-agent gemini
```

Multiple agents:

```bash
psm-memory install-agent codex,claude,gemini
```

The installed hooks call internal PSM hook commands that automate the same flow:

```bash
psm-memory hook recall
psm-memory hook remember
```

Gemini CLI hooks are installed in `~/.gemini/settings.json` and use Gemini's JSON hook protocol:

```bash
psm-memory hook recall --agent gemini
psm-memory hook remember --agent gemini
```

The hook commands read agent JSON from stdin, use the local PSM model, and write to the shared PSM-owned memory store. They do not depend on PowerShell or repository source paths.

The current retrieval path for hook context is:

```text
agent/user prompt
-> PSM creates a retrieval plan
-> SDK reads candidate rows from episodic, semantic, archival, memory_facts, and embeddings when enabled
-> SDK ranks candidates with hybrid lexical/vector scoring
-> PSM renders grounded context notes from the selected rows
-> SDK validates the rendered notes against source rows and falls back to complete stored statements if needed
-> agent receives the private PSM Memory Context block
```

## Review Logs

PSM runs automatically during agent sessions, so users need a way to inspect what happened afterward.

Review today’s hook activity and memory decisions:

```bash
psm-memory review --date 2026-05-14
```

The review report includes:

- hook type
- status
- timings
- PSM decision
- memory content
- suggested feedback labels

No data is uploaded. Users can choose what, if anything, to share.

## Full PSM I/O Tracing

For debugging or collecting feedback from trusted testers, enable full local tracing of every prompt sent to PSM and every raw model response:

```bash
psm-memory setup --trace-psm --trace-path C:\psm-memory\psm-model-io.jsonl
```

You can also enable it per shell:

```bash
PSM_MEMORY_TRACE=1 PSM_MEMORY_TRACE_PATH=./psm-model-io.jsonl psm-memory recall "What should I do?"
```

Trace records are JSONL and include the operation name, generation options, full prompt, raw output, errors, and timing. The file is local only and is not uploaded automatically. It contains private prompts, memories, and assistant responses, so testers should share it only when they explicitly choose to.

## Why Vector Search Is Still Used

The article argues that memory is more than vector search. That does not mean vectors are useless.

PSM Memory uses vectors as candidate retrieval, not as the memory system itself:

```text
embedding search -> candidate memories
PSM recall plan  -> target tables and hints
PSM render        -> grounded agent context
```

Vectors help find semantically similar memories. PSM decides which memory rows matter and can render concise context notes from those rows. The SDK validates rendered context against selected DB rows and falls back to complete stored statements if PSM returns invalid or ungrounded output.

## Local-First Privacy Model

PSM Memory is designed so user content stays outside model weights.

- PSM weights are shared memory skill.
- SQLite memory stores are local and user-specific.
- Hook audit logs are local JSONL files.
- Full model I/O traces are local and opt-in.
- No upload path is implemented.
- Training feedback should be exported only by explicit user action.

## Repository Layout

```text
src/psm-core       SDK package
src/psm-cli        CLI package
src/psm-pi-plugin  plugin helper package
benchmark/locomo   local benchmark tooling
docs/              architecture and release notes
```

## Development

Install dependencies:

```bash
npm install
```

Build:

```bash
npm run build
```

Test:

```bash
npm test
```

Run package dry-runs:

```bash
npm pack --workspace src/psm-core --dry-run
npm pack --workspace src/psm-cli --dry-run
npm pack --workspace src/psm-pi-plugin --dry-run
```

## Release

The repo uses Changesets.

```bash
npm run changeset
npm run version-packages
```

Publishing is handled by GitHub Actions on:

- GitHub Release publication
- manual workflow dispatch

A plain `git push` does not publish to npm.

## Status

This is an early implementation of the PSM architecture. The current system is usable locally, but important work remains:

- daemon hardening and lifecycle polish
- better consolidation and decay jobs
- richer user review/feedback export
- OpenCode/Cursor/Antigravity integrations
- stronger embedding backends and possible multimodal memory
