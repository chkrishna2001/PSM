# PSM Memory

PSM Memory is a local-first memory layer for AI agents.

The project is based on the idea described in [The Personal Small Model (PSM): Memory as a Learned Cognitive Primitive](https://dev.to/chkrishna2001/the-personal-small-model-psm-memory-as-a-learned-cognitive-primitive-324f): memory should not be treated as a database problem alone. A dedicated small model should learn memory operations such as relevance gating, storage decisions, consolidation, contradiction detection, decay, and recall weighting.

In this repo, the Personal Small Model (PSM) is used as a memory specialist around a per-user SQLite store. The primary LLM keeps doing the main reasoning work. PSM decides what should be remembered, how it should be stored, and what context should be surfaced later.

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
- `@psm-memory/cli`: `psm-memory` command for setup, hooks, recall, remember, review logs, and agent installation.
- `@psm-memory/pi-plugin`: helper APIs for agent/plugin runtimes.

Implemented capabilities:

- Local GGUF PSM runtime through `node-llama-cpp`.
- Automatic model setup from Hugging Face.
- SQLite memory tables for episodic, semantic, archival, conflicts, decay schedule, and decisions.
- Text embeddings with Hugging Face Transformers.
- Vector-backed candidate retrieval with lexical fallback.
- PSM-authored final context rendering.
- JSON repair retry when PSM returns malformed remember output.
- Codex and Claude hook installers.
- Local hook audit logs and review reports.

## Install

After publishing:

```bash
npm install -g @psm-memory/cli
```

Then download or verify the local PSM model and embedding model:

```bash
psm-memory setup
```

The CLI installs:

- PSM GGUF model: `chkrishna2001/psm-memory-qwen-1.5b-gguf`
- default text embedding model: `Xenova/all-MiniLM-L6-v2`

To skip model download during package install:

```bash
PSM_MEMORY_SKIP_MODEL_DOWNLOAD=1 npm install -g @psm-memory/cli
psm-memory setup
```

## Basic Usage

Initialize a memory database:

```bash
psm-memory init --db user_memory.db
```

Store memory from an agent response:

```bash
psm-memory remember \
  --db user_memory.db \
  --user demo \
  --llm-response "User prefers SQLite for local-first tools."
```

Recall relevant memory:

```bash
psm-memory recall \
  --db user_memory.db \
  --user demo \
  --question "What database should I use?"
```

Build private context for an agent prompt:

```bash
psm-memory context \
  --db user_memory.db \
  --user demo \
  --prompt "Which database should this local tool use?"
```

Show stored memories:

```bash
psm-memory show --db user_memory.db --table episodic --pretty
psm-memory show --db user_memory.db --table semantic --pretty
psm-memory show --db user_memory.db --table decisions --pretty
```

## Agent Hooks

PSM Memory can install hooks for supported local agents.

Codex:

```bash
psm-memory install-agent --agent codex
```

Claude Code:

```bash
psm-memory install-agent --agent claude
```

Both:

```bash
psm-memory install-agent --agent codex,claude
```

The installed hooks call:

```bash
psm-memory hook context
psm-memory hook remember
```

The hook commands read JSON from stdin, use the local PSM model, and write to the default database:

```text
~/.codex/memories/psm-memory.db
```

They do not depend on PowerShell or repository source paths.

## Review Logs

PSM runs automatically during agent sessions, so users need a way to inspect what happened afterward.

Review today’s hook activity and memory decisions:

```bash
psm-memory review-log --date 2026-05-14
```

The review report includes:

- hook type
- status
- timings
- PSM decision
- memory content
- suggested feedback labels

No data is uploaded. Users can choose what, if anything, to share.

## Why Vector Search Is Still Used

The article argues that memory is more than vector search. That does not mean vectors are useless.

PSM Memory uses vectors as candidate retrieval, not as the memory system itself:

```text
embedding search -> candidate memories
PSM recall plan  -> target tables and hints
PSM renderer     -> final concise context
```

Vectors help find semantically similar memories. PSM decides whether those memories matter and how they should be presented.

## Local-First Privacy Model

PSM Memory is designed so user content stays outside model weights.

- PSM weights are shared memory skill.
- SQLite memory stores are local and user-specific.
- Hook audit logs are local JSONL files.
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

- warm daemon to avoid repeated model cold starts
- better consolidation and decay jobs
- richer user review/feedback export
- OpenCode/Cursor/Antigravity integrations
- stronger embedding backends and possible multimodal memory

