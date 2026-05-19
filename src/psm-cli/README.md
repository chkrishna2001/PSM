# @psm-memory/cli

Command line interface for shared local PSM Memory.

```bash
psm-memory setup
psm-memory remember "User prefers concise answers."
psm-memory recall "How should I answer?"
psm-memory config --path
```

Install globally:

```bash
npm install -g @psm-memory/cli
```

Install only the CLI for normal use. npm installs `@psm-memory/sdk` automatically because it is a CLI dependency. A separate SDK install is only needed for custom SDK integrations or unpublished local-tarball testing.

Global install runs setup. In an interactive terminal it asks for the shared memory directory, local user id, recall count, embedding settings, daemon settings, and optional tracing settings. The answers are stored in an editable `config.json`; run `psm-memory config --path` to locate it.

When daemon autostart is enabled, `remember` and `recall` start a background daemon on first use. The daemon uses an OS-assigned local port, records it in `daemon.json` in the memory directory, and shuts down after the configured idle timeout.

Install Codex hooks:

```bash
psm-memory install-agent codex
```

Install multiple agent integrations:

```bash
psm-memory install-agent codex,claude,gemini
```

Gemini CLI:

```bash
psm-memory install-agent gemini
```

This writes cross-platform hook commands to each agent's settings file. For Codex, that is `~/.codex/hooks.json`:

```bash
psm-memory hook recall
psm-memory hook remember
```

For Gemini CLI, the installer writes `BeforeAgent` and `AfterAgent` hooks to `~/.gemini/settings.json`:

```bash
psm-memory hook recall --agent gemini
psm-memory hook remember --agent gemini
```

The hook commands read agent hook JSON from stdin, use the shared PSM-owned memory store, and do not depend on PowerShell or repository source paths.

Recall/context injected into agents is grounded in stored DB rows. PSM plans retrieval, SDK ranks DB/fact candidates, PSM renders concise context notes from selected rows, and SDK validation falls back to complete stored statements if rendering is invalid or ungrounded.

Enable full local PSM model I/O tracing for debugging:

```bash
psm-memory setup --trace-psm --trace-path C:\psm-memory\psm-model-io.jsonl
```

Or per shell:

```bash
PSM_MEMORY_TRACE=1 PSM_MEMORY_TRACE_PATH=./psm-model-io.jsonl psm-memory recall "What should I do?"
```

Trace files contain full prompts, raw model outputs, errors, and timings. They are local only and can include private data.

The CLI downloads the default PSM Memory Qwen 1.5B Q4_K_M GGUF model during npm installation.

If the install-time download is skipped or interrupted, run:

```bash
psm-memory setup
```

Set `PSM_MEMORY_SKIP_MODEL_DOWNLOAD=1` to skip model download in CI or packaging environments. Set `PSM_MEMORY_SKIP_SETUP=1` to skip install-time setup entirely.
