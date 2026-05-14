# @psm-memory/cli

Command line interface for PSM Memory operations.

```bash
psm-memory init --db user_memory.db
psm-memory remember --db user_memory.db --user demo --llm-response "User prefers concise answers."
psm-memory recall --db user_memory.db --user demo --question "How should I answer?"
```

Install Codex hooks:

```bash
psm-memory install-agent --agent codex
```

Install multiple agent integrations:

```bash
psm-memory install-agent --agent codex,claude
```

This writes cross-platform hook commands to `~/.codex/hooks.json`:

```bash
psm-memory hook context
psm-memory hook remember
```

The hook commands read agent hook JSON from stdin, use `~/.codex/memories/psm-memory.db` by default, and do not depend on PowerShell or repository source paths.

The CLI downloads the default PSM Memory Qwen 1.5B Q4_K_M GGUF model during npm installation.

If the install-time download is skipped or interrupted, run:

```bash
psm-memory setup
```

Set `PSM_MEMORY_SKIP_MODEL_DOWNLOAD=1` to skip model download in CI or packaging environments.
