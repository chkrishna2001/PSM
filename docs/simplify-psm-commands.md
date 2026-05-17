# Simplify PSM CLI Into One Shared Memory Flow

  ## Summary

  PSM is the product-owned memory layer. Codex, Claude, PI plugin, and direct CLI usage are all clients of the same flow: remember writes memory, recall
  reads memory. There should be no separate “agent flow” or public context concept. Agent integrations should automate the same remember/recall operations
  against one shared PSM-owned store.

  ## Key Changes

  - Move default memory storage to a PSM-owned per-user app data location:
      - Windows: %LOCALAPPDATA%\psm-memory\psm-memory.db
      - Non-Windows: platform-appropriate PSM-owned user data location.
  - Add setup-time storage configuration:

    psm-memory setup
    psm-memory setup --memory-dir <path>

  - Persist the selected memory directory in PSM config so all CLI commands and agent hooks use the same shared store.
  - Replace option-heavy public commands with positional memory verbs:

    psm-memory remember "psm is a memory tool"
    psm-memory recall "what is psm?"

  - Remove public context from normal help/docs. Any hook-specific prompt wrapper should be an internal formatter around recall, not a separate product
    command.
  - Make recall return readable Markdown/text by default, suitable for both humans and LLM agents.
  - Add optional machine-readable output only for tooling:

    psm-memory recall "what is psm?" --json

  - Keep agent provenance as metadata using existing source fields, but never partition memory by agent.
  - Use one default identity derived from the OS username across CLI and hooks.
  - Remove internal knobs from normal help and README usage:
      - --top-k
      - --embedding-model
      - --context-size
      - --gpu
      - --gpu-layers
      - --no-embeddings
  - Keep --db only for admin/developer commands where explicit database targeting is useful:
      - backup
      - export
      - import
      - migrate
      - tests, benchmarks, and dev scripts.

  ## CLI Behavior

  - Main product flow:

    psm-memory setup
    psm-memory remember "psm is a memory tool"
    psm-memory recall "what is psm?"
    psm-memory install-agent codex
    psm-memory install-agent claude
    psm-memory review
    psm-memory export memories.json

  - install-agent wires agents into the same flow:
      - before prompt: recall relevant memories as Markdown/text
      - after response/session: remember useful durable information
  - Hooks may use hidden/internal plumbing if required by agent protocols, but docs/help should describe only the product concepts.
  - Existing .codex\memories\psm-memory.db development data can be ignored because this has not been released.

  ## Test Plan

  - Verify setup --memory-dir <path> persists the selected PSM memory directory.
  - Verify remember "..." writes to the configured shared DB without --db.
  - Verify recall "..." reads from the configured shared DB and returns Markdown/text by default.
  - Verify agent source metadata is recorded without separating retrieval by agent.
  - Verify public help/docs omit context and internal runtime/retrieval flags.
  - Verify --db remains available for admin/import/export/migration and test scenarios.
  - Run:

    npm test
    npm run build

  ## Assumptions

  - PSM owns memory storage, not Codex, Claude, or any one agent.
  - There is one shared local memory flow for humans and agents.
  - Markdown/text recall is the default because LLM agents can consume it directly.
  - JSON is only for tools, tests, review UIs, plugins, and automation.
  - --memory-dir is the only normal setup-time storage option users need.
