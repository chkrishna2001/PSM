# PSM Memory Migration And Setup

## Goal

PSM memory databases must be portable across machines and resilient to schema changes. PSM's main job is memory management, so setup must protect existing memories before migrating and must provide an explicit export/import path.

## Local Setup

Run setup after installing or updating the CLI:

```powershell
psm-memory setup --pretty
```

Global install runs the same setup:

```powershell
npm install -g @psm-memory/cli
```

When setup is run from an interactive terminal, it asks for:

- shared memory directory
- local user id
- recall candidate count
- embedding enablement and model
- daemon enablement, autostart preference, idle timeout, and startup timeout

Press Enter to accept a default. The answers are written to an editable config file:

```powershell
psm-memory config --path
psm-memory config
```

To choose a shared memory directory:

```powershell
psm-memory setup --memory-dir C:\psm-memory --pretty
```

Setup now:

- installs or verifies the local PSM model
- prepares the embedding model unless `--skip-embeddings` is set
- creates or migrates the shared PSM memory DB
- writes editable config, including daemon settings for the upcoming daemon
- reports the DB path

Daemon autostart does not reserve a fixed port. On first `remember` or `recall`, PSM starts a background daemon, lets the OS choose a local port, writes the active endpoint to `daemon.json` in the memory directory, and shuts the daemon down after the configured idle timeout.

To migrate only the DB schema:

```powershell
psm-memory migrate --pretty
```

By default the DB is:

```text
%LOCALAPPDATA%\psm-memory\psm-memory.db
```

Override the shared memory directory during setup:

```powershell
psm-memory setup --memory-dir C:\path\to\psm-memory
```

## Backup Before Migration

Before any schema-changing release:

```powershell
psm-memory backup --pretty
```

This creates a timestamped SQLite copy next to the active DB. Use `--out` to choose a location:

```powershell
psm-memory backup --out C:\backups\psm-memory-before-upgrade.db --pretty
```

## Portable Export / Import

Use JSON export when moving memories to another system or when a raw SQLite copy is inconvenient:

```powershell
psm-memory export C:\backups\psm-memory-export.json --pretty
```

Import on the new machine:

```powershell
psm-memory import C:\backups\psm-memory-export.json --pretty
psm-memory migrate --pretty
```

The export includes episodic, semantic, archival, conflicts, decisions, and decay schedule rows. Embeddings are intentionally not exported; they can be regenerated as needed and should not be treated as the source of truth.

## Release Rule For Future DB Changes

Any future schema change must include:

- an idempotent migration in `MemoryStore.initializeSchema()`
- a backup/export path that works before migration
- a test that imports old rows into the new schema
- setup behavior that migrates the default DB

The invariant is simple: PSM may evolve, but user memories remain portable and recoverable.
