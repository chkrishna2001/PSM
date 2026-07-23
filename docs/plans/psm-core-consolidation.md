# PSM Core consolidation: warm model host + bootstrap/domain/path duplication

## Context

Model loading is the expensive part of every PSM call (GGUF mmap + persistent `LLamaContext` +
loading all available domains' LoRA adapters up front — confirmed in `LlamaSharpPsmRuntime.CreateAsync`).
Every short-lived process (a CLI one-shot command, and especially `hook recall`, which fires on
every single agent turn once `install-agent` wires it into a harness's hook config) pays this cost
from scratch, every time, because there's no mechanism for a fresh process to reach an
already-warm model running elsewhere.

Investigating this surfaced a broader pattern the user flagged directly: "why does MCP have to do
it in its own way and why does CLI have to do it in its own way" — model+store+service bootstrapping
is hand-duplicated 6 times across `Commands.cs`/`HookCommands.cs`, domain-string parsing exists as 3
separate implementations, and CLI/MCP resolve the model directory with genuinely different (and in
MCP's case, buggy) logic. This plan fixes the warm-host gap and this broader duplication together,
since they're the same root cause: bootstrapping logic living in every host instead of once in Core.

Original TS precedent exists (`src/psm-cli/src/daemon.ts`) and is proven but has one real gap (no
locking around concurrent auto-spawn, so two near-simultaneous callers can spawn two daemons) that
this plan fixes rather than ports.

**Confirmed via direct code reading, not assumption:** `PsmService`'s constructor depends only on
`MemoryStore`/`IPsmRuntime`/`IEmbeddingRuntime?` — zero references anywhere to the concrete
`LlamaSharpPsmRuntime`/`LlamaSharpEmbeddingRuntime` types. A proxy implementation of the two
interfaces that forwards over HTTP to a warm host requires **zero changes to `PsmService.cs`**.
Both concrete runtimes already self-serialize internally (private `SemaphoreSlim(1,1)` each), so a
warm host serving multiple concurrent clients against one shared runtime instance needs no extra
locking for generation/embedding calls. `MemoryStore` has no such internal locking and is
deliberately **not** part of this plan — every process keeps its own store connection exactly as
today (WAL + busy_timeout already added earlier this session for this reason); only the two
runtimes get shared.

## Architecture decisions

- **Warm-host subsystem lives entirely in `PsmMemory.Core`**, new `Runtime/WarmHost/` folder — not
  CLI-specific in nature, and Core is already the shared dependency of both CLI and MCP. CLI's
  `daemon-run` command becomes a one-line wrapper around `WarmHostServer.RunAsync(...)`.
- **Transport: loopback HTTP via `System.Net.HttpListener`** (BCL, zero new package references —
  confirmed `PsmMemory.Core.csproj` is plain `Microsoft.NET.Sdk`, not Web/Kestrel). Matches the
  proven TS shape (`GET /health`, `POST /v1`) and is trivially poke-able with `curl` for manual
  testing. Port-0 auto-assign: probe a free port via a throwaway `TcpListener`, close it, bind
  `HttpListener` to that port number (standard, small, accepted race).
- **Race-condition fix (the one thing NOT ported from TS as-is):** an atomic exclusive-create lock
  file (`daemon.lock`, `FileMode.CreateNew`) around the "check health → decide to spawn → spawn"
  sequence, with a staleness timeout (~30s) so a crashed lock-holder doesn't wedge everyone else
  forever — one bounded stale-reclaim retry, then a clear failure rather than spinning.
- **Sliding expiration**: default 15 minutes (matches TS's `idleTimeoutMs=900_000`), reset on every
  request including `/health`, via an idle-check timer that shuts the server down when exceeded.
- **Store is never proxied** — `WarmHostServer` owns only the two runtimes (`LlamaSharpPsmRuntime` +
  `LlamaSharpEmbeddingRuntime`), constructed once via their existing `CreateAsync` factories.
- **Enablement**: `PSM_MEMORY_DAEMON=on|off` env var (default **off** for initial rollout — a new
  subsystem that spawns detached processes should be opt-in until proven), overridable per-invocation
  via `--daemon`/`--no-daemon` flags on `remember`/`recall`/`context`/`hook recall`. No config-file
  system needed for this (that's a separate, already-deferred phase) — same env-var-first pattern
  already used for `PSM_DB_PATH`/`PSM_MODEL_DIR`/`PSM_MEMORY_HOOK_LOG`.
- **Fallback behavior**: if the daemon is enabled but unreachable/fails to spawn, log to stderr and
  fall back to a direct local `CreateAsync` load rather than failing the command — matches this
  codebase's existing "never hard-fail the caller" bias (`HookCommands.cs`'s try/catch-everything,
  `PlanAndRankAsync`'s catch-and-fallback).
- **MCP is explicitly untouched** by the warm-host mechanism — it's already a long-lived process
  that loads once and stays warm; routing it through the daemon would mean two processes holding a
  loaded model simultaneously for zero benefit. The one MCP change in this plan (model-dir walk-up)
  is an unrelated, independent bug fix.
- **`MemoryStore`/`PsmService` construction stays per-call-site** — only the two runtimes get a
  shared acquisition factory (`PsmRuntimeAcquisition`). Bundling store construction into the same
  factory would force one opinionated shape back onto MCP's already-correct two-store setup
  (foreground + background-worker stores, from the earlier `RememberQueueDrainer` work).

## New files (all in `dotnet/src/PsmMemory.Core/Runtime/`)

- **`WarmHost/WarmHostProtocol.cs`** — wire DTOs: a flat request `{ Operation, Prompt, Domain, Text }`
  (operation ∈ `storage-decision|recall-plan|consolidation-decision|embed`) and response
  `{ Ok, Result, Error }`, mirroring the existing NDJSON `cmd`-dispatch idiom already used in
  `Commands.cs`'s `serve` command.
- **`WarmHost/WarmHostState.cs`** — `DaemonState { Pid, Host, Port, StartedAt, LastSeenAt }` +
  atomic read/write (write-to-temp-then-`File.Move`, improving on TS's non-atomic rewrite).
- **`WarmHost/WarmHostLock.cs`** — `TryAcquire(path, staleThreshold, nowProvider) -> IDisposable?`
  encapsulating the exclusive-create + staleness-reclaim logic. Takes an injectable clock so tests
  can simulate elapsed time without real sleeps.
- **`WarmHost/WarmHostServer.cs`** — owns one `LlamaSharpPsmRuntime` + one `LlamaSharpEmbeddingRuntime`,
  routes the two HTTP endpoints, updates state on every request (the sliding-expiration mechanism),
  runs the idle-check timer. Static `RunAsync(WarmHostOptions, CancellationToken)` entry point.
- **`WarmHost/WarmHostClient.cs`** — `EnsureDaemonAsync(WarmHostOptions, ct) -> Uri` implementing
  health-check → lock → spawn-detached → poll, plus thin POST helpers.
- **`WarmHost/WarmHostPsmRuntime.cs`** / **`WarmHost/WarmHostEmbeddingRuntime.cs`** — `IPsmRuntime`/
  `IEmbeddingRuntime` implementations forwarding each interface method as one `POST /v1` call.
- **`WarmHost/WarmHostOptions.cs`** — tunables record: `ModelDir, StateDirectory, IdleTimeout
  (default 15 min), StartupTimeout (default 60s), HfRepoId`. `StateDirectory` is passed in explicitly
  (not resolved from `Environment.SpecialFolder` inside Core) — matches `InstallAgentCommand.cs`'s
  existing `configDir` parameter pattern, keeps this testable against a temp dir.
- **`PsmRuntimeAcquisition.cs`** — `AcquireAsync(modelDir, WarmHostOptions? /* null = disabled */, ct)
  -> AcquiredRuntime { IPsmRuntime Runtime; IEmbeddingRuntime EmbeddingRuntime; IAsyncDisposable }`.
  Tries the warm-host client first when enabled, catches any failure and falls back to direct
  `LlamaSharpPsmRuntime.CreateAsync`/`LlamaSharpEmbeddingRuntime.CreateAsync`.
- **`PsmDomainParser.cs`** — `Parse(string? raw, DomainParseMode mode)` where mode is
  `Strict` (throws `PsmDomainParseException`) or `LenientDefaultCoding` (never throws). Each of the
  3 existing call sites keeps its own exception type via a one-line catch-and-rewrap, preserving
  current behavior exactly (verified all 3 already do identical `.Trim().ToLowerInvariant()`
  normalization, so unifying that part is risk-free).
- **`ModelDirResolver.cs`** — `ResolveFromBaseDirectory(string startDirectory, string
  relativeModelDir) -> string`, the walk-up logic extracted from `CliRunner.cs`'s
  `Defaults.ResolveModelDir()` as a pure, previously-untested function.

## Changes to existing files

- **`CliRunner.cs`**: `Defaults.ResolveModelDir()` becomes a one-line call into `ModelDirResolver`.
  Add `Defaults.DaemonEnabled(ArgParser)` reading the env var + flag override.
- **`Mcp/Program.cs`**: fix the confirmed model-dir bug — replace the flat
  `Path.Combine(Directory.GetCurrentDirectory(), ...)` with
  `ModelDirResolver.ResolveFromBaseDirectory(Directory.GetCurrentDirectory(), ...)` when
  `PSM_MODEL_DIR` isn't set. This is the only MCP change in this plan.
- **`Commands.cs`**: the 5 bootstrap blocks (`RunRememberAsync`, `RunRecallAsync`, `RunContextAsync`,
  `RunDrainQueueAsync`, `RunServeAsync`) each replace their two `using var runtime =.../embeddingRuntime
  = ...` lines with one `await using var acquired = await PsmRuntimeAcquisition.AcquireAsync(...)`.
  Domain parsing (`ParseDomain`/`ParseDomainString`) becomes a thin wrapper over `PsmDomainParser`.
  New `RunDaemonRunAsync` command calling `WarmHostServer.RunAsync` directly.
- **`HookCommands.cs`**: the recall handler's bootstrap block gets the same `PsmRuntimeAcquisition`
  treatment (this is the highest-value call site — the one that fires every agent turn).
  `ParseHookDomain` becomes a thin wrapper over `PsmDomainParser.Parse(raw, LenientDefaultCoding)`.
- **`PsmMemoryTools.cs`** (MCP): `ParseDomain` becomes a thin wrapper over `PsmDomainParser.Parse(raw,
  Strict)`, rethrown as `ArgumentException` — no other MCP changes (MCP doesn't use the warm host).
- **`CliRunner.cs`**: dispatch `"daemon-run"` to the new command.
- **`HelpText.cs`**: add `--daemon`/`--no-daemon` documentation to `Remember`/`Recall`/`Context`/`Hook`.

## Execution order (respects file-overlap safety — same rule established earlier this session)

1. **Round 1, two parallel agents** (confirmed disjoint files):
   - Agent A: `PsmDomainParser` + `ModelDirResolver` + the MCP model-dir bug fix + wiring the 3
     domain-parsing call sites. Touches: new files, `Commands.cs` (domain parsing only),
     `HookCommands.cs` (domain parsing only), `PsmMemoryTools.cs`, `CliRunner.cs` (ResolveModelDir),
     `Mcp/Program.cs`.
   - Agent B: the entire `WarmHost/` subsystem + `PsmRuntimeAcquisition` — **new files only**, does
     not touch any file Agent A touches.
2. **Round 2, sequential** (depends on both Round 1 agents landing and being verified): wire
   `PsmRuntimeAcquisition` into the 6 bootstrap call sites in `Commands.cs`/`HookCommands.cs` (the
   same files Agent A just edited for domain parsing — must run after, not parallel), add
   `daemon-run` command, add `--daemon`/`--no-daemon` flags, add `Defaults.DaemonEnabled`.

Each round: independently rebuild + retest + review the actual diff against this plan before moving
to the next round, same verification discipline as every prior phase this session.

## Testing

**Unit tests (`dotnet/tests/PsmMemory.Core.Tests/`, no process spawning, no real model):**
- `WarmHostState` read/write round-trip incl. atomic-write behavior and corrupt-file handling.
- `WarmHostLock` acquire/already-held/stale-reclaim — inject a fake clock, no real sleeps needed.
- `WarmHostProtocol` DTO round-trip.
- `PsmDomainParser` — table test across both modes and all 3 wrapper exception types.
- `ModelDirResolver` — temp directory trees with the target at various ancestor depths + not-found.
- `WarmHostPsmRuntime`/`WarmHostEmbeddingRuntime` — fake `HttpMessageHandler`, no real listener.
- `PsmRuntimeAcquisition`'s fallback branch — inject a client that always throws, assert fallback.

**Manual smoke tests (new `[Fact(Skip = "manual — ...")]` in `WarmHostSmokeTests.cs`, matching
`LlamaSharpSmokeTests.cs`'s existing convention, run by hand against the real GGUF model):**
- Real end-to-end: start `WarmHostServer`, hit it via `WarmHostPsmRuntime`/`WarmHostEmbeddingRuntime`,
  confirm a real `PsmService.RememberAsync`/`RecallAsync` call succeeds through the proxy.
- Idle-timeout: short configured timeout (e.g. 3s), confirm the server actually shuts itself down.
- Concurrent-spawn race: launch several `hook recall`/`recall` invocations in parallel (a throwaway
  PowerShell loop) against a clean state directory, confirm exactly one daemon process survives.
- Detached-process lifetime: confirm the spawned daemon outlives the parent CLI process that spawned
  it exiting (Windows-specific verification, since Windows detached-process semantics differ from
  POSIX).

Also re-run the full existing suite (currently 74 Core + 69 Cli tests) after each round to confirm
no regression, plus a real CLI smoke test of `remember`/`recall`/`hook recall` with the daemon
force-disabled (`--no-daemon`) to confirm the non-daemon path is completely unaffected by this change.