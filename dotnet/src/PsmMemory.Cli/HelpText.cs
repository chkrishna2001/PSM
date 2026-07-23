namespace PsmMemory.Cli;

internal static class HelpText
{
    public const string ModelDirNote =
        "  --model-dir <path>    Directory containing the exported PSM GGUF model: the base model\n" +
        "                        GGUF, tokenizer files, and LoRA adapters for storage/retrieval_plan/\n" +
        "                        consolidation. Downloaded from HuggingFace automatically on first run\n" +
        "                        if missing. Defaults to psm-model/prod-memory/gguf-runtime/v1 resolved\n" +
        "                        against the repo root (found by walking up from the executable). When\n" +
        "                        this tool is installed elsewhere (e.g. `dotnet tool install --global`),\n" +
        "                        that auto-detection will not find the repo and you MUST pass this\n" +
        "                        flag explicitly.";

    public const string DomainNote =
        "  --domain <name>       One of coding|conversational (default: coding). Selects which\n" +
        "                        trained adapter set to use for this call. conversational requires\n" +
        "                        conversational_* GGUF LoRA adapter files to exist in --model-dir\n" +
        "                        -- requesting it when those are missing fails with a clear error.";

    public const string DaemonNote =
        "  --daemon              Force-enable the warm-host daemon for this call even if\n" +
        "                        PSM_MEMORY_DAEMON is unset/off: reach an already-warm model over\n" +
        "                        loopback HTTP (spawning one via 'daemon-run' if none is running yet)\n" +
        "                        instead of loading the model directly in this process.\n" +
        "  --no-daemon           Force-disable the daemon for this call even if PSM_MEMORY_DAEMON=on\n" +
        "                        -- always load the model directly in this process.\n" +
        "                        Default (neither flag passed): controlled by the PSM_MEMORY_DAEMON\n" +
        "                        env var (\"on\" enables it, unset/anything else = off). Any daemon\n" +
        "                        failure (unreachable, failed to spawn, timed out) falls back to a\n" +
        "                        direct local model load automatically -- this never hard-fails a call.";

    public const string Root = """
        psm-memory - thin CLI wrapper around the PsmMemory.Core SDK (local-first PSM memory store
        backed by an in-process GGUF model, loaded via LlamaSharp, with storage/retrieval_plan/
        consolidation LoRA adapters).

        Usage: psm-memory <command> [options]

        Commands:
          remember    Analyze an assistant response and store durable memory from it.
          recall      Answer a question by retrieving and ranking relevant stored memories.
          context     Retrieve relevant memories to ground a prompt (lower relevance threshold than recall).
          serve       Load the model once and answer many remember/recall/context requests over NDJSON stdin/stdout.
          show        Print stored memories for a user from one table (episodic/semantic/archival).
          conflicts   Print logged memory conflicts by status.
          init        Create (or update) the SQLite schema at --db. Alias: migrate.
          enqueue-remember  Queue a remember request for background processing; returns instantly.
          drain-queue       Process one batch of the queued remember requests right now.
          hook        Agent-harness integration points: recall|remember|session-start|session-end.
          install-agent  Wire psm-memory hooks into a coding agent's global config (codex/claude/gemini).
          daemon-run  (internal) Run the warm-host server directly -- spawned automatically, not
                      normally invoked by hand. See --daemon on other commands.
          help        Show this message.

        Run 'psm-memory <command> --help' for a command's flags.
        """;

    public const string Remember = """
        psm-memory remember - analyze an assistant response and store durable memory from it.

        The model only ever looks at the assistant response text you pass via --message; that is
        the material it decides whether/how to store (ignore, store as an episodic memory, or
        promote to semantic memory). It does not see --user-message directly during storage
        decisioning -- that field is accepted by the SDK for future/optional use but is not
        currently read by the storage prompt.

        Usage: psm-memory remember --message <text> --user <id> [options]

        Required:
          --message <text>      The assistant/LLM response text to extract durable memory from.
          --user <id>           User id memories are scoped to.

        Optional:
          --user-message <text> The user's message that prompted the response (accepted for
                                 forward-compatibility; not read by the current storage prompt).
          --extra-tags <a,b,c>  Comma-separated tags appended to any memory that gets written.
          --no-existing         Skip loading existing memories before deciding (IncludeExistingMemories=false).
          --source-kind <k>     Source metadata override: kind (e.g. "chat", "email").
          --source-id <id>      Source metadata override: id.
          --source-label <s>    Source metadata override: label.
          --source-timestamp <t> Source metadata override: ISO timestamp.
          --db <path>           SQLite db path (default: user_memory.db in the current directory).
        {{DOMAIN}}
        {{MODEL_DIR}}
        {{DAEMON}}

        Goes through the same durable queue every caller (including the MCP server) uses -- see
        RememberQueueDrainer -- it just also waits for the result instead of returning instantly.
        Prints a JSON array of RememberResult (one per chunk if the response was long enough for
        TextSegmenter to split it into multiple independent storage decisions -- one element in the
        common case) to stdout.
        """;

    public const string Recall = """
        psm-memory recall - answer a question by retrieving and ranking relevant stored memories.

        Usage: psm-memory recall --question <text> --user <id> [options]

        Required:
          --question <text>     The question to plan and rank memories for.
          --user <id>           User id memories are scoped to.

        Optional:
          --top-k <n>           Max memories to return (default determined by the recall plan, usually 5).
          --db <path>           SQLite db path (default: user_memory.db in the current directory).
        {{DOMAIN}}
        {{MODEL_DIR}}
        {{DAEMON}}

        Prints the RecallResult as indented JSON to stdout.
        """;

    public const string Context = """
        psm-memory context - retrieve relevant memories to ground a prompt (lower relevance
        threshold than recall, intended for pre-loading context rather than answering a question).

        Usage: psm-memory context --prompt <text> --user <id> [options]

        Required:
          --prompt <text>       The prompt to plan and rank memories for.
          --user <id>           User id memories are scoped to.

        Optional:
          --top-k <n>           Max memories to return (default determined by the recall plan, usually 5).
          --db <path>           SQLite db path (default: user_memory.db in the current directory).
        {{DOMAIN}}
        {{MODEL_DIR}}
        {{DAEMON}}

        Prints the RecallResult as indented JSON to stdout.
        """;

    public const string Serve = """
        psm-memory serve - load the model once, then answer many remember/recall/context requests
        read as NDJSON lines from stdin, writing one NDJSON response line per request to stdout.

        Built for batch workloads (e.g. running a benchmark's full ingest+recall loop) where paying
        a fresh model+adapter load per call would dominate total runtime -- the model loads exactly
        once here, then every request reuses it.

        Usage: psm-memory serve [options]

        Request (one JSON object per stdin line):
          {"id":"<opaque>","cmd":"remember"|"recall"|"context", ...fields}
          remember fields: llmResponse (required), userId (required), userMessage, includeExistingMemories
                           (bool, default true), extraTags (string[]), source ({sourceKind,sourceId,
                           sourceLabel,sourceTimestamp}), domain ("coding"|"conversational", default "coding").
          recall fields:   question (required), userId (required), topK, domain.
          context fields:  prompt (required), userId (required), topK, domain.

        Response (one JSON object per stdout line):
          {"id":"<echoed>","ok":true,"result":{...RecallResult...}}                 (recall/context)
          {"id":"<echoed>","ok":true,"result":[{...RememberResult...}, ...]}        (remember -- an
              array: one element per chunk if the response was long enough for TextSegmenter to
              split it into multiple independent storage decisions, one element in the common case)
          {"id":"<echoed>","ok":false,"error":"<message>"}     (a bad request does not crash the server)

        Optional:
          --db <path>           SQLite db path (default: user_memory.db in the current directory).
        {{MODEL_DIR}}
        {{DAEMON}}

        Runs until stdin closes (EOF), then exits 0. A "psm-memory serve: model loaded, ready for
        NDJSON requests on stdin." line is printed to STDERR (not stdout) once startup is complete,
        so a driver can block on that line before sending its first request.
        """;

    public const string Show = """
        psm-memory show - print stored memories for a user from one table.

        Usage: psm-memory show --user <id> [options]

        Required:
          --user <id>           User id to show memories for.

        Optional:
          --table <name>        One of episodic|semantic|archival (default: episodic).
          --limit <n>           Max rows to return (default: 20).
          --db <path>           SQLite db path (default: user_memory.db in the current directory).

        Does not load the GGUF model (no --model-dir needed). Prints a JSON array of memory records.
        """;

    public const string Conflicts = """
        psm-memory conflicts - print logged memory conflicts by status.

        Usage: psm-memory conflicts [options]

        Optional:
          --status <name>       One of unresolved|resolved|dismissed (default: unresolved).
          --limit <n>           Max rows to return (default: 20).
          --db <path>           SQLite db path (default: user_memory.db in the current directory).

        Does not load the GGUF model (no --model-dir needed). Prints a JSON array of conflict rows.
        """;

    public const string Init = """
        psm-memory init (alias: migrate) - create or update the SQLite schema at --db.

        Usage: psm-memory init [options]

        Optional:
          --db <path>           SQLite db path (default: user_memory.db in the current directory).

        Safe to run repeatedly (uses CREATE TABLE IF NOT EXISTS / adds missing columns only).
        Does not load the GGUF model.
        """;

    public const string EnqueueRemember = """
        psm-memory enqueue-remember - queue a remember request for background processing (the
        fire-and-forget path). Returns instantly with a pending id; does NOT run the storage
        decision itself -- see 'drain-queue' (or the MCP server's own background worker) for that.

        Usage: psm-memory enqueue-remember --message <text> --user <id> [options]

        Required:
          --message <text>      The assistant/LLM response text to extract durable memory from.
          --user <id>           User id memories are scoped to.

        Optional:
          --user-message <text> The user's message that prompted the response.
          --extra-tags <a,b,c>  Comma-separated tags appended to any memory that gets written.
          --no-existing         Skip loading existing memories before deciding (IncludeExistingMemories=false).
          --source-kind <k>     Source metadata override: kind (e.g. "chat", "email").
          --source-id <id>      Source metadata override: id.
          --source-label <s>    Source metadata override: label.
          --source-timestamp <t> Source metadata override: ISO timestamp.
          --db <path>           SQLite db path (default: user_memory.db in the current directory).
        {{DOMAIN}}

        Does not load the GGUF model (no --model-dir needed) -- this is a pure store write.
        Prints {"id":"...","status":"pending"} to stdout.
        """;

    public const string DrainQueue = """
        psm-memory drain-queue - process one batch of the queued remember requests right now, using
        the same synchronous PsmService.RememberAsync pipeline a direct 'remember' call would use
        (chunked via TextSegmenter first if a queued response is long).

        Usage: psm-memory drain-queue [options]

        Optional:
          --batch-size <n>       Max pending rows to process this call (default: 10).
          --db <path>           SQLite db path (default: user_memory.db in the current directory).
        {{MODEL_DIR}}

        Does not loop -- call repeatedly (e.g. in a test/verification script) to drain a larger
        backlog. Prints {"processed": <n>} to stdout.
        """;

    public const string Hook = """
        psm-memory hook - agent-harness integration points, invoked by an installed hook (see
        'install-agent') on every prompt/session event. Reads a JSON blob from stdin (tolerant of
        empty/invalid input); options come from CLI flags only, never from stdin.

        Usage: psm-memory hook recall|remember|session-start|session-end [options]

        Modes:
          recall (alias context)  Ground the current prompt: extracts a prompt string from stdin
                                   JSON (prompt|user_prompt|message|input, first non-empty wins) and
                                   prints rendered context to stdout (or a Gemini BeforeAgent JSON
                                   envelope with --agent gemini).
          remember                 Resolves the assistant's last response from stdin JSON fields,
                                   else a transcript_path's tail, else the newest Codex session file,
                                   and enqueues it (fire-and-forget) via the same remember queue every
                                   caller uses. Silently does nothing if no response can be resolved.
          session-start/session-end  Builds a short session summary (agent, package.json name,
                                   git repo/branch/dirty files, transcript path) and enqueues it the
                                   same way.

        Optional:
          --agent <claude|codex|gemini>  Selects output shape (Gemini's hook contract differs).
          --user <id>            Defaults to the OS user name if omitted (hooks are installed without
                                  --user by default).
          --top-k <n>            recall only: max memories to consider.
          --domain <name>        coding|conversational (default coding); falls back to coding on an
                                  unrecognized value rather than failing the hook.
          --db <path>            SQLite db path (default: user_memory.db in the current directory).
        {{MODEL_DIR}}
        {{DAEMON}}

        Every invocation appends one JSON line to a hook audit log at
        <directory of --db>/psm-memory-hooks.jsonl (or $PSM_MEMORY_HOOK_LOG if set). Internal
        failures are caught and logged there -- this command always exits 0 so a bad hook run never
        breaks the calling agent's turn.
        """;

    public const string InstallAgent = """
        psm-memory install-agent - wires 'psm-memory hook ...' commands into a coding agent's global
        hook configuration.

        Usage: psm-memory install-agent <codex|claude|gemini|all[,agent,...]> [options]

        SAFETY: with no --dry-run and no --config-dir, this WRITES REAL GLOBAL CONFIG FILES
        (~/.codex/config.toml, ~/.codex/hooks.json, ~/.claude/settings.json, ~/.gemini/settings.json).
        Always try --dry-run first, and use --config-dir to point at a throwaway directory for any
        testing.

        Optional:
          --dry-run              Print what would be written; touch nothing.
          --config-dir <path>    Override the config root (writes <path>/.codex/..., <path>/.claude/...,
                                  <path>/.gemini/... instead of the real home directory). Required for
                                  any non-dry-run test invocation.

        Idempotent: re-running removes any previously-installed 'psm-memory hook ...' entries before
        adding fresh ones, so repeated installs never duplicate hooks.

        Prints {"installed": bool, "dryRun": bool, "agents": [{"agent", "configPath",
        "wouldWrite"|"wrote": <content>}, ...]} to stdout.
        """;

    public const string DaemonRun = """
        psm-memory daemon-run - internal: run the warm-host server directly (loads the model once,
        then serves storage-decision/recall-plan/consolidation-decision/embed requests over loopback
        HTTP until idle for its configured idle timeout, then shuts itself down). Not meant to be
        invoked by hand in normal use -- spawned automatically by other commands' --daemon path (see
        WarmHostClient.SpawnDetached) when no warm host is already running.

        Usage: psm-memory daemon-run --model-dir <path> --state-dir <path>

        Required:
          --model-dir <path>    GGUF model directory (see the note under other commands' --model-dir).
          --state-dir <path>    Directory for daemon.json/daemon.lock -- callers poll daemon.json here
                                to discover the running server's host/port.

        Runs until idle-timeout elapses with no requests, or until killed.
        """;

    public static string For(string body) => body
        .Replace("{{MODEL_DIR}}", ModelDirNote)
        .Replace("{{DOMAIN}}", DomainNote)
        .Replace("{{DAEMON}}", DaemonNote);
}
