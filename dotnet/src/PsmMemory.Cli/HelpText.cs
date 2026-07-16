namespace PsmMemory.Cli;

internal static class HelpText
{
    public const string ModelDirNote =
        "  --model-dir <path>    Directory containing model.onnx and adapters/*.onnx_adapter,\n" +
        "                        produced by psm-model/scripts/convert_adapters_onnx.py.\n" +
        "                        Defaults to psm-model/prod-memory/onnx-runtime/v1 resolved against\n" +
        "                        the repo root (found by walking up from the executable). When this\n" +
        "                        tool is installed elsewhere (e.g. `dotnet tool install --global`),\n" +
        "                        that auto-detection will not find the repo and you MUST pass this\n" +
        "                        flag explicitly.";

    public const string DomainNote =
        "  --domain <name>       One of coding|conversational (default: coding). Selects which\n" +
        "                        trained adapter set to use for this call. conversational requires\n" +
        "                        conversational_*.onnx_adapter files to exist in --model-dir/adapters/\n" +
        "                        (not yet trained as of this writing) -- requesting it before those\n" +
        "                        exist fails with a clear error.";

    public const string Root = """
        psm-memory - thin CLI wrapper around the PsmMemory.Core SDK (local-first PSM memory store
        backed by an in-process ONNX Runtime GenAI model with storage/retrieval_plan/consolidation
        LoRA adapters).

        Usage: psm-memory <command> [options]

        Commands:
          remember    Analyze an assistant response and store durable memory from it.
          recall      Answer a question by retrieving and ranking relevant stored memories.
          context     Retrieve relevant memories to ground a prompt (lower relevance threshold than recall).
          show        Print stored memories for a user from one table (episodic/semantic/archival).
          conflicts   Print logged memory conflicts by status.
          init        Create (or update) the SQLite schema at --db. Alias: migrate.
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

        Prints the RememberResult as indented JSON to stdout.
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

        Prints the RecallResult as indented JSON to stdout.
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

        Does not load the ONNX model (no --model-dir needed). Prints a JSON array of memory records.
        """;

    public const string Conflicts = """
        psm-memory conflicts - print logged memory conflicts by status.

        Usage: psm-memory conflicts [options]

        Optional:
          --status <name>       One of unresolved|resolved|dismissed (default: unresolved).
          --limit <n>           Max rows to return (default: 20).
          --db <path>           SQLite db path (default: user_memory.db in the current directory).

        Does not load the ONNX model (no --model-dir needed). Prints a JSON array of conflict rows.
        """;

    public const string Init = """
        psm-memory init (alias: migrate) - create or update the SQLite schema at --db.

        Usage: psm-memory init [options]

        Optional:
          --db <path>           SQLite db path (default: user_memory.db in the current directory).

        Safe to run repeatedly (uses CREATE TABLE IF NOT EXISTS / adds missing columns only).
        Does not load the ONNX model.
        """;

    public static string For(string body) => body.Replace("{{MODEL_DIR}}", ModelDirNote).Replace("{{DOMAIN}}", DomainNote);
}
