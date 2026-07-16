using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;

namespace PsmMemory.Cli;

internal static class Commands
{
    /// <summary>Parses --domain coding|conversational (default coding) into <see cref="PsmDomain"/>.</summary>
    private static PsmDomain ParseDomain(ArgParser parsed)
    {
        var raw = parsed.GetString("domain", "coding");
        return raw.Trim().ToLowerInvariant() switch
        {
            "coding" => PsmDomain.Coding,
            "conversational" => PsmDomain.Conversational,
            _ => throw new CliUsageException($"--domain must be one of coding|conversational, got '{raw}'."),
        };
    }

    public static async Task<int> RunRememberAsync(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.For(HelpText.Remember)); return 0; }

        var message = parsed.GetRequiredString("message");
        var userId = parsed.GetRequiredString("user");
        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var modelDir = parsed.GetString("model-dir", Defaults.ResolveModelDir());
        Defaults.EnsureModelDir(modelDir);
        var domain = ParseDomain(parsed);

        var request = new RememberRequest
        {
            LlmResponse = message,
            UserId = userId,
            UserMessage = parsed.GetString("user-message"),
            IncludeExistingMemories = !parsed.HasFlag("no-existing"),
            Domain = domain
        };

        var extraTags = parsed.GetString("extra-tags");
        if (extraTags is not null)
        {
            request.ExtraTags = extraTags
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .ToList();
        }

        var sourceKind = parsed.GetString("source-kind");
        var sourceId = parsed.GetString("source-id");
        var sourceLabel = parsed.GetString("source-label");
        var sourceTimestamp = parsed.GetString("source-timestamp");
        if (sourceKind is not null || sourceId is not null || sourceLabel is not null || sourceTimestamp is not null)
        {
            request.Source = new MemorySourceMetadata
            {
                SourceKind = sourceKind,
                SourceId = sourceId,
                SourceLabel = sourceLabel,
                SourceTimestamp = sourceTimestamp
            };
        }

        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var runtime = await OnnxPsmRuntime.CreateAsync(modelDir).ConfigureAwait(false);
        var service = new PsmService(store, runtime);

        var result = await service.RememberAsync(request).ConfigureAwait(false);
        CliRunner.PrintJson(result);
        return 0;
    }

    public static async Task<int> RunRecallAsync(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.For(HelpText.Recall)); return 0; }

        var question = parsed.GetRequiredString("question");
        var userId = parsed.GetRequiredString("user");
        var topK = parsed.GetInt("top-k");
        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var modelDir = parsed.GetString("model-dir", Defaults.ResolveModelDir());
        Defaults.EnsureModelDir(modelDir);
        var domain = ParseDomain(parsed);

        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var runtime = await OnnxPsmRuntime.CreateAsync(modelDir).ConfigureAwait(false);
        var service = new PsmService(store, runtime);

        var result = await service.RecallAsync(new RecallRequest { Question = question, UserId = userId, TopK = topK, Domain = domain })
            .ConfigureAwait(false);
        CliRunner.PrintJson(result);
        return 0;
    }

    public static async Task<int> RunContextAsync(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.For(HelpText.Context)); return 0; }

        var prompt = parsed.GetRequiredString("prompt");
        var userId = parsed.GetRequiredString("user");
        var topK = parsed.GetInt("top-k");
        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var modelDir = parsed.GetString("model-dir", Defaults.ResolveModelDir());
        Defaults.EnsureModelDir(modelDir);
        var domain = ParseDomain(parsed);

        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var runtime = await OnnxPsmRuntime.CreateAsync(modelDir).ConfigureAwait(false);
        var service = new PsmService(store, runtime);

        var result = await service.ContextAsync(new ContextRequest { Prompt = prompt, UserId = userId, TopK = topK, Domain = domain })
            .ConfigureAwait(false);
        CliRunner.PrintJson(result);
        return 0;
    }

    public static int RunShow(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.Show); return 0; }

        var table = parsed.GetString("table", MemoryTables.Episodic);
        if (table is not (MemoryTables.Episodic or MemoryTables.Semantic or MemoryTables.Archival))
            throw new CliUsageException($"--table must be one of episodic|semantic|archival, got '{table}'.");

        var userId = parsed.GetRequiredString("user");
        var limit = parsed.GetInt("limit", 20);
        var dbPath = parsed.GetString("db", Defaults.DbPath);

        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        var records = store.SelectMemories(userId, new[] { table }, limit);
        CliRunner.PrintJson(records);
        return 0;
    }

    public static int RunConflicts(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.Conflicts); return 0; }

        var status = parsed.GetString("status", "unresolved");
        if (status is not ("unresolved" or "resolved" or "dismissed"))
            throw new CliUsageException($"--status must be one of unresolved|resolved|dismissed, got '{status}'.");

        var limit = parsed.GetInt("limit", 20);
        var dbPath = parsed.GetString("db", Defaults.DbPath);

        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        var conflicts = store.SelectConflicts(status, limit);
        CliRunner.PrintJson(conflicts);
        return 0;
    }

    public static int RunInit(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.Init); return 0; }

        var dbPath = parsed.GetString("db", Defaults.DbPath);
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        Console.WriteLine($"Initialized PSM memory schema at '{Path.GetFullPath(dbPath)}'.");
        return 0;
    }
}
