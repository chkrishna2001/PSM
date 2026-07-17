using System.Text.Json;
using System.Text.Json.Serialization;
using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;

namespace PsmMemory.Cli;

internal static class Commands
{
    /// <summary>Parses --domain coding|conversational (default coding) into <see cref="PsmDomain"/>.</summary>
    private static PsmDomain ParseDomain(ArgParser parsed) => ParseDomainString(parsed.GetString("domain", "coding"));

    private static PsmDomain ParseDomainString(string? raw) => (raw ?? "coding").Trim().ToLowerInvariant() switch
    {
        "coding" => PsmDomain.Coding,
        "conversational" => PsmDomain.Conversational,
        _ => throw new CliUsageException($"domain must be one of coding|conversational, got '{raw}'."),
    };

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

    private static readonly JsonSerializerOptions ServeJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>
    /// One request per stdin line, one response per stdout line (NDJSON) -- lets a batch driver
    /// (e.g. the LoCoMo benchmark harness) issue thousands of remember/recall/context calls against
    /// ONE loaded model instance instead of paying a full model + adapter load per CLI invocation
    /// (which would otherwise dominate total runtime for any benchmark-sized workload).
    /// Request shape: {"id":"...","cmd":"remember"|"recall"|"context", ...fields matching the
    /// corresponding *Request DTO in camelCase, e.g. "llmResponse"/"userId"/"domain"}.
    /// Response shape: {"id":"...","ok":true,"result":{...}} or {"id":"...","ok":false,"error":"..."}.
    /// </summary>
    public static async Task<int> RunServeAsync(string[] args)
    {
        var parsed = ArgParser.Parse(args);
        if (parsed.HasFlag("help")) { Console.WriteLine(HelpText.Serve); return 0; }

        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var modelDir = parsed.GetString("model-dir", Defaults.ResolveModelDir());
        Defaults.EnsureModelDir(modelDir);

        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var runtime = await OnnxPsmRuntime.CreateAsync(modelDir).ConfigureAwait(false);
        var service = new PsmService(store, runtime);

        await Console.Error.WriteLineAsync("psm-memory serve: model loaded, ready for NDJSON requests on stdin.").ConfigureAwait(false);

        string? line;
        while ((line = await Console.In.ReadLineAsync().ConfigureAwait(false)) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string? id = null;
            try
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                id = root.TryGetProperty("id", out var idEl) ? idEl.GetString() : null;
                var cmd = root.TryGetProperty("cmd", out var cmdEl) ? cmdEl.GetString() : null;
                object result = cmd switch
                {
                    "remember" => await service.RememberAsync(ToRememberRequest(root)).ConfigureAwait(false),
                    "recall" => await service.RecallAsync(ToRecallRequest(root)).ConfigureAwait(false),
                    "context" => await service.ContextAsync(ToContextRequest(root)).ConfigureAwait(false),
                    _ => throw new CliUsageException($"serve: unknown cmd '{cmd}' (expected remember|recall|context)."),
                };
                Console.WriteLine(JsonSerializer.Serialize(new { id, ok = true, result }, ServeJsonOptions));
            }
            catch (Exception ex)
            {
                Console.WriteLine(JsonSerializer.Serialize(new { id, ok = false, error = ex.Message }, ServeJsonOptions));
            }
            Console.Out.Flush();
        }
        return 0;
    }

    private static string? GetStr(JsonElement root, string name) =>
        root.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.String ? el.GetString() : null;

    private static int? GetIntOpt(JsonElement root, string name) =>
        root.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.Number ? el.GetInt32() : null;

    private static RememberRequest ToRememberRequest(JsonElement root)
    {
        var request = new RememberRequest
        {
            LlmResponse = GetStr(root, "llmResponse") ?? throw new CliUsageException("remember: missing 'llmResponse'."),
            UserId = GetStr(root, "userId") ?? throw new CliUsageException("remember: missing 'userId'."),
            UserMessage = GetStr(root, "userMessage"),
            IncludeExistingMemories = !root.TryGetProperty("includeExistingMemories", out var incEl) || incEl.GetBoolean(),
            Domain = ParseDomainString(GetStr(root, "domain")),
        };
        if (root.TryGetProperty("extraTags", out var tagsEl) && tagsEl.ValueKind == JsonValueKind.Array)
        {
            request.ExtraTags = tagsEl.EnumerateArray().Select(t => t.GetString() ?? "").Where(t => t.Length > 0).ToList();
        }
        if (root.TryGetProperty("source", out var sourceEl) && sourceEl.ValueKind == JsonValueKind.Object)
        {
            request.Source = new MemorySourceMetadata
            {
                SourceKind = GetStr(sourceEl, "sourceKind"),
                SourceId = GetStr(sourceEl, "sourceId"),
                SourceLabel = GetStr(sourceEl, "sourceLabel"),
                SourceTimestamp = GetStr(sourceEl, "sourceTimestamp"),
            };
        }
        return request;
    }

    private static RecallRequest ToRecallRequest(JsonElement root) => new()
    {
        Question = GetStr(root, "question") ?? throw new CliUsageException("recall: missing 'question'."),
        UserId = GetStr(root, "userId") ?? throw new CliUsageException("recall: missing 'userId'."),
        TopK = GetIntOpt(root, "topK"),
        Domain = ParseDomainString(GetStr(root, "domain")),
    };

    private static ContextRequest ToContextRequest(JsonElement root) => new()
    {
        Prompt = GetStr(root, "prompt") ?? throw new CliUsageException("context: missing 'prompt'."),
        UserId = GetStr(root, "userId") ?? throw new CliUsageException("context: missing 'userId'."),
        TopK = GetIntOpt(root, "topK"),
        Domain = ParseDomainString(GetStr(root, "domain")),
    };
}
