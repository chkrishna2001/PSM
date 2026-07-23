using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace PsmMemory.Cli;

/// <summary>
/// `psm-memory install-agent codex|claude|gemini|all[,agent...] [--dry-run] [--config-dir &lt;path&gt;]`
/// -- ported from src/psm-cli/src/index.ts's installAgents()/installCodexHooks()/installClaudeHooks()/
/// installGeminiHooks()/ensureCodexHooksFeature()/removeOldPsmHooks()/addHook().
///
/// SAFETY: this command writes to real, global user configuration files (~/.claude/settings.json,
/// ~/.gemini/settings.json, ~/.codex/config.toml + ~/.codex/hooks.json) when run for real (no
/// --dry-run) with no --config-dir override. --config-dir exists specifically so this can be tested
/// end-to-end against a throwaway directory instead. Every path resolver below
/// (CodexConfigPath/CodexHooksPath/ClaudeSettingsPath/GeminiSettingsPath) only falls back to a real
/// home-directory path when configDir is null -- tests must always pass a --config-dir (or call the
/// resolvers directly with a non-null configDir).
/// </summary>
internal static partial class InstallAgentCommand
{
    internal static readonly string[] AllAgents = { "codex", "claude", "gemini" };

    private static readonly JsonSerializerOptions WriteOptions = new() { WriteIndented = true };

    public static int Run(string[] args)
    {
        string? positional = null;
        var remainder = args;
        if (args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal))
        {
            positional = args[0];
            remainder = args[1..];
        }

        var parsed = ArgParser.Parse(remainder);
        if (parsed.HasFlag("help"))
        {
            Console.WriteLine(HelpText.For(HelpText.InstallAgent));
            return 0;
        }

        // --agent flag takes precedence over the positional, matching TS's
        // stringOption(options, "agent", positionals[0] ?? "").
        var agentListRaw = parsed.GetString("agent") ?? positional ?? "";
        var agents = ParseAgentList(agentListRaw);

        var dryRun = parsed.HasFlag("dry-run");
        var configDir = parsed.GetString("config-dir");

        var results = new List<object>();
        foreach (var agent in agents)
        {
            results.AddRange(InstallAgent(agent, configDir, dryRun));
        }

        CliRunner.PrintJson(new
        {
            installed = !dryRun,
            dryRun,
            agents = results,
        });
        return 0;
    }

    /// <summary>Ported from TS's parseAgentList(): comma-separated, "all" expands to every known
    /// agent, order-preserving de-dup, throws on any unrecognized agent name.</summary>
    internal static List<string> ParseAgentList(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new CliUsageException("Usage: psm-memory install-agent codex|claude|gemini|all");

        var requested = value.Split(',').Select(a => a.Trim().ToLowerInvariant()).Where(a => a.Length > 0).ToList();
        var expanded = requested.Contains("all") ? AllAgents.ToList() : requested;
        var unknown = expanded.Where(a => !AllAgents.Contains(a)).ToList();
        if (unknown.Count > 0) throw new CliUsageException($"Unsupported agent: {string.Join(", ", unknown)}");
        return expanded.Distinct().ToList();
    }

    private static List<object> InstallAgent(string agent, string? configDir, bool dryRun) => agent switch
    {
        "codex" => InstallCodex(configDir, dryRun),
        "claude" => new List<object> { InstallSingleFileJson("claude", ClaudeSettingsPath(configDir), BuildClaudeSettingsJson, dryRun) },
        "gemini" => new List<object> { InstallSingleFileJson("gemini", GeminiSettingsPath(configDir), BuildGeminiSettingsJson, dryRun) },
        _ => throw new CliUsageException($"Unsupported agent: {agent}"),
    };

    private static List<object> InstallCodex(string? configDir, bool dryRun)
    {
        var configPath = CodexConfigPath(configDir);
        var existingConfigContent = SafeReadAllText(configPath);
        var newConfigContent = ComputeCodexConfigContent(existingConfigContent);

        var hooksPath = CodexHooksPath(configDir);
        var existingHooksJson = ParseJsonObjectTolerant(SafeReadAllText(hooksPath));
        var newHooksJson = BuildCodexHooksJson(existingHooksJson);
        var serializedHooksJson = JsonSerializer.Serialize(newHooksJson, WriteOptions);

        if (!dryRun)
        {
            var configDirName = Path.GetDirectoryName(configPath);
            if (!string.IsNullOrEmpty(configDirName)) Directory.CreateDirectory(configDirName);
            File.WriteAllText(configPath, newConfigContent);

            var hooksDirName = Path.GetDirectoryName(hooksPath);
            if (!string.IsNullOrEmpty(hooksDirName)) Directory.CreateDirectory(hooksDirName);
            File.WriteAllText(hooksPath, serializedHooksJson + "\n");
        }

        var key = dryRun ? "wouldWrite" : "wrote";
        return new List<object>
        {
            new Dictionary<string, object?> { ["agent"] = "codex", ["kind"] = "config", ["configPath"] = configPath, [key] = newConfigContent },
            new Dictionary<string, object?> { ["agent"] = "codex", ["kind"] = "hooks", ["configPath"] = hooksPath, [key] = newHooksJson },
        };
    }

    private static object InstallSingleFileJson(string agent, string path, Func<JsonObject, JsonObject> build, bool dryRun)
    {
        var existing = ParseJsonObjectTolerant(SafeReadAllText(path));
        var updated = build(existing);

        if (!dryRun)
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(path, JsonSerializer.Serialize(updated, WriteOptions) + "\n");
        }

        var key = dryRun ? "wouldWrite" : "wrote";
        return new Dictionary<string, object?> { ["agent"] = agent, ["configPath"] = path, [key] = updated };
    }

    // ---- Path resolution -------------------------------------------------------------------

    private static string RealHomeDir() => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    internal static string CodexConfigPath(string? configDir) =>
        configDir is not null ? Path.Combine(configDir, ".codex", "config.toml") : Path.Combine(RealHomeDir(), ".codex", "config.toml");

    internal static string CodexHooksPath(string? configDir) =>
        configDir is not null ? Path.Combine(configDir, ".codex", "hooks.json") : Path.Combine(RealHomeDir(), ".codex", "hooks.json");

    internal static string ClaudeSettingsPath(string? configDir) =>
        configDir is not null ? Path.Combine(configDir, ".claude", "settings.json") : Path.Combine(RealHomeDir(), ".claude", "settings.json");

    internal static string GeminiSettingsPath(string? configDir) =>
        configDir is not null ? Path.Combine(configDir, ".gemini", "settings.json") : Path.Combine(RealHomeDir(), ".gemini", "settings.json");

    // ---- Codex config.toml patching (regex patch, deliberately not a real TOML round-tripper) ----

    /// <summary>Ported from TS's ensureCodexHooksFeature(). Pure given the existing file content (or
    /// null if the file doesn't exist yet), so directly testable without touching disk.</summary>
    internal static string ComputeCodexConfigContent(string? existingContent)
    {
        if (existingContent is null) return "[features]\ncodex_hooks = true\n";
        if (CodexHooksFeatureLineRegex().IsMatch(existingContent)) return existingContent;
        if (CodexFeaturesSectionRegex().IsMatch(existingContent))
        {
            return CodexFeaturesSectionRegex().Replace(existingContent, "$1\ncodex_hooks = true", 1);
        }
        return existingContent.TrimEnd() + "\n\n[features]\ncodex_hooks = true\n";
    }

    [GeneratedRegex(@"^\s*codex_hooks\s*=\s*true\s*$", RegexOptions.Multiline)]
    private static partial Regex CodexHooksFeatureLineRegex();

    [GeneratedRegex(@"^(\[features\]\s*)$", RegexOptions.Multiline)]
    private static partial Regex CodexFeaturesSectionRegex();

    // ---- hooks.json / settings.json JSON merge logic ----------------------------------------

    /// <summary>Ported from TS's readJsonObject(): missing/empty/invalid content all tolerate to
    /// {}. Strips a leading UTF-8 BOM, matching TS's explicit BOM-stripping regex.</summary>
    internal static JsonObject ParseJsonObjectTolerant(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return new JsonObject();
        var content = raw.TrimStart('﻿');
        if (string.IsNullOrWhiteSpace(content)) return new JsonObject();
        try
        {
            return JsonNode.Parse(content) as JsonObject ?? new JsonObject();
        }
        catch
        {
            return new JsonObject();
        }
    }

    private static string? SafeReadAllText(string path)
    {
        if (!File.Exists(path)) return null;
        try
        {
            return File.ReadAllText(path);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Ported from TS's removeOldPsmHooks(): for every event's hook-entry array, drops any
    /// individual hook whose command matches a previously-installed psm-memory hook invocation, then
    /// drops any entry left with zero hooks. Mutates <paramref name="hooks"/> in place.</summary>
    internal static void RemoveOldPsmHooks(JsonObject hooks)
    {
        foreach (var eventName in hooks.Select(kv => kv.Key).ToList())
        {
            if (hooks[eventName] is not JsonArray entries) continue;

            var newEntries = new JsonArray();
            foreach (var entryNode in entries.ToList())
            {
                if (entryNode is not JsonObject entry)
                {
                    if (entryNode is not null) newEntries.Add(entryNode.DeepClone());
                    continue;
                }

                var childHooks = entry["hooks"] as JsonArray ?? new JsonArray();
                var filtered = new JsonArray();
                foreach (var hookNode in childHooks.ToList())
                {
                    if (hookNode is JsonObject hookObj
                        && hookObj["command"] is JsonValue cmdVal
                        && cmdVal.TryGetValue<string>(out var command)
                        && OldPsmHookCommandRegex().IsMatch(command))
                    {
                        continue; // drop: this is a previously-installed psm-memory hook entry.
                    }
                    if (hookNode is not null) filtered.Add(hookNode.DeepClone());
                }

                if (filtered.Count == 0) continue; // mirrors TS: entry.hooks now empty -> drop the whole entry.

                var clonedEntry = (JsonObject)entry.DeepClone();
                clonedEntry["hooks"] = filtered;
                newEntries.Add(clonedEntry);
            }

            hooks[eventName] = newEntries;
        }
    }

    // Includes the legacy "psm-codex-hook.ps1" pattern from an older, pre-CLI install method (ported
    // verbatim from TS's removeOldPsmHooks() regex) alongside the current "psm-memory hook ..." form,
    // so a reinstall cleans up both generations of previously-installed hook entries.
    [GeneratedRegex(@"psm-codex-hook\.ps1|psm-memory hook (context|recall|remember|session-start|session-end)", RegexOptions.IgnoreCase)]
    private static partial Regex OldPsmHookCommandRegex();

    /// <summary>Ported from TS's addHook(): appends one {matcher:"*", hooks:[{type:"command",
    /// command, ...extraFields}]} entry to the named event's array.</summary>
    internal static void AddHook(JsonObject hooks, string eventName, string command, JsonObject? extraFields = null)
    {
        var existing = hooks[eventName] is JsonArray existingArray ? (JsonArray)existingArray.DeepClone() : new JsonArray();

        var hookObj = new JsonObject { ["type"] = "command", ["command"] = command };
        if (extraFields is not null)
        {
            foreach (var kv in extraFields.ToList())
            {
                hookObj[kv.Key] = kv.Value?.DeepClone();
            }
        }

        var newEntry = new JsonObject { ["matcher"] = "*", ["hooks"] = new JsonArray(hookObj) };
        existing.Add(newEntry);
        hooks[eventName] = existing;
    }

    private static JsonObject BuildHooksSettings(JsonObject root, IEnumerable<(string EventName, string Command, JsonObject? Extra)> hookDefs)
    {
        var updated = (JsonObject)root.DeepClone();
        var hooks = updated["hooks"] is JsonObject existingHooks ? (JsonObject)existingHooks.DeepClone() : new JsonObject();
        RemoveOldPsmHooks(hooks);
        foreach (var (eventName, command, extra) in hookDefs)
        {
            AddHook(hooks, eventName, command, extra);
        }
        updated["hooks"] = hooks;
        return updated;
    }

    internal static JsonObject BuildCodexHooksJson(JsonObject existingHooksJson) => BuildHooksSettings(existingHooksJson, new (string, string, JsonObject?)[]
    {
        ("SessionStart", "psm-memory hook session-start", null),
        ("UserPromptSubmit", "psm-memory hook recall", null),
        ("Stop", "psm-memory hook remember", null),
        ("SessionEnd", "psm-memory hook session-end", null),
    });

    internal static JsonObject BuildClaudeSettingsJson(JsonObject existingSettings) => BuildHooksSettings(existingSettings, new (string, string, JsonObject?)[]
    {
        ("SessionStart", "psm-memory hook session-start", new JsonObject { ["async"] = true }),
        ("UserPromptSubmit", "psm-memory hook recall", null),
        ("Stop", "psm-memory hook remember", new JsonObject { ["async"] = true }),
        ("SessionEnd", "psm-memory hook session-end", new JsonObject { ["async"] = true }),
    });

    internal static JsonObject BuildGeminiSettingsJson(JsonObject existingSettings)
    {
        var updated = BuildHooksSettings(existingSettings, new (string, string, JsonObject?)[]
        {
            ("BeforeAgent", "psm-memory hook recall --agent gemini", null),
            ("AfterAgent", "psm-memory hook remember --agent gemini", null),
        });

        var hooksConfig = existingSettings["hooksConfig"] is JsonObject existingHooksConfig
            ? (JsonObject)existingHooksConfig.DeepClone()
            : new JsonObject();
        hooksConfig["enabled"] = true;
        updated["hooksConfig"] = hooksConfig;
        return updated;
    }
}
