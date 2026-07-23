using System.Text.Json;

namespace PsmMemory.Cli;

/// <summary>
/// Pure, stdin-shape and file-parsing helpers shared by every `psm-memory hook &lt;mode&gt;` command
/// (see HookCommands.cs). Ported behavior-for-behavior from src/psm-cli/src/index.ts's
/// readHookInput/firstText/transcriptAssistantText/latestCodexAssistantText/latestSessionPath/
/// walkJsonlFiles/writeHookAudit functions. Kept free of any actual stdin/Console access so every
/// piece here is directly unit-testable against strings/temp files/temp directories.
/// </summary>
internal static class HookIo
{
    /// <summary>
    /// Mirrors TS's readHookInput(): empty/whitespace-only input -&gt; {}; valid JSON that parses to a
    /// non-object (array, string, number, ...) -&gt; {}; invalid JSON -&gt; {"raw": &lt;original text&gt;}.
    /// </summary>
    public static JsonElement ParseHookInput(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return EmptyObject();
        try
        {
            using var doc = JsonDocument.Parse(raw);
            return doc.RootElement.ValueKind == JsonValueKind.Object ? doc.RootElement.Clone() : EmptyObject();
        }
        catch (JsonException)
        {
            using var doc = JsonDocument.Parse(JsonSerializer.Serialize(new { raw }));
            return doc.RootElement.Clone();
        }
    }

    private static JsonElement EmptyObject()
    {
        using var doc = JsonDocument.Parse("{}");
        return doc.RootElement.Clone();
    }

    /// <summary>Ported from TS's firstText(): returns the first named field whose value is a
    /// non-empty (after trim) JSON string, checked in the given order.</summary>
    public static string? FirstNonEmptyString(JsonElement input, params string[] names)
    {
        if (input.ValueKind != JsonValueKind.Object) return null;
        foreach (var name in names)
        {
            if (input.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String)
            {
                var s = value.GetString();
                if (!string.IsNullOrWhiteSpace(s)) return s;
            }
        }
        return null;
    }

    /// <summary>
    /// Ported from TS's transcriptAssistantText(): given the *content* of a transcript JSONL file,
    /// tail the last ~200 lines, reverse-scan for the last line whose parsed JSON looks like an
    /// assistant message, and extract its text. Checks (in order): a payload.last_agent_message
    /// string field; otherwise, for lines that look like an assistant event (role == "assistant" at
    /// the top level or under .payload), the first non-empty of text/content/message at the top
    /// level, then the same fields under .payload.
    /// </summary>
    public static string? ExtractLastAssistantMessage(string transcriptContent)
    {
        var allLines = transcriptContent.TrimEnd().Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
        var tailStart = Math.Max(0, allLines.Length - 200);
        for (var i = allLines.Length - 1; i >= tailStart; i--)
        {
            var line = allLines[i];
            if (string.IsNullOrWhiteSpace(line)) continue;

            JsonElement evt;
            try
            {
                using var doc = JsonDocument.Parse(line);
                evt = doc.RootElement.Clone();
            }
            catch (JsonException)
            {
                continue;
            }
            if (evt.ValueKind != JsonValueKind.Object) continue;

            if (evt.TryGetProperty("payload", out var payloadForLam)
                && payloadForLam.ValueKind == JsonValueKind.Object
                && payloadForLam.TryGetProperty("last_agent_message", out var lam)
                && lam.ValueKind == JsonValueKind.String)
            {
                var lamText = lam.GetString();
                if (!string.IsNullOrWhiteSpace(lamText)) return lamText;
            }

            if (!IsAssistantEvent(evt)) continue;

            var text = FirstNonEmptyString(evt, "text", "content", "message");
            if (text is not null) return text;

            if (evt.TryGetProperty("payload", out var payload) && payload.ValueKind == JsonValueKind.Object)
            {
                var payloadText = FirstNonEmptyString(payload, "text", "content", "message");
                if (payloadText is not null) return payloadText;
            }
        }
        return null;
    }

    private static bool IsAssistantEvent(JsonElement value)
    {
        if (value.ValueKind != JsonValueKind.Object) return false;
        if (value.TryGetProperty("role", out var role) && role.ValueKind == JsonValueKind.String && role.GetString() == "assistant")
            return true;
        return value.TryGetProperty("payload", out var payload) && payload.ValueKind == JsonValueKind.Object
            && payload.TryGetProperty("role", out var payloadRole) && payloadRole.ValueKind == JsonValueKind.String
            && payloadRole.GetString() == "assistant";
    }

    /// <summary>File-reading wrapper around <see cref="ExtractLastAssistantMessage"/>. Missing file,
    /// unreadable file, or null/empty path all resolve to null rather than throwing -- callers
    /// (hook remember) treat "nothing found" as a normal, silent skip.</summary>
    public static string? ExtractLastAssistantMessageFromTranscriptFile(string? path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;
        try
        {
            return ExtractLastAssistantMessage(File.ReadAllText(path));
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Ported from TS's walkJsonlFiles() + latestSessionPath(): finds the most-recently
    /// modified *.jsonl file under <paramref name="root"/>, walking up to <paramref name="maxDepth"/>
    /// levels deep. Pure over the given root so tests can point it at a temp directory instead of a
    /// real home-directory path.</summary>
    public static string? FindLatestJsonlFile(string root, int maxDepth = 4)
    {
        string? latestPath = null;
        var latestMtime = DateTime.MinValue;
        foreach (var path in WalkJsonlFiles(root, maxDepth))
        {
            DateTime mtime;
            try
            {
                mtime = File.GetLastWriteTimeUtc(path);
            }
            catch
            {
                continue;
            }
            if (mtime > latestMtime)
            {
                latestMtime = mtime;
                latestPath = path;
            }
        }
        return latestPath;
    }

    private static IEnumerable<string> WalkJsonlFiles(string root, int maxDepth)
    {
        if (maxDepth < 0 || !Directory.Exists(root)) yield break;

        IEnumerable<string> entries;
        try
        {
            entries = Directory.EnumerateFileSystemEntries(root).ToList();
        }
        catch
        {
            yield break;
        }

        foreach (var path in entries)
        {
            bool isDir;
            try
            {
                isDir = Directory.Exists(path);
            }
            catch
            {
                continue;
            }

            if (isDir)
            {
                foreach (var sub in WalkJsonlFiles(path, maxDepth - 1)) yield return sub;
            }
            else if (path.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase))
            {
                yield return path;
            }
        }
    }

    /// <summary>Ported from TS's latestSessionPath(): the newest *.jsonl file under
    /// ~/.codex/sessions (or <paramref name="sessionsRootOverride"/>, used by tests so real home
    /// directories are never touched).</summary>
    public static string? LatestCodexSessionPath(string? sessionsRootOverride = null)
    {
        var root = sessionsRootOverride
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".codex", "sessions");
        return FindLatestJsonlFile(root);
    }

    /// <summary>Ported from TS's defaultHookLogPath(): $PSM_MEMORY_HOOK_LOG if set, else
    /// &lt;directory of dbPath&gt;/psm-memory-hooks.jsonl. Pure given the env override value so tests
    /// don't need to mutate process-wide environment state.</summary>
    public static string ResolveAuditLogPath(string dbPath, string? envOverride)
    {
        if (!string.IsNullOrEmpty(envOverride)) return envOverride;
        var dir = Path.GetDirectoryName(Path.GetFullPath(dbPath));
        return Path.Combine(string.IsNullOrEmpty(dir) ? "." : dir, "psm-memory-hooks.jsonl");
    }

    /// <summary>Ported from TS's writeHookAudit(): appends one JSON line. Never throws -- hook
    /// logging must never break the calling agent's workflow.</summary>
    public static void AppendAuditLog(string dbPath, string mode, string? agent, bool ok, string? error, string? envOverride)
    {
        try
        {
            var path = ResolveAuditLogPath(dbPath, envOverride);
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            var entry = new Dictionary<string, object?>
            {
                ["ts"] = DateTime.UtcNow.ToString("o"),
                ["mode"] = mode,
                ["agent"] = agent,
                ["ok"] = ok,
            };
            if (error is not null) entry["error"] = error;

            File.AppendAllText(path, JsonSerializer.Serialize(entry) + "\n");
        }
        catch
        {
            // Hook logging must never break the agent workflow.
        }
    }
}
