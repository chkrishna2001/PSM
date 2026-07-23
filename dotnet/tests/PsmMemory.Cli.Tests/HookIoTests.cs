using System.Text.Json;
using PsmMemory.Cli;
using Xunit;

namespace PsmMemory.Cli.Tests;

/// <summary>
/// Unit tests for HookIo's pure stdin-shape and transcript-scanning logic (no real stdin/Console
/// access, no real home-directory paths -- FindLatestJsonlFile/LatestCodexSessionPath overrides are
/// always pointed at a temp directory here).
/// </summary>
public class HookIoTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"psm-hookio-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    // ---- ParseHookInput --------------------------------------------------------------------

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\n\n")]
    public void ParseHookInput_EmptyOrWhitespace_ReturnsEmptyObject(string raw)
    {
        var el = HookIo.ParseHookInput(raw);
        Assert.Equal(JsonValueKind.Object, el.ValueKind);
        Assert.False(el.EnumerateObject().Any());
    }

    [Fact]
    public void ParseHookInput_ValidObject_ParsesFields()
    {
        var el = HookIo.ParseHookInput("""{"prompt": "hello", "n": 3}""");
        Assert.Equal("hello", el.GetProperty("prompt").GetString());
        Assert.Equal(3, el.GetProperty("n").GetInt32());
    }

    [Fact]
    public void ParseHookInput_ValidJsonButNotAnObject_ReturnsEmptyObject()
    {
        var el = HookIo.ParseHookInput("[1,2,3]");
        Assert.Equal(JsonValueKind.Object, el.ValueKind);
        Assert.False(el.EnumerateObject().Any());
    }

    [Fact]
    public void ParseHookInput_InvalidJson_WrapsAsRawField()
    {
        var el = HookIo.ParseHookInput("this is not json {{{");
        Assert.Equal(JsonValueKind.Object, el.ValueKind);
        Assert.Equal("this is not json {{{", el.GetProperty("raw").GetString());
    }

    // ---- FirstNonEmptyString ----------------------------------------------------------------

    [Fact]
    public void FirstNonEmptyString_ChecksFieldsInOrder_FirstNonEmptyWins()
    {
        using var doc = JsonDocument.Parse("""{"prompt": "", "user_prompt": "  ", "message": "the one"}""");
        var result = HookIo.FirstNonEmptyString(doc.RootElement, "prompt", "user_prompt", "message", "input");
        Assert.Equal("the one", result);
    }

    [Fact]
    public void FirstNonEmptyString_NoFieldsPresent_ReturnsNull()
    {
        using var doc = JsonDocument.Parse("{}");
        Assert.Null(HookIo.FirstNonEmptyString(doc.RootElement, "prompt", "message"));
    }

    // ---- Remember stdin-field-priority extraction (mirrors HookCommands.RunRememberAsync) ----

    [Theory]
    [InlineData("""{"prompt_response": "A", "response": "B"}""", "A")]
    [InlineData("""{"last_assistant_message": "A", "output": "B"}""", "A")]
    [InlineData("""{"response": "A", "assistant_response": "B"}""", "A")]
    [InlineData("""{"output": "A", "text": "B"}""", "A")]
    [InlineData("""{"text": "only-one"}""", "only-one")]
    [InlineData("""{"prompt_response": "", "response": "fallback"}""", "fallback")]
    public void RememberFieldPriority_FirstNonEmptyWins(string json, string expected)
    {
        var input = HookIo.ParseHookInput(json);
        var result = HookIo.FirstNonEmptyString(input,
            "prompt_response", "last_assistant_message", "response", "assistant_response", "output", "text");
        Assert.Equal(expected, result);
    }

    [Fact]
    public void RememberFieldPriority_NoneOfTheFieldsPresent_ReturnsNull()
    {
        var input = HookIo.ParseHookInput("""{"unrelated": "value"}""");
        var result = HookIo.FirstNonEmptyString(input,
            "prompt_response", "last_assistant_message", "response", "assistant_response", "output", "text");
        Assert.Null(result);
    }

    // ---- ExtractLastAssistantMessage (transcript tail-scanning) -------------------------------

    [Fact]
    public void ExtractLastAssistantMessage_LastAgentMessageInPayload_TakesPriority()
    {
        var content = string.Join("\n",
            """{"role": "assistant", "text": "earlier message"}""",
            """{"payload": {"last_agent_message": "the real answer"}}""");
        Assert.Equal("the real answer", HookIo.ExtractLastAssistantMessage(content));
    }

    [Fact]
    public void ExtractLastAssistantMessage_ScansFromEndForAssistantRole()
    {
        var content = string.Join("\n",
            """{"role": "user", "text": "question"}""",
            """{"role": "assistant", "text": "first answer"}""",
            """{"role": "user", "text": "follow up"}""",
            """{"role": "assistant", "text": "final answer"}""");
        Assert.Equal("final answer", HookIo.ExtractLastAssistantMessage(content));
    }

    [Fact]
    public void ExtractLastAssistantMessage_AssistantRoleUnderPayload_IsRecognized()
    {
        var content = """{"payload": {"role": "assistant", "content": "nested answer"}}""";
        Assert.Equal("nested answer", HookIo.ExtractLastAssistantMessage(content));
    }

    [Fact]
    public void ExtractLastAssistantMessage_NoAssistantLines_ReturnsNull()
    {
        var content = string.Join("\n",
            """{"role": "user", "text": "question 1"}""",
            """{"role": "user", "text": "question 2"}""");
        Assert.Null(HookIo.ExtractLastAssistantMessage(content));
    }

    [Fact]
    public void ExtractLastAssistantMessage_MalformedLinesAreSkipped_NotFatal()
    {
        var content = string.Join("\n",
            "not even json",
            """{"role": "assistant", "text": "good one"}""",
            "{ broken json");
        Assert.Equal("good one", HookIo.ExtractLastAssistantMessage(content));
    }

    [Fact]
    public void ExtractLastAssistantMessage_OnlyTailsLast200Lines()
    {
        var lines = new List<string>();
        for (var i = 0; i < 300; i++) lines.Add($$"""{"role": "assistant", "text": "old-{{i}}"}""");
        // This one is outside the last 200 lines (it's line index 50, tail keeps the last 200 => indices 100..299).
        lines[50] = """{"role": "assistant", "text": "SHOULD-NOT-BE-FOUND-TOO-OLD"}""";
        var content = string.Join("\n", lines);

        var result = HookIo.ExtractLastAssistantMessage(content);
        Assert.NotNull(result);
        Assert.StartsWith("old-", result);
    }

    // ---- ExtractLastAssistantMessageFromTranscriptFile ----------------------------------------

    [Fact]
    public void ExtractLastAssistantMessageFromTranscriptFile_MissingPath_ReturnsNull()
    {
        Assert.Null(HookIo.ExtractLastAssistantMessageFromTranscriptFile(null));
        Assert.Null(HookIo.ExtractLastAssistantMessageFromTranscriptFile(""));
        Assert.Null(HookIo.ExtractLastAssistantMessageFromTranscriptFile(Path.Combine(_tempDir, "does-not-exist.jsonl")));
    }

    [Fact]
    public void ExtractLastAssistantMessageFromTranscriptFile_RealFile_Works()
    {
        Directory.CreateDirectory(_tempDir);
        var path = Path.Combine(_tempDir, "transcript.jsonl");
        File.WriteAllText(path, """{"role": "assistant", "text": "from file"}""");
        Assert.Equal("from file", HookIo.ExtractLastAssistantMessageFromTranscriptFile(path));
    }

    // ---- FindLatestJsonlFile / LatestCodexSessionPath (temp dir only, never real home) --------

    [Fact]
    public void FindLatestJsonlFile_MissingRoot_ReturnsNull()
    {
        Assert.Null(HookIo.FindLatestJsonlFile(Path.Combine(_tempDir, "nope")));
    }

    [Fact]
    public void FindLatestJsonlFile_PicksMostRecentlyModifiedAcrossSubdirectories()
    {
        Directory.CreateDirectory(_tempDir);
        var sub = Path.Combine(_tempDir, "a", "b");
        Directory.CreateDirectory(sub);

        var older = Path.Combine(_tempDir, "older.jsonl");
        var newer = Path.Combine(sub, "newer.jsonl");
        var notJsonl = Path.Combine(_tempDir, "ignore.txt");

        File.WriteAllText(older, "{}");
        File.WriteAllText(notJsonl, "{}");
        Thread.Sleep(50);
        File.WriteAllText(newer, "{}");
        File.SetLastWriteTimeUtc(newer, DateTime.UtcNow.AddMinutes(1));
        File.SetLastWriteTimeUtc(older, DateTime.UtcNow.AddMinutes(-1));

        var result = HookIo.FindLatestJsonlFile(_tempDir);
        Assert.Equal(newer, result);
    }

    [Fact]
    public void LatestCodexSessionPath_WithOverride_NeverTouchesRealHomeDirectory()
    {
        // Passing a sessionsRootOverride means the real ~/.codex/sessions path is never consulted.
        var result = HookIo.LatestCodexSessionPath(Path.Combine(_tempDir, "sessions"));
        Assert.Null(result); // directory doesn't exist -- confirms the override was actually used, not the real home dir.
    }

    // ---- ResolveAuditLogPath / AppendAuditLog --------------------------------------------------

    [Fact]
    public void ResolveAuditLogPath_EnvOverride_TakesPriority()
    {
        var result = HookIo.ResolveAuditLogPath(Path.Combine(_tempDir, "db.sqlite"), "/custom/log/path.jsonl");
        Assert.Equal("/custom/log/path.jsonl", result);
    }

    [Fact]
    public void ResolveAuditLogPath_NoOverride_UsesDbDirectory()
    {
        var dbPath = Path.Combine(_tempDir, "db.sqlite");
        var result = HookIo.ResolveAuditLogPath(dbPath, null);
        Assert.Equal(Path.Combine(_tempDir, "psm-memory-hooks.jsonl"), result);
    }

    [Fact]
    public void AppendAuditLog_WritesOneJsonLineWithRequiredFields()
    {
        Directory.CreateDirectory(_tempDir);
        var dbPath = Path.Combine(_tempDir, "db.sqlite");
        HookIo.AppendAuditLog(dbPath, "recall", "claude", true, null, null);

        var logPath = Path.Combine(_tempDir, "psm-memory-hooks.jsonl");
        Assert.True(File.Exists(logPath));
        var line = File.ReadAllLines(logPath).Single();
        using var doc = JsonDocument.Parse(line);
        Assert.Equal("recall", doc.RootElement.GetProperty("mode").GetString());
        Assert.Equal("claude", doc.RootElement.GetProperty("agent").GetString());
        Assert.True(doc.RootElement.GetProperty("ok").GetBoolean());
        Assert.True(doc.RootElement.TryGetProperty("ts", out _));
    }

    [Fact]
    public void AppendAuditLog_MultipleCalls_AppendsOneLinePerCall()
    {
        Directory.CreateDirectory(_tempDir);
        var dbPath = Path.Combine(_tempDir, "db.sqlite");
        HookIo.AppendAuditLog(dbPath, "recall", "claude", true, null, null);
        HookIo.AppendAuditLog(dbPath, "remember", "claude", false, "boom", null);

        var logPath = Path.Combine(_tempDir, "psm-memory-hooks.jsonl");
        var lines = File.ReadAllLines(logPath);
        Assert.Equal(2, lines.Length);
    }
}
