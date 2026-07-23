using System.Text.Json.Nodes;
using PsmMemory.Cli;
using Xunit;

namespace PsmMemory.Cli.Tests;

/// <summary>
/// Exercises InstallAgentCommand's pure logic (agent-list parsing, TOML patching, JSON hook
/// merge/idempotency) plus a full --dry-run run against a temp --config-dir. Never touches a real
/// home-directory path -- every path resolver call here passes a non-null configDir, and Run() is
/// only ever invoked with --config-dir (never for real against this machine's actual home directory).
/// </summary>
public class InstallAgentCommandTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"psm-install-agent-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    // ---- ParseAgentList -------------------------------------------------------------------

    [Fact]
    public void ParseAgentList_SingleAgent_ReturnsThatAgent()
    {
        Assert.Equal(new[] { "codex" }, InstallAgentCommand.ParseAgentList("codex"));
    }

    [Fact]
    public void ParseAgentList_CommaSeparated_ReturnsAllInOrder()
    {
        Assert.Equal(new[] { "codex", "gemini" }, InstallAgentCommand.ParseAgentList("codex,gemini"));
    }

    [Fact]
    public void ParseAgentList_All_ExpandsToEveryAgent()
    {
        Assert.Equal(InstallAgentCommand.AllAgents, InstallAgentCommand.ParseAgentList("all"));
    }

    [Fact]
    public void ParseAgentList_Duplicates_AreDeduped()
    {
        Assert.Equal(new[] { "codex", "claude" }, InstallAgentCommand.ParseAgentList("codex,claude,codex"));
    }

    [Fact]
    public void ParseAgentList_UnknownAgent_Throws()
    {
        var ex = Assert.Throws<CliUsageException>(() => InstallAgentCommand.ParseAgentList("codex,nope"));
        Assert.Contains("nope", ex.Message);
    }

    [Fact]
    public void ParseAgentList_Empty_Throws()
    {
        Assert.Throws<CliUsageException>(() => InstallAgentCommand.ParseAgentList(""));
    }

    // ---- Codex TOML patching ---------------------------------------------------------------

    [Fact]
    public void ComputeCodexConfigContent_MissingFile_CreatesFeaturesBlock()
    {
        var content = InstallAgentCommand.ComputeCodexConfigContent(null);
        Assert.Equal("[features]\ncodex_hooks = true\n", content);
    }

    [Fact]
    public void ComputeCodexConfigContent_ExistingFileWithoutFeaturesSection_AppendsNewBlock()
    {
        var existing = "[other]\nfoo = 1\n";
        var content = InstallAgentCommand.ComputeCodexConfigContent(existing);
        Assert.Contains("[other]\nfoo = 1", content);
        Assert.Contains("[features]\ncodex_hooks = true", content);
        // The original content must not be mangled/reformatted -- only appended to.
        Assert.StartsWith("[other]\nfoo = 1", content);
    }

    [Fact]
    public void ComputeCodexConfigContent_ExistingFeaturesSectionMissingFlag_InsertsIntoIt()
    {
        var existing = "[features]\nother_flag = true\n\n[more]\nx = 1\n";
        var content = InstallAgentCommand.ComputeCodexConfigContent(existing);
        Assert.Contains("codex_hooks = true", content);
        // Must be inserted into the existing [features] block, not a second one appended.
        Assert.Single(System.Text.RegularExpressions.Regex.Matches(content, @"\[features\]"));
        Assert.Contains("other_flag = true", content); // untouched sibling key survives
        Assert.Contains("[more]\nx = 1", content); // untouched trailing section survives
    }

    [Fact]
    public void ComputeCodexConfigContent_FlagAlreadyPresent_ReturnsContentUnchanged()
    {
        var existing = "[features]\ncodex_hooks = true\n";
        var content = InstallAgentCommand.ComputeCodexConfigContent(existing);
        Assert.Equal(existing, content);
    }

    // ---- JSON tolerant parsing ---------------------------------------------------------------

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not json at all")]
    [InlineData("[1,2,3]")] // valid JSON, but not an object
    [InlineData("\"just a string\"")]
    public void ParseJsonObjectTolerant_InvalidOrNonObjectInput_ReturnsEmptyObject(string? raw)
    {
        var obj = InstallAgentCommand.ParseJsonObjectTolerant(raw);
        Assert.Empty(obj);
    }

    [Fact]
    public void ParseJsonObjectTolerant_ValidObject_ParsesIt()
    {
        var obj = InstallAgentCommand.ParseJsonObjectTolerant("""{"a": 1, "b": "two"}""");
        Assert.Equal(1, obj["a"]!.GetValue<int>());
        Assert.Equal("two", obj["b"]!.GetValue<string>());
    }

    [Fact]
    public void ParseJsonObjectTolerant_StripsLeadingBom()
    {
        var raw = "﻿{\"a\": 1}";
        var obj = InstallAgentCommand.ParseJsonObjectTolerant(raw);
        Assert.Equal(1, obj["a"]!.GetValue<int>());
    }

    // ---- Hook merge / idempotency ------------------------------------------------------------

    [Fact]
    public void BuildClaudeSettingsJson_FreshFile_AddsAllFourEvents()
    {
        var result = InstallAgentCommand.BuildClaudeSettingsJson(new JsonObject());
        var hooks = Assert.IsType<JsonObject>(result["hooks"]);

        AssertHasCommand(hooks, "SessionStart", "psm-memory hook session-start");
        AssertHasCommand(hooks, "UserPromptSubmit", "psm-memory hook recall");
        AssertHasCommand(hooks, "Stop", "psm-memory hook remember");
        AssertHasCommand(hooks, "SessionEnd", "psm-memory hook session-end");

        // SessionStart/Stop/SessionEnd are async:true per spec; UserPromptSubmit is not.
        Assert.True(GetHookEntry(hooks, "SessionStart")["async"]!.GetValue<bool>());
        Assert.True(GetHookEntry(hooks, "Stop")["async"]!.GetValue<bool>());
        Assert.True(GetHookEntry(hooks, "SessionEnd")["async"]!.GetValue<bool>());
        Assert.Null(GetHookEntry(hooks, "UserPromptSubmit")["async"]);
    }

    [Fact]
    public void BuildGeminiSettingsJson_SetsHooksConfigEnabledAndPreservesExistingKeys()
    {
        var existing = new JsonObject { ["hooksConfig"] = new JsonObject { ["someOtherFlag"] = "keep-me" } };
        var result = InstallAgentCommand.BuildGeminiSettingsJson(existing);

        var hooksConfig = Assert.IsType<JsonObject>(result["hooksConfig"]);
        Assert.True(hooksConfig["enabled"]!.GetValue<bool>());
        Assert.Equal("keep-me", hooksConfig["someOtherFlag"]!.GetValue<string>());

        var hooks = Assert.IsType<JsonObject>(result["hooks"]);
        AssertHasCommand(hooks, "BeforeAgent", "psm-memory hook recall --agent gemini");
        AssertHasCommand(hooks, "AfterAgent", "psm-memory hook remember --agent gemini");
    }

    [Fact]
    public void BuildClaudeSettingsJson_PreservesUnrelatedTopLevelKeys()
    {
        var existing = new JsonObject { ["unrelatedSetting"] = "keep-me" };
        var result = InstallAgentCommand.BuildClaudeSettingsJson(existing);
        Assert.Equal("keep-me", result["unrelatedSetting"]!.GetValue<string>());
    }

    [Fact]
    public void BuildClaudeSettingsJson_PreservesUnrelatedHookEntries()
    {
        var existing = new JsonObject
        {
            ["hooks"] = new JsonObject
            {
                ["UserPromptSubmit"] = new JsonArray
                {
                    new JsonObject
                    {
                        ["matcher"] = "*",
                        ["hooks"] = new JsonArray { new JsonObject { ["type"] = "command", ["command"] = "some-other-tool submit" } },
                    },
                },
            },
        };
        var result = InstallAgentCommand.BuildClaudeSettingsJson(existing);
        var hooks = Assert.IsType<JsonObject>(result["hooks"]);
        var userPromptEntries = Assert.IsType<JsonArray>(hooks["UserPromptSubmit"]);

        // Both the pre-existing unrelated hook AND the newly-added psm-memory one should be present.
        var allCommands = userPromptEntries
            .SelectMany(e => ((JsonObject)e!)["hooks"]!.AsArray())
            .Select(h => ((JsonObject)h!)["command"]!.GetValue<string>())
            .ToList();
        Assert.Contains("some-other-tool submit", allCommands);
        Assert.Contains("psm-memory hook recall", allCommands);
    }

    [Fact]
    public void RunTwice_AgainstSameConfigDir_DoesNotDuplicateHookEntries()
    {
        // First install.
        var first = InstallAgentCommand.BuildClaudeSettingsJson(new JsonObject());
        // Re-install against the result of the first (simulates re-running install-agent).
        var second = InstallAgentCommand.BuildClaudeSettingsJson(first);

        var hooks = Assert.IsType<JsonObject>(second["hooks"]);
        foreach (var eventName in new[] { "SessionStart", "UserPromptSubmit", "Stop", "SessionEnd" })
        {
            var entries = Assert.IsType<JsonArray>(hooks[eventName]);
            var commands = entries.SelectMany(e => ((JsonObject)e!)["hooks"]!.AsArray())
                .Select(h => ((JsonObject)h!)["command"]!.GetValue<string>())
                .ToList();
            Assert.Single(commands); // exactly one psm-memory entry, not two
        }
    }

    private static void AssertHasCommand(JsonObject hooks, string eventName, string expectedCommand)
    {
        var entries = Assert.IsType<JsonArray>(hooks[eventName]);
        var commands = entries.SelectMany(e => ((JsonObject)e!)["hooks"]!.AsArray())
            .Select(h => ((JsonObject)h!)["command"]!.GetValue<string>())
            .ToList();
        Assert.Contains(expectedCommand, commands);
    }

    private static JsonObject GetHookEntry(JsonObject hooks, string eventName)
    {
        var entries = (JsonArray)hooks[eventName]!;
        var entry = (JsonObject)entries[0]!;
        var hookList = (JsonArray)entry["hooks"]!;
        return (JsonObject)hookList[0]!;
    }

    // ---- Full dry-run invocation against a temp --config-dir --------------------------------

    [Fact]
    public void Run_DryRunAllAgents_AgainstTempConfigDir_ProducesExpectedShapesAndTouchesNoFiles()
    {
        var exitCode = InstallAgentCommand.Run(new[] { "all", "--dry-run", "--config-dir", _tempDir });
        Assert.Equal(0, exitCode);

        // --dry-run must never create the config directory or any file in it.
        Assert.False(Directory.Exists(_tempDir));
    }

    [Fact]
    public void Run_DryRunSingleAgent_DoesNotThrow()
    {
        var exitCode = InstallAgentCommand.Run(new[] { "codex", "--dry-run", "--config-dir", _tempDir });
        Assert.Equal(0, exitCode);
        Assert.False(Directory.Exists(_tempDir));
    }

    [Fact]
    public void PathResolvers_WithConfigDir_ResolveUnderConfigDirNotTheRealHomeDotfolderPath()
    {
        var realHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var realCodexConfigPath = Path.Combine(realHome, ".codex", "config.toml");
        var realClaudeSettingsPath = Path.Combine(realHome, ".claude", "settings.json");
        var realGeminiSettingsPath = Path.Combine(realHome, ".gemini", "settings.json");

        // All four resolve to a path rooted under the given --config-dir override...
        Assert.StartsWith(_tempDir, InstallAgentCommand.CodexConfigPath(_tempDir));
        Assert.StartsWith(_tempDir, InstallAgentCommand.CodexHooksPath(_tempDir));
        Assert.StartsWith(_tempDir, InstallAgentCommand.ClaudeSettingsPath(_tempDir));
        Assert.StartsWith(_tempDir, InstallAgentCommand.GeminiSettingsPath(_tempDir));

        // ...and specifically do NOT equal the real ~/.codex, ~/.claude, ~/.gemini paths (note: a
        // plain substring check against realHome is not a valid assertion here since _tempDir is
        // itself under the current user's profile/temp directory on this machine).
        Assert.NotEqual(realCodexConfigPath, InstallAgentCommand.CodexConfigPath(_tempDir));
        Assert.NotEqual(realClaudeSettingsPath, InstallAgentCommand.ClaudeSettingsPath(_tempDir));
        Assert.NotEqual(realGeminiSettingsPath, InstallAgentCommand.GeminiSettingsPath(_tempDir));
    }

    // ---- Real (non-dry-run) write, but ONLY against a temp --config-dir ---------------------

    [Fact]
    public void Run_RealWrite_AgainstTempConfigDir_WritesFilesAndIsIdempotentOnRerun()
    {
        var exit1 = InstallAgentCommand.Run(new[] { "claude,gemini", "--config-dir", _tempDir });
        Assert.Equal(0, exit1);

        var claudePath = InstallAgentCommand.ClaudeSettingsPath(_tempDir);
        var geminiPath = InstallAgentCommand.GeminiSettingsPath(_tempDir);
        Assert.True(File.Exists(claudePath));
        Assert.True(File.Exists(geminiPath));

        // Re-run: must not duplicate hook entries (idempotent reinstall).
        var exit2 = InstallAgentCommand.Run(new[] { "claude,gemini", "--config-dir", _tempDir });
        Assert.Equal(0, exit2);

        var claudeJson = InstallAgentCommand.ParseJsonObjectTolerant(File.ReadAllText(claudePath));
        var hooks = Assert.IsType<JsonObject>(claudeJson["hooks"]);
        var stopEntries = Assert.IsType<JsonArray>(hooks["Stop"]);
        var stopCommands = stopEntries.SelectMany(e => ((JsonObject)e!)["hooks"]!.AsArray())
            .Select(h => ((JsonObject)h!)["command"]!.GetValue<string>())
            .ToList();
        Assert.Single(stopCommands);
    }
}
