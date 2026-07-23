using PsmMemory.Cli;
using Xunit;

namespace PsmMemory.Cli.Tests;

/// <summary>Covers the pure/best-effort pieces of HookCommands that don't require a model load:
/// session-summary composition and package.json name sniffing. Git-dependent behavior is exercised
/// indirectly (BuildSessionSummary must never throw, with or without a real repo underneath).</summary>
public class HookCommandsTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"psm-hookcommands-test-{Guid.NewGuid():N}");
    private readonly string _originalCwd = Directory.GetCurrentDirectory();

    public HookCommandsTests()
    {
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        Directory.SetCurrentDirectory(_originalCwd);
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    [Fact]
    public void PackageNameFor_NoPackageJson_ReturnsNull()
    {
        Assert.Null(HookCommands.PackageNameFor(_tempDir));
    }

    [Fact]
    public void PackageNameFor_ValidPackageJson_ReturnsName()
    {
        File.WriteAllText(Path.Combine(_tempDir, "package.json"), """{"name": "my-project", "version": "1.0.0"}""");
        Assert.Equal("my-project", HookCommands.PackageNameFor(_tempDir));
    }

    [Fact]
    public void PackageNameFor_InvalidJson_ReturnsNullNotThrows()
    {
        File.WriteAllText(Path.Combine(_tempDir, "package.json"), "not valid json {{{");
        Assert.Null(HookCommands.PackageNameFor(_tempDir));
    }

    [Fact]
    public void PackageNameFor_MissingNameField_ReturnsNull()
    {
        File.WriteAllText(Path.Combine(_tempDir, "package.json"), """{"version": "1.0.0"}""");
        Assert.Null(HookCommands.PackageNameFor(_tempDir));
    }

    [Fact]
    public void BuildSessionSummary_SessionStart_IncludesHeaderAgentAndTranscript()
    {
        Directory.SetCurrentDirectory(_tempDir);
        var summary = HookCommands.BuildSessionSummary("session-start", "claude", "/path/to/transcript.jsonl");

        Assert.Contains("Developer session started.", summary);
        Assert.Contains("Agent: claude.", summary);
        Assert.Contains("Transcript: /path/to/transcript.jsonl.", summary);
    }

    [Fact]
    public void BuildSessionSummary_SessionEnd_UsesEndHeader()
    {
        Directory.SetCurrentDirectory(_tempDir);
        var summary = HookCommands.BuildSessionSummary("session-end", null, null);
        Assert.Contains("Developer session ended.", summary);
    }

    [Fact]
    public void BuildSessionSummary_IncludesPackageNameWhenPresent()
    {
        Directory.SetCurrentDirectory(_tempDir);
        File.WriteAllText(Path.Combine(_tempDir, "package.json"), """{"name": "psm-memory-test-fixture"}""");
        var summary = HookCommands.BuildSessionSummary("session-start", null, null);
        Assert.Contains("Project: psm-memory-test-fixture.", summary);
    }

    [Fact]
    public void BuildSessionSummary_NeverThrows_EvenOutsideAnyGitRepo()
    {
        Directory.SetCurrentDirectory(_tempDir);
        // Should not throw even though _tempDir is (almost certainly) not a git repo -- git absence
        // or failure must be swallowed, matching TS's gitOutput() try/catch-and-return-undefined.
        var summary = HookCommands.BuildSessionSummary("session-start", "codex", null);
        Assert.False(string.IsNullOrWhiteSpace(summary));
        Assert.Contains("Developer session started.", summary);
    }
}
