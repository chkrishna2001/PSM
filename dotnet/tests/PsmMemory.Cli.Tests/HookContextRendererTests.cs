using PsmMemory.Cli;
using PsmMemory.Core;
using PsmMemory.Core.Models;
using Xunit;

namespace PsmMemory.Cli.Tests;

public class HookContextRendererTests
{
    private static RecallResult EmptyResult(List<RankedMemory>? memories = null, List<Ranking.ScoredFact>? facts = null) => new()
    {
        UserId = "u1",
        Query = "q",
        Plan = new RecallPlan { Intent = "context", TargetTables = new List<string> { "episodic" }, TopK = 5, RawJson = "{}" },
        Memories = memories ?? new List<RankedMemory>(),
        Facts = facts ?? new List<Ranking.ScoredFact>(),
    };

    private static RankedMemory Memory(string content) => new()
    {
        Memory = new MemoryRecord { Id = Guid.NewGuid().ToString(), UserId = "u1", Content = content, Table = MemoryTables.Episodic },
        Score = 1.0,
        Metadata = new Dictionary<string, object?>(),
    };

    private static Ranking.ScoredFact Fact(string subject, string predicate, string valueText) => new()
    {
        Fact = new MemoryFactRecord { Id = Guid.NewGuid().ToString(), UserId = "u1", Subject = subject, Predicate = predicate, ValueText = valueText },
        Score = 1.0,
    };

    [Fact]
    public void Render_NoMemoriesOrFacts_ReturnsEmptyString()
    {
        Assert.Equal("", HookContextRenderer.Render(EmptyResult()));
    }

    [Fact]
    public void Render_MemoriesOnly_ProducesOneLinePerMemory()
    {
        var result = EmptyResult(memories: new List<RankedMemory> { Memory("Likes Neovim"), Memory("Prefers dark mode") });
        var rendered = HookContextRenderer.Render(result);
        Assert.Contains("- Likes Neovim", rendered);
        Assert.Contains("- Prefers dark mode", rendered);
    }

    [Fact]
    public void Render_FactsBeforeMemories()
    {
        var result = EmptyResult(
            memories: new List<RankedMemory> { Memory("some memory line") },
            facts: new List<Ranking.ScoredFact> { Fact("user", "prefers", "dark mode") });
        var rendered = HookContextRenderer.Render(result);

        var factIndex = rendered.IndexOf("user prefers dark mode", StringComparison.Ordinal);
        var memoryIndex = rendered.IndexOf("some memory line", StringComparison.Ordinal);
        Assert.True(factIndex >= 0);
        Assert.True(memoryIndex >= 0);
        Assert.True(factIndex < memoryIndex, "facts must render before memories");
    }

    [Fact]
    public void Render_CapsMemoryAndFactCount()
    {
        var memories = Enumerable.Range(0, 20).Select(i => Memory($"memory-{i}")).ToList();
        var facts = Enumerable.Range(0, 20).Select(i => Fact("s", "p", $"fact-{i}")).ToList();
        var result = EmptyResult(memories: memories, facts: facts);
        var rendered = HookContextRenderer.Render(result);

        var memoryLineCount = rendered.Split('\n').Count(l => l.Contains("memory-"));
        var factLineCount = rendered.Split('\n').Count(l => l.Contains("fact-"));
        Assert.True(memoryLineCount <= 8);
        Assert.True(factLineCount <= 8);
    }
}
