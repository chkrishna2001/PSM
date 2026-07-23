using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Ported from psm-core/src/temporal.ts (normalizeMemoryTemporalFields, normalizeFactTemporalFields,
/// resolveRelativeTime, detectRelativeExpression). Anchor date used throughout: 15 June 2026, matching
/// a source_timestamp of "2026-06-15T00:00:00Z".
/// </summary>
public class TemporalNormalizerTests
{
    private const string Anchor = "2026-06-15T00:00:00Z";

    [Theory]
    [InlineData("today", "15 June 2026")]
    [InlineData("yesterday", "14 June 2026")]
    [InlineData("tomorrow", "16 June 2026")]
    [InlineData("last week", "week before 15 June 2026")]
    [InlineData("next week", "week after 15 June 2026")]
    [InlineData("last month", "May 2026")]
    [InlineData("next month", "July 2026")]
    [InlineData("last year", "2025")]
    [InlineData("next year", "2027")]
    public void ResolveRelativeTime_ResolvesEachRelativePhrase(string expression, string expected)
    {
        var resolved = TemporalNormalizer.ResolveRelativeTime(expression, Anchor);
        Assert.Equal(expected, resolved);
    }

    [Fact]
    public void ResolveRelativeTime_LastMonth_HandlesYearRollover()
    {
        var resolved = TemporalNormalizer.ResolveRelativeTime("last month", "2026-01-15T00:00:00Z");
        Assert.Equal("December 2025", resolved);
    }

    [Fact]
    public void ResolveRelativeTime_NextMonth_HandlesYearRollover()
    {
        var resolved = TemporalNormalizer.ResolveRelativeTime("next month", "2026-12-15T00:00:00Z");
        Assert.Equal("January 2027", resolved);
    }

    [Fact]
    public void ResolveRelativeTime_UnparseableSourceTimestamp_ReturnsNull()
    {
        var resolved = TemporalNormalizer.ResolveRelativeTime("today", "not-a-date");
        Assert.Null(resolved);
    }

    [Fact]
    public void DetectRelativeExpression_FindsPhraseCaseInsensitively()
    {
        Assert.Equal("yesterday", TemporalNormalizer.DetectRelativeExpression("I saw them Yesterday afternoon."));
        Assert.Null(TemporalNormalizer.DetectRelativeExpression("Nothing relative here."));
    }

    [Fact]
    public void NormalizeMemoryTemporalFields_DetectsAndResolves_FromContent()
    {
        var memory = new MemoryPayload { Content = "We shipped the fix yesterday." };

        TemporalNormalizer.NormalizeMemoryTemporalFields(memory, Anchor);

        Assert.Equal("yesterday", memory.TemporalExpression);
        Assert.Equal("14 June 2026", memory.ResolvedTime);
        Assert.Equal(0.9, memory.ResolvedTimeConfidence);
    }

    [Fact]
    public void NormalizeMemoryTemporalFields_NoExpressionDetected_LeavesFieldsUntouched()
    {
        var memory = new MemoryPayload
        {
            Content = "Nothing temporal about this sentence.",
            TemporalExpression = null,
            ResolvedTime = null,
            ResolvedTimeConfidence = null
        };

        TemporalNormalizer.NormalizeMemoryTemporalFields(memory, Anchor);

        Assert.Null(memory.TemporalExpression);
        Assert.Null(memory.ResolvedTime);
        Assert.Null(memory.ResolvedTimeConfidence);
    }

    [Fact]
    public void NormalizeMemoryTemporalFields_UnsupportedPreSetExpression_ClearsAllThreeFields()
    {
        // A caller pre-set temporal_expression to something that doesn't match the supported
        // relative-phrase / bare-year / day-month checks -- the rare "clear fields" branch.
        var memory = new MemoryPayload
        {
            Content = "Some content with no relative phrase.",
            TemporalExpression = "sometime soon",
            ResolvedTime = "stale value",
            ResolvedTimeConfidence = 0.7
        };

        TemporalNormalizer.NormalizeMemoryTemporalFields(memory, Anchor);

        Assert.Null(memory.TemporalExpression);
        Assert.Null(memory.ResolvedTime);
        Assert.Null(memory.ResolvedTimeConfidence);
    }

    [Fact]
    public void NormalizeMemoryTemporalFields_NoSourceTimestamp_LeavesFieldsUntouched()
    {
        var memory = new MemoryPayload { Content = "We shipped the fix yesterday." };

        TemporalNormalizer.NormalizeMemoryTemporalFields(memory, sourceTimestamp: null);

        Assert.Null(memory.TemporalExpression);
        Assert.Null(memory.ResolvedTime);
        Assert.Null(memory.ResolvedTimeConfidence);
    }

    [Fact]
    public void NormalizeMemoryTemporalFields_PreservesExistingExpression_ButFloorsConfidenceAt09()
    {
        var memory = new MemoryPayload
        {
            Content = "We shipped the fix yesterday.",
            TemporalExpression = "Yesterday", // caller pre-set, different casing -- must be preserved as-is
            ResolvedTimeConfidence = 0.95 // already above the 0.9 floor -- must not be lowered
        };

        TemporalNormalizer.NormalizeMemoryTemporalFields(memory, Anchor);

        Assert.Equal("Yesterday", memory.TemporalExpression);
        Assert.Equal("14 June 2026", memory.ResolvedTime);
        Assert.Equal(0.95, memory.ResolvedTimeConfidence);
    }

    [Fact]
    public void NormalizeFactTemporalFields_DetectsFromEvidenceAndValueText()
    {
        var fact = new MemoryFactPayload
        {
            Subject = "user",
            Predicate = "shipped",
            EvidenceText = "Shipped last month per the changelog.",
            ValueText = "changelog entry"
        };

        TemporalNormalizer.NormalizeFactTemporalFields(fact, Anchor);

        Assert.Equal("last month", fact.TemporalExpression);
        Assert.Equal("May 2026", fact.ResolvedTime);
        Assert.Equal(0.9, fact.ResolvedTimeConfidence);
    }

    [Fact]
    public void NormalizeFactTemporalFields_NoExpression_LeavesFieldsUntouched()
    {
        var fact = new MemoryFactPayload
        {
            Subject = "user",
            Predicate = "likes",
            EvidenceText = "Nothing temporal here.",
            ValueText = "coffee"
        };

        TemporalNormalizer.NormalizeFactTemporalFields(fact, Anchor);

        Assert.Null(fact.TemporalExpression);
        Assert.Null(fact.ResolvedTime);
        Assert.Null(fact.ResolvedTimeConfidence);
    }

    private sealed class FakePsmRuntime : IPsmRuntime
    {
        public required string StorageDecisionJson { get; init; }

        public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult(StorageDecisionJson);

        public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("not needed for this test");

        public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("not needed for this test");
    }

    [Fact]
    public async Task RememberAsync_PopulatesResolvedTemporalFields_OnPersistedMemory()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-temporal-e2e-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        const string content = "We deployed the hotfix yesterday and it resolved the outage.";
        var storageJson = $$"""
            {"reasoning":"A concrete event worth remembering.","action":"store_episodic","memory":{"content":"{{content}}"},"facts":[],"indexables":[]}
            """;

        var runtime = new FakePsmRuntime { StorageDecisionJson = storageJson };
        var service = new PsmService(store, runtime);

        var result = await service.RememberAsync(new RememberRequest
        {
            UserId = "temporal-e2e-user",
            LlmResponse = content,
            Source = new MemorySourceMetadata { SourceTimestamp = Anchor },
            IncludeExistingMemories = false
        });

        Assert.NotNull(result.Memory);
        Assert.Equal("yesterday", result.Memory!.TemporalExpression);
        Assert.Equal("14 June 2026", result.Memory.ResolvedTime);
        Assert.Equal(0.9, result.Memory.ResolvedTimeConfidence);

        var stored = store.SelectMemories("temporal-e2e-user", MemoryTables.All, 100);
        Assert.Contains(stored, m =>
            m.Content.Contains("deployed the hotfix")
            && m.ResolvedTime == "14 June 2026"
            && m.ResolvedTimeConfidence == 0.9);
    }
}
