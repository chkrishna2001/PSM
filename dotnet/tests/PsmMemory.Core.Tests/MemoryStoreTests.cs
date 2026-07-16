using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

public sealed class MemoryStoreFixture : IDisposable
{
    public MemoryStore CreateStore()
    {
        var path = Path.Combine(Path.GetTempPath(), $"psm-memory-tests-{Guid.NewGuid():N}.db");
        _paths.Add(path);
        var store = new MemoryStore(path);
        store.InitializeSchema();
        return store;
    }

    private readonly List<string> _paths = new();

    public void Dispose()
    {
        foreach (var path in _paths)
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { /* best-effort cleanup */ }
        }
    }
}

public class MemoryStoreTests : IClassFixture<MemoryStoreFixture>
{
    private readonly MemoryStoreFixture _fixture;

    public MemoryStoreTests(MemoryStoreFixture fixture) => _fixture = fixture;

    private static StorageDecision Decision(string action, string? content, string reasoning = "test reasoning") => new()
    {
        Action = action,
        Memory = content is null ? null : new MemoryPayload { Content = content },
        Reasoning = reasoning,
        RawJson = "{}"
    };

    [Fact]
    public void ApplyDecision_IgnoreWritesNothing()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-1", "src-1", Decision(Actions.Kinds.Ignore, null));

        Assert.Equal(Actions.Routes.Ignore, result.Route);
        Assert.Empty(result.Written);
        Assert.Empty(store.SelectMemories("user-1", new[] { MemoryTables.Episodic }));
    }

    [Fact]
    public void ApplyDecision_StoreEpisodicWritesEpisodicRow()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-1", "src-1", Decision(Actions.Kinds.StoreEpisodic, "Sam adopted a puppy."));

        Assert.Equal(Actions.Routes.EpisodicInsert, result.Route);
        Assert.Contains(MemoryTables.Episodic, result.Written);
        var rows = store.SelectMemories("user-1", new[] { MemoryTables.Episodic });
        var row = Assert.Single(rows);
        Assert.Equal("Sam adopted a puppy.", row.Content);
        Assert.Equal(MemoryTables.Episodic, row.Table);
    }

    [Fact]
    public void ApplyDecision_PromoteSemanticWritesSemanticRow()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-2", "src-1", Decision(Actions.Kinds.PromoteSemantic, "Sam always brings his dog hiking."));

        Assert.Equal(Actions.Routes.SemanticUpsert, result.Route);
        Assert.Contains(MemoryTables.Semantic, result.Written);
        var rows = store.SelectMemories("user-2", new[] { MemoryTables.Semantic });
        Assert.Single(rows);
    }

    [Fact]
    public void ApplyDecision_UpdateExistingRoutesToSemanticUpsertLikePromote()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-3", "src-1", Decision(Actions.Kinds.UpdateExisting, "Updated fact about Sam."));

        Assert.Equal(Actions.Routes.UpdateWithSupersede, result.Route);
        Assert.Contains(MemoryTables.Semantic, result.Written);
    }

    [Fact]
    public void ApplyDecision_DecayRoutesToDecayScheduleThenEpisodic()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-4", "src-1", Decision(Actions.Kinds.Decay, "Decaying fact."));

        Assert.Equal(Actions.Routes.DecayExistingThenInsert, result.Route);
        Assert.Contains("decay_schedule", result.Written);
        Assert.Contains(MemoryTables.Episodic, result.Written);
    }

    [Fact]
    public void ApplyDecision_FlagConflictWritesConflictOnlyByDefault()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-5", "src-1", Decision(Actions.Kinds.FlagConflict, "Contradicts an existing memory."));

        Assert.Equal(Actions.Routes.ConflictLogAndHold, result.Route);
        Assert.Contains("conflicts", result.Written);
        Assert.DoesNotContain(MemoryTables.Episodic, result.Written);
    }

    [Fact]
    public void ApplyDecision_FlagAndStoreWritesConflictAndEpisodic()
    {
        using var store = _fixture.CreateStore();

        var result = store.ApplyDecision("user-6", "src-1", Decision(Actions.Kinds.FlagAndStore, "Contradicts but also worth storing."));

        Assert.Equal(Actions.Routes.ConflictLogAndHold, result.Route);
        Assert.Contains("conflicts", result.Written);
        Assert.Contains(MemoryTables.Episodic, result.Written);
    }

    [Fact]
    public void ApplyDecision_ConflictAgainstRecordsExistingMemoryReference()
    {
        using var store = _fixture.CreateStore();

        store.ApplyDecision(
            "user-7",
            "src-1",
            Decision(Actions.Kinds.FlagConflict, "Conflicting content."),
            conflictAgainst: ("existing-mem-id", MemoryTables.Semantic));

        var conflicts = store.SelectConflicts("unresolved", 10);
        var row = Assert.Single(conflicts);
        Assert.Equal("existing-mem-id", row["existing_memory_id"]);
        Assert.Equal(MemoryTables.Semantic, row["existing_memory_type"]);
    }

    [Fact]
    public void ApplyDecision_DuplicateContentFromSameSourceIsSkipped()
    {
        using var store = _fixture.CreateStore();

        var first = store.ApplyDecision("user-8", "src-dup", Decision(Actions.Kinds.StoreEpisodic, "The same durable fact."));
        var second = store.ApplyDecision("user-8", "src-dup", Decision(Actions.Kinds.StoreEpisodic, "The same durable fact."));

        Assert.Contains(MemoryTables.Episodic, first.Written);
        Assert.Equal("dedupe_skip", second.Route);
        Assert.Empty(second.Written);
        Assert.Single(store.SelectMemories("user-8", new[] { MemoryTables.Episodic }));
    }

    [Fact]
    public void InsertIgnoredDecision_IsQueryableAfterInsert()
    {
        using var store = _fixture.CreateStore();

        var id = store.InsertIgnoredDecision("user-9", "src-1", "some content", "low confidence ignore", 0.2, "{}");

        Assert.False(string.IsNullOrEmpty(id));
    }
}
