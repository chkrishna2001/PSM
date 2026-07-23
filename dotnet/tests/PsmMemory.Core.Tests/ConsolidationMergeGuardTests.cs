using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Regression test for the consolidation false-merge bug found via a real probe
/// (2026-07-20, coding-agent-cx-store-10, see project memory
/// project_psm_storage_detail_probe_findings.md): the consolidation adapter matched a new, distinct
/// finding against an unrelated existing memory that only shared boilerplate experiment-log
/// phrasing, decided "update_existing", and its merged_content discarded the new finding's actual
/// substance entirely -- the new information was silently lost. <see cref="FakePsmRuntime"/>
/// reproduces exactly that scripted verdict (a bad-faith/wrong consolidation decision) so the test
/// doesn't depend on a real model agreeing to reproduce the bug; it proves PsmService's guard
/// rejects that merge and falls back to storing the new content independently either way.
/// </summary>
public class ConsolidationMergeGuardTests
{
    // The exact two contents from the real probe case that triggered the bug.
    private const string ExistingUnrelatedContent =
        "Experiment 25 did break the structural analogy: source/position reuse is `1/15`, single-channel local reconstruction is `1/15` and `3/17`, while full interaction comparison is the only improved condition at `6/15`. I'm documenting that as a partial failure, not a new mechanism.";

    private const string NewDistinctContent =
        "The cleaned temporal benchmark has the shape we wanted: each component is `7/7` alone on its own task, but each fails when the task is composed, while the paired temporal relation gets `5/9`. I'm documenting that as non-decomposability under composition, not as a new mechanism.";

    private sealed class FakePsmRuntime : IPsmRuntime
    {
        public required string StorageDecisionJson { get; init; }
        public required string ConsolidationDecisionJson { get; init; }

        public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult(StorageDecisionJson);

        public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("not needed for this test");

        public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult(ConsolidationDecisionJson);
    }

    [Fact]
    public async Task RememberAsync_RejectsConsolidationMerge_ThatDiscardsNewContent()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-consolidation-guard-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        // Seed the unrelated existing memory that the (buggy) consolidation match will fire against.
        store.InsertEpisodic("guard-test-user", ExistingUnrelatedContent);

        var storageJson = $$"""
            {"reasoning":"A concrete verified finding worth remembering.","action":"store_episodic","memory":{"content":"{{NewDistinctContent}}"},"facts":[],"indexables":[]}
            """;
        // Reproduces the real bad verdict: update_existing, merged toward the OLD memory's content,
        // discarding the new finding entirely.
        var consolidationJson = $$"""
            {"action":"update_existing","merged_content":"{{ExistingUnrelatedContent}}","reasoning":"Treating as an update of the existing experiment log entry."}
            """;

        var runtime = new FakePsmRuntime { StorageDecisionJson = storageJson, ConsolidationDecisionJson = consolidationJson };
        var service = new PsmService(store, runtime);

        var result = await service.RememberAsync(new RememberRequest
        {
            UserId = "guard-test-user",
            LlmResponse = NewDistinctContent,
        });

        // The fix: the untrustworthy merge is rejected, so the new content is stored on its own
        // terms (not silently discarded, not overwritten with the unrelated old memory's content).
        Assert.NotEqual(Actions.Kinds.Ignore, result.Action);
        Assert.NotEqual("dedupe_skip", result.Route);
        Assert.NotNull(result.Memory);
        Assert.Contains("7/7", result.Memory!.Content);
        Assert.Contains("5/9", result.Memory.Content);
        Assert.DoesNotContain("1/15", result.Memory.Content);

        var stored = store.SelectMemories("guard-test-user", MemoryTables.All, 100);
        Assert.Contains(stored, m => m.Content.Contains("7/7") && m.Content.Contains("5/9"));
    }

    [Fact]
    public async Task RememberAsync_AcceptsConsolidationMerge_ThatGenuinelyRetainsNewContent()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-consolidation-guard-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        const string existingContent = "The user's favorite programming language is Python.";
        const string newContent = "The user's favorite programming language is actually Rust now, not Python.";
        const string goodMergedContent = "The user's favorite programming language changed from Python to Rust.";

        store.InsertEpisodic("guard-test-user-2", existingContent);

        var storageJson = $$"""
            {"reasoning":"An updated preference worth remembering.","action":"store_episodic","memory":{"content":"{{newContent}}"},"facts":[],"indexables":[]}
            """;
        var consolidationJson = $$"""
            {"action":"update_existing","merged_content":"{{goodMergedContent}}","reasoning":"Genuinely the same fact, updated."}
            """;

        var runtime = new FakePsmRuntime { StorageDecisionJson = storageJson, ConsolidationDecisionJson = consolidationJson };
        var service = new PsmService(store, runtime);

        var result = await service.RememberAsync(new RememberRequest
        {
            UserId = "guard-test-user-2",
            LlmResponse = newContent,
        });

        // A merge that genuinely retains the new content's own substance should still be accepted.
        Assert.Equal(Actions.Kinds.UpdateExisting, result.Action);
        Assert.NotNull(result.Memory);
        Assert.Equal(goodMergedContent, result.Memory!.Content);
    }
}
