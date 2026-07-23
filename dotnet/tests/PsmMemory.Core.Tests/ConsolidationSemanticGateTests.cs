using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Regression test for the consolidation over-merge bug found via a real probe (2026-07-21,
/// locomo-conv26-charity-race, see project memory project_psm_storage_detail_probe_findings.md):
/// two completely unrelated facts about the same person ("Melanie got a cat" / "Melanie ran a
/// charity race") scored 0.60 on Ranking.HybridRankMemories -- past the 0.3 candidate threshold --
/// almost entirely from the shared-entity boost, not real content overlap (lexical score was only
/// 0.09). Uses the REAL <see cref="LlamaSharpEmbeddingRuntime"/> (small, fast to load) so the test
/// exercises the actual cosine-similarity gate, combined with a tracking fake <see cref="IPsmRuntime"/>
/// so it doesn't depend on the full LLM's generation quality/speed to prove the point: did the
/// consolidation adapter get called at all.
/// </summary>
public class ConsolidationSemanticGateTests
{
    private sealed class TrackingFakeRuntime : IPsmRuntime
    {
        public bool ConsolidationCalled { get; private set; }
        public required string StorageDecisionJson { get; init; }
        public string ConsolidationDecisionJson { get; init; } = """{"action":"store_episodic","reasoning":"not actually reached in the blocked case"}""";

        public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult(StorageDecisionJson);

        public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("not needed for this test");

        public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
        {
            ConsolidationCalled = true;
            return Task.FromResult(ConsolidationDecisionJson);
        }
    }

    [Fact(Skip = "manual — requires the real embedding GGUF model on disk; run manually to validate the semantic consolidation gate")]
    public async Task ConsolidateAsync_SkipsConsolidationAdapter_ForLexicallySimilarButSemanticallyUnrelatedContent()
    {
        var modelDir = Path.Combine(LlamaSharpSmokeTests.FindRepoRoot(), "psm-model", "prod-memory", "gguf-runtime", "v1");
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-semantic-gate-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var embeddingRuntime = new LlamaSharpEmbeddingRuntime(modelDir);

        const string existingContent = "Melanie got a cat named Bailey as a gift.";
        const string newContent = "Melanie ran a charity race for mental health last Saturday, which she described as really rewarding and made her think about taking care of our minds.";

        store.InsertEpisodic("semantic-gate-user", existingContent);

        var storageJson = $$"""
            {"reasoning":"A concrete event worth remembering.","action":"store_episodic","memory":{"content":"{{newContent}}"},"facts":[],"indexables":[]}
            """;
        var runtime = new TrackingFakeRuntime { StorageDecisionJson = storageJson };
        var service = new PsmService(store, runtime, embeddingRuntime);

        var result = await service.RememberAsync(new RememberRequest
        {
            UserId = "semantic-gate-user",
            LlmResponse = newContent,
        });

        // The gate should reject this as a consolidation candidate before ever calling the
        // consolidation adapter -- these are two unrelated facts, not the same fact restated.
        Assert.False(runtime.ConsolidationCalled);
        Assert.Equal(Actions.Kinds.StoreEpisodic, result.Action);
        Assert.NotNull(result.Memory);
        Assert.Equal(newContent, result.Memory!.Content);
    }

    [Fact(Skip = "manual — requires the real embedding GGUF model on disk; run manually to validate the semantic consolidation gate")]
    public async Task ConsolidateAsync_StillCallsConsolidationAdapter_ForGenuinelySimilarContent()
    {
        var modelDir = Path.Combine(LlamaSharpSmokeTests.FindRepoRoot(), "psm-model", "prod-memory", "gguf-runtime", "v1");
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-semantic-gate-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var embeddingRuntime = new LlamaSharpEmbeddingRuntime(modelDir);

        const string existingContent = "The user's favorite programming language is Python.";
        const string newContent = "The user's favorite programming language is actually Rust now, not Python.";

        store.InsertEpisodic("semantic-gate-user-2", existingContent);

        var storageJson = $$"""
            {"reasoning":"An updated preference worth remembering.","action":"store_episodic","memory":{"content":"{{newContent}}"},"facts":[],"indexables":[]}
            """;
        var runtime = new TrackingFakeRuntime
        {
            StorageDecisionJson = storageJson,
            ConsolidationDecisionJson = """{"action":"store_episodic","reasoning":"Different enough to keep independent for this test."}""",
        };
        var service = new PsmService(store, runtime, embeddingRuntime);

        await service.RememberAsync(new RememberRequest
        {
            UserId = "semantic-gate-user-2",
            LlmResponse = newContent,
        });

        // A genuinely related update should still reach the consolidation adapter -- the gate
        // must not be so strict it blocks real consolidation candidates.
        Assert.True(runtime.ConsolidationCalled);
    }
}
