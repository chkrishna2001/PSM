using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Integration proof for the embedding-based vector recall fix (the porting regression documented
/// on <see cref="PsmService"/>'s class remarks): stores one memory whose content shares essentially
/// no vocabulary with a semantically-related query, then asserts recall only surfaces it when an
/// <see cref="IEmbeddingRuntime"/> is wired in -- and that this holds for BOTH
/// <see cref="PsmDomain.Coding"/> and <see cref="PsmDomain.Conversational"/>, since the recall-plan
/// adapter differs per domain but the embedding/ranking layer underneath it does not.
/// Skipped by default (loads two real GGUF models); run manually to validate end-to-end.
/// </summary>
public class EmbeddingRecallTests
{
    // Deliberately shares zero tokens with Query below after tokenization/stopword removal (no
    // shared entities either -- "Priya"/"Biscuit" never recur in the query) so a pure-lexical
    // ranker scores this at (or near) zero; only semantic similarity connects the two.
    private const string MemoryContent =
        "Priya adopted a golden retriever puppy last weekend and named him Biscuit.";

    private const string Query =
        "What kind of canine did the person recently welcome into their home?";

    private static readonly PsmDomain[] Domains = { PsmDomain.Coding, PsmDomain.Conversational };

    [Theory(Skip = "manual — requires the real GGUF LLM + embedding models on disk; run manually to validate the vector-recall fix for both domains")]
    [MemberData(nameof(Cases))]
    public async Task RecallAsync_SurfacesSemanticMatch_OnlyWithEmbeddingRuntime_ForDomain(PsmDomain domain, string query)
    {
        var modelDir = Path.Combine(LlamaSharpSmokeTests.FindRepoRoot(), "psm-model", "prod-memory", "gguf-runtime", "v1");
        Assert.True(Directory.Exists(modelDir), $"expected model directory at {modelDir}");

        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-embed-recall-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        using var runtime = new LlamaSharpPsmRuntime(modelDir);
        using var embeddingRuntime = new LlamaSharpEmbeddingRuntime(modelDir);

        var userId = $"embed-test-{domain}";
        var memoryId = store.InsertEpisodic(userId, MemoryContent);
        var reference = new WrittenMemoryRef(MemoryTables.Episodic, memoryId, MemoryContent);
        var embedding = await embeddingRuntime.EmbedAsync(MemoryContent);
        store.UpsertMemoryEmbedding(reference, userId, LlamaSharpEmbeddingRuntime.ModelName, embedding);

        // Pure-lexical baseline (no embedding runtime, the pre-fix behavior): the semantically
        // related but lexically disjoint query should NOT surface this memory.
        var lexicalOnlyService = new PsmService(store, runtime);
        var lexicalResult = await lexicalOnlyService.RecallAsync(new RecallRequest
        {
            Question = query,
            UserId = userId,
            Domain = domain,
            TopK = 5,
        });
        Assert.DoesNotContain(lexicalResult.Memories, m => m.Id == memoryId);

        // With the embedding runtime wired in (the fix): the same query should now surface it via
        // cosine similarity, regardless of which domain's recall-plan adapter ran.
        var vectorAwareService = new PsmService(store, runtime, embeddingRuntime);
        var vectorResult = await vectorAwareService.RecallAsync(new RecallRequest
        {
            Question = query,
            UserId = userId,
            Domain = domain,
            TopK = 5,
        });
        Assert.Contains(vectorResult.Memories, m => m.Id == memoryId);

        // No explicit File.Delete here: Microsoft.Data.Sqlite pools connections, so the underlying
        // file handle can outlive this method's `using var store` disposal -- left for the OS to
        // reclaim from the temp directory, same as other transient test artifacts.
    }

    public static IEnumerable<object[]> Cases() =>
        Domains.Select(domain => new object[] { domain, Query });
}
