using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Verifies the fire-and-forget remember() queue end-to-end: enqueue via
/// <see cref="MemoryStore.InsertPendingRememberRequest"/>, drain via
/// <see cref="RememberQueueDrainer.DrainOnceAsync"/>, confirm chunking (via
/// <see cref="TextSegmenter"/>) and per-chunk source-id tagging behave as designed, and that one bad
/// row never blocks the rest of a batch.
/// </summary>
public class RememberQueueDrainerTests
{
    private const string AlphaMarker = "ALPHA-TOPIC-MARKER";
    private const string BetaMarker = "BETA-TOPIC-MARKER";

    /// <summary>Scripted runtime: inspects the built prompt for a marker and echoes a distinguishable
    /// memory.content back, so each chunk's stored content can be told apart in assertions without
    /// depending on a real model.</summary>
    private sealed class FakePsmRuntime : IPsmRuntime
    {
        public Func<string, string>? OnStorageDecision { get; init; }

        public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult(OnStorageDecision!(prompt));

        public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("not needed for this test");

        public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("no existing memories to consolidate against in this test -- should never be called");
    }

    private static string StoreEpisodicJson(string content) =>
        $$"""
        {"reasoning":"A concrete detail worth remembering.","action":"store_episodic","memory":{"content":"{{content}}"},"facts":[],"indexables":[]}
        """;

    private static string BuildLongTwoSectionResponse()
    {
        // Each section needs to be well over TextSegmenter's ~200-token min chunk size and the whole
        // response needs to be well over its 1200-token max before it will split at all (estimate:
        // ~4 chars/token). A repeated filler sentence keeps this readable while reaching that length.
        var filler = string.Concat(Enumerable.Repeat("This is a detailed filler sentence about the topic. ", 60));
        return $"# Topic Alpha\n{AlphaMarker} {filler}\n\n# Topic Beta\n{BetaMarker} {filler}";
    }

    [Fact]
    public async Task DrainOnceAsync_ChunksLongResponse_WritesDistinctMemoriesWithChunkedSourceIds()
    {
        var longResponse = BuildLongTwoSectionResponse();

        // Confirm the fixture text actually forces a split -- otherwise this test would silently
        // degrade into testing the single-segment path instead of chunking.
        var segments = TextSegmenter.SegmentLlmResponse(longResponse);
        Assert.True(segments.Count > 1, $"expected the fixture response to split into multiple segments, got {segments.Count}");

        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-queue-drainer-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        var runtime = new FakePsmRuntime
        {
            OnStorageDecision = prompt => prompt.Contains(AlphaMarker)
                ? StoreEpisodicJson($"Remembered: {AlphaMarker}")
                : prompt.Contains(BetaMarker)
                    ? StoreEpisodicJson($"Remembered: {BetaMarker}")
                    : throw new InvalidOperationException("prompt matched neither expected marker"),
        };
        var service = new PsmService(store, runtime);
        var drainer = new RememberQueueDrainer(store, service);

        var enqueuedId = store.InsertPendingRememberRequest(
            userId: "queue-test-user",
            llmResponse: longResponse,
            userMessage: null,
            includeExistingMemories: true,
            extraTags: null,
            sourceKind: null,
            sourceId: "conv-42",
            sourceTimestamp: null,
            sourceLabel: null,
            domain: PsmDomain.Coding.ToString());

        var processed = await drainer.DrainOnceAsync();

        Assert.Equal(1, processed);

        var stored = store.SelectMemories("queue-test-user", MemoryTables.All, 100);
        Assert.Contains(stored, m => m.Content.Contains(AlphaMarker) && m.SourceId == TextSegmenter.ChunkSourceId("conv-42", 0));
        Assert.Contains(stored, m => m.Content.Contains(BetaMarker) && m.SourceId == TextSegmenter.ChunkSourceId("conv-42", 1));

        // Row should now be marked processed -- no longer selectable as pending.
        var stillPending = store.SelectPendingRememberRequests();
        Assert.DoesNotContain(stillPending, r => r.Id == enqueuedId);
    }

    [Fact]
    public async Task DrainOnceAsync_SingleSegmentResponse_PassesSourceIdThroughUnchanged()
    {
        const string shortResponse = "The user's favorite editor is Neovim.";
        Assert.Single(TextSegmenter.SegmentLlmResponse(shortResponse));

        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-queue-drainer-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        var runtime = new FakePsmRuntime { OnStorageDecision = _ => StoreEpisodicJson(shortResponse) };
        var service = new PsmService(store, runtime);
        var drainer = new RememberQueueDrainer(store, service);

        store.InsertPendingRememberRequest(
            userId: "queue-test-user-2",
            llmResponse: shortResponse,
            userMessage: null,
            includeExistingMemories: true,
            extraTags: null,
            sourceKind: "chat",
            sourceId: "conv-7",
            sourceTimestamp: null,
            sourceLabel: null,
            domain: PsmDomain.Coding.ToString());

        await drainer.DrainOnceAsync();

        var stored = store.SelectMemories("queue-test-user-2", MemoryTables.All, 100);
        var written = Assert.Single(stored);
        Assert.Equal("conv-7", written.SourceId); // no ":chunk-0" suffix -- single-segment passthrough
    }

    [Fact]
    public async Task DrainOnceAsync_OneBadRow_MarksItFailedAndStillProcessesOthers()
    {
        const string throwingResponse = "ROW-ONE-SHOULD-THROW this content triggers a scripted failure.";
        const string goodResponse = "ROW-TWO-SHOULD-SUCCEED the user likes hiking on weekends.";

        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-queue-drainer-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        var runtime = new FakePsmRuntime
        {
            OnStorageDecision = prompt => prompt.Contains("ROW-ONE-SHOULD-THROW")
                ? throw new InvalidOperationException("boom")
                : StoreEpisodicJson(goodResponse),
        };
        var service = new PsmService(store, runtime);
        var drainer = new RememberQueueDrainer(store, service);

        var badId = store.InsertPendingRememberRequest(
            userId: "queue-test-user-3", llmResponse: throwingResponse, userMessage: null,
            includeExistingMemories: true, extraTags: null,
            sourceKind: null, sourceId: null, sourceTimestamp: null, sourceLabel: null,
            domain: PsmDomain.Coding.ToString());
        var goodId = store.InsertPendingRememberRequest(
            userId: "queue-test-user-3", llmResponse: goodResponse, userMessage: null,
            includeExistingMemories: true, extraTags: null,
            sourceKind: null, sourceId: null, sourceTimestamp: null, sourceLabel: null,
            domain: PsmDomain.Coding.ToString());

        var processed = await drainer.DrainOnceAsync();
        Assert.Equal(2, processed);

        var failedRows = store.SelectPendingRememberRequestsByStatus("failed");
        Assert.Contains(failedRows, r => r.Id == badId);

        var doneRows = store.SelectPendingRememberRequestsByStatus("done");
        Assert.Contains(doneRows, r => r.Id == goodId);

        var stored = store.SelectMemories("queue-test-user-3", MemoryTables.All, 100);
        Assert.Contains(stored, m => m.Content.Contains(goodResponse));
    }
}
