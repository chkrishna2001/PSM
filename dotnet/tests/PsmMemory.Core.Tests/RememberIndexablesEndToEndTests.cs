using System.Text.Json;
using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// End-to-end proof that <see cref="PsmService.RememberAsync"/> synthesizes indexables
/// deterministically (via <see cref="Indexables.BuildIndexablesForRemember"/>) when the model's
/// storage decision didn't already emit its own, that they're actually persisted to the store (not
/// just returned in the RememberResult), and that they surface via
/// <see cref="PsmService.RecallAsync"/>'s Indexables/Workflows fields but NOT via
/// <see cref="PsmService.ContextAsync"/> -- matching service.ts's recall()/context() split (only
/// recall() calls rankIndexables in the original TS source).
/// </summary>
public class RememberIndexablesEndToEndTests
{
    // The exact fixture from tests/indexables.test.ts in the original TS repo.
    private const string ReviewPrFixture =
        "# Review a pull request\n\n" +
        "1. Get PR info with `gh pr view`.\n" +
        "2. Check the target branch tracks the intended base.\n" +
        "3. List changed files with `gh pr diff --name-only`.\n" +
        "4. Review each changed file for correctness and scope.\n" +
        "5. Summarize findings and request changes or approve.";

    private sealed class FakePsmRuntime : IPsmRuntime
    {
        public required string StorageDecisionJson { get; init; }

        public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult(StorageDecisionJson);

        public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("no recall-plan adapter needed -- this test relies on DeterministicPlanFallback");

        public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            throw new NotSupportedException("store is empty for this test user, so ConsolidateAsync never calls this");
    }

    [Fact]
    public async Task RememberAsync_SynthesizesAndPersistsWorkflowIndexable_SurfacedByRecallButNotContext()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-indexables-e2e-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();

        var storageJson = JsonSerializer.Serialize(new
        {
            reasoning = "Storing the PR review workflow the user just described.",
            action = "store_episodic",
            memory = new { content = ReviewPrFixture },
            facts = Array.Empty<object>(),
            indexables = Array.Empty<object>()
        });

        var runtime = new FakePsmRuntime { StorageDecisionJson = storageJson };
        var service = new PsmService(store, runtime);
        const string userId = "indexables-e2e-user";

        var rememberResult = await service.RememberAsync(new RememberRequest
        {
            UserId = userId,
            LlmResponse = ReviewPrFixture,
            IncludeExistingMemories = true
        });

        Assert.NotEqual(Actions.Kinds.Ignore, rememberResult.Action);
        var synthesized = Assert.Single(rememberResult.Indexables);
        Assert.Equal(IndexableKinds.Workflow, synthesized.Kind);
        Assert.Equal("review-pr", synthesized.Key);

        // Persisted directly in the store, not just returned in the in-memory result.
        var storedIndexables = store.SelectIndexables(userId, 100);
        Assert.Contains(storedIndexables, row => row.Kind == IndexableKinds.Workflow && row.Key == "review-pr");

        const string question = "how do I review a pull request";

        var recallResult = await service.RecallAsync(new RecallRequest { UserId = userId, Question = question, TopK = 5 });
        Assert.Contains(recallResult.Indexables, row => row.Key == "review-pr");
        Assert.Contains(recallResult.Workflows, row => row.Key == "review-pr");

        // context() must never surface indexables/workflows, even for the exact same question.
        var contextResult = await service.ContextAsync(new ContextRequest { UserId = userId, Prompt = question, TopK = 5 });
        Assert.Empty(contextResult.Indexables);
        Assert.Empty(contextResult.Workflows);
    }
}
