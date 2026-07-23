using PsmMemory.Core;
using PsmMemory.Core.Models;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Tests for <see cref="Indexables"/> -- the deterministic (no-LLM) indexable synthesis/ranking
/// ported from psm-core/src/indexables.ts. The review-pr fixture is the canonical example from
/// the original TS repo's tests/indexables.test.ts:12-31.
/// </summary>
public class IndexablesTests
{
    // The exact fixture from tests/indexables.test.ts in the original TS repo.
    private const string ReviewPrFixture =
        "# Review a pull request\n\n" +
        "1. Get PR info with `gh pr view`.\n" +
        "2. Check the target branch tracks the intended base.\n" +
        "3. List changed files with `gh pr diff --name-only`.\n" +
        "4. Review each changed file for correctness and scope.\n" +
        "5. Summarize findings and request changes or approve.";

    [Fact]
    public void BuildIndexablesForRemember_SynthesizesWorkflowRow_ForReviewPrFixture()
    {
        var rows = Indexables.BuildIndexablesForRemember(new Indexables.BuildIndexablesInput
        {
            LlmResponse = ReviewPrFixture,
            MemoryContent = ReviewPrFixture
        });

        var row = Assert.Single(rows);
        Assert.Equal(IndexableKinds.Workflow, row.Kind);
        Assert.Equal("review-pr", row.Key);
        Assert.NotNull(row.Steps);
        Assert.Equal(5, row.Steps!.Count);
        Assert.Equal("get_pr_info_with_gh_pr_view", row.Steps[0]);
        Assert.Equal(0.95, row.Salience);
        Assert.Contains("workflow:review-pr", row.Tags!);
        Assert.Contains("workflow", row.Tags!);
    }

    [Fact]
    public void BuildIndexablesForRemember_FallsBackToMnemonicPlusFactAnchor_WhenNoWorkflowDetected()
    {
        const string content =
            "The user prefers dark mode in the editor and wants notifications muted after 9pm.";

        var rows = Indexables.BuildIndexablesForRemember(new Indexables.BuildIndexablesInput
        {
            LlmResponse = content,
            MemoryContent = content,
            Tags = new List<string> { "preference" },
            Facts = new List<MemoryFactPayload>
            {
                new() { Subject = "user", Predicate = "prefers", ValueText = "dark_mode" }
            }
        });

        Assert.Equal(2, rows.Count);
        Assert.Equal(IndexableKinds.Mnemonic, rows[0].Kind);
        Assert.Equal(IndexableKinds.FactAnchor, rows[1].Kind);
        Assert.NotEqual(rows[0].Key, rows[1].Key);
        Assert.False(string.IsNullOrEmpty(rows[0].Key));
        Assert.False(string.IsNullOrEmpty(rows[1].Key));
        // fact_anchor salience is max(mnemonicSalience, 0.82) per indexables.ts.
        Assert.True(rows[1].Salience >= 0.82);
    }

    [Fact]
    public void BuildIndexablesForRemember_ProducesOnlyMnemonic_WhenNoQualifyingFactSupplied()
    {
        const string content = "The deployment window moved to Thursday evenings.";

        var rows = Indexables.BuildIndexablesForRemember(new Indexables.BuildIndexablesInput
        {
            LlmResponse = content,
            MemoryContent = content
        });

        var row = Assert.Single(rows);
        Assert.Equal(IndexableKinds.Mnemonic, row.Kind);
    }

    [Fact]
    public void BuildIndexablesForRemember_RequiresBothWorkflowKeyAndTwoNumberedSteps()
    {
        // Has the review-pr header pattern but only ONE numbered step -- workflow detection must
        // NOT fire (needs >=2 steps), so this should fall through to the mnemonic path.
        const string content = "# Review a pull request\n\n1. Get PR info with `gh pr view`.";

        var rows = Indexables.BuildIndexablesForRemember(new Indexables.BuildIndexablesInput
        {
            LlmResponse = content,
            MemoryContent = content
        });

        var row = Assert.Single(rows);
        Assert.Equal(IndexableKinds.Mnemonic, row.Kind);
    }

    [Fact]
    public void BuildIndexablesForRemember_NormalizesExplicitIndexables_WithoutSynthesizing()
    {
        var rows = Indexables.BuildIndexablesForRemember(new Indexables.BuildIndexablesInput
        {
            LlmResponse = "irrelevant text, should not be used for synthesis",
            MemoryContent = "irrelevant content",
            ExplicitIndexables = new List<IndexablePayload>
            {
                new() { Kind = "mnemonic", Key = "My Custom KEY!!", Salience = 1.5, Tags = new List<string> { "already tagged" } }
            }
        });

        var row = Assert.Single(rows);
        Assert.Equal("my-custom-key", row.Key); // cleanKey: lowercase, non [a-z0-9-] runs -> "-", trim
        Assert.Equal(1.0, row.Salience); // clamp01
        Assert.Contains("already_tagged", row.Tags!); // uniqueTags replaces whitespace with "_"
    }

    private static IndexableRecord Row(
        string id, string kind, string key, double salience,
        string? hint = null, List<string>? tags = null, List<string>? steps = null) => new()
    {
        Id = id,
        UserId = "user-1",
        Kind = kind,
        Key = key,
        Salience = salience,
        ReconstructiveHint = hint,
        Tags = tags ?? new List<string>(),
        Steps = steps ?? new List<string>()
    };

    [Fact]
    public void RankIndexables_OrdersByScoreAndFiltersBelowThreshold()
    {
        var rows = new List<IndexableRecord>
        {
            Row("i1", IndexableKinds.Workflow, "review-pr", 0.95,
                hint: "Review a pull request end to end.",
                tags: new List<string> { "workflow:review-pr", "workflow" },
                steps: new List<string> { "get_pr_info", "check_base_branch" }),
            // Low salience, zero relevance to the query -- must fall below the 0.35 threshold.
            Row("i2", IndexableKinds.Mnemonic, "grocery-list-milk-eggs", 0.2,
                hint: "Buy milk and eggs."),
            // Some relevance and mid salience -- should survive but rank behind i1.
            Row("i3", IndexableKinds.Mnemonic, "pull-request-notes", 0.5,
                hint: "Some pull request notes.")
        };

        var ranked = Indexables.RankIndexables("how do I review a pull request", rows, 5);

        Assert.Equal(new[] { "i1", "i3" }, ranked.Select(r => r.Id));
    }

    [Fact]
    public void RankIndexables_RespectsTopK()
    {
        var rows = Enumerable.Range(0, 10)
            .Select(i => Row($"i{i}", IndexableKinds.Mnemonic, $"widget-testing-{i}", 0.6, hint: "testing widget"))
            .ToList();

        var ranked = Indexables.RankIndexables("testing widget", rows, 3);

        Assert.True(ranked.Count <= 3);
    }

    [Fact]
    public void NormalizeRecallKey_LowercasesAndHyphenatesAndTrims()
    {
        Assert.Equal("how-do-i-review-a-pr", Indexables.NormalizeRecallKey("  How do I review a PR?! "));
    }
}
