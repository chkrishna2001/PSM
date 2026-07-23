using PsmMemory.Core;
using PsmMemory.Core.Models;
using Xunit;

namespace PsmMemory.Core.Tests;

public class RankingTests
{
    private static MemoryRecord Memory(string id, string content, string table = MemoryTables.Episodic, double confidence = 0.8, double strength = 0.7) => new()
    {
        Id = id,
        UserId = "user-1",
        Content = content,
        Table = table,
        Confidence = confidence,
        Strength = strength
    };

    [Fact]
    public void HybridRankMemories_RanksMoreRelevantMemoryFirst()
    {
        var memories = new List<MemoryRecord>
        {
            Memory("m1", "Sam went hiking in the Rockies with his dog last weekend."),
            Memory("m2", "The user's favorite pizza topping is pepperoni."),
            Memory("m3", "Sam's dog Biscuit loves swimming in the lake near the cabin.")
        };

        var ranked = Ranking.HybridRankMemories("Sam's dog", memories, new Ranking.HybridRankOptions { TopK = 5 });

        Assert.NotEmpty(ranked);
        Assert.Contains(ranked[0].Id, new[] { "m1", "m3" });
        Assert.DoesNotContain(ranked, r => r.Id == "m2" && r.Score > ranked[0].Score);
    }

    [Fact]
    public void HybridRankMemories_RespectsTopK()
    {
        var memories = Enumerable.Range(0, 10).Select(i => Memory($"m{i}", $"fact number {i} about testing")).ToList();

        var ranked = Ranking.HybridRankMemories("testing fact", memories, new Ranking.HybridRankOptions { TopK = 3 });

        Assert.True(ranked.Count <= 3);
    }

    [Fact]
    public void HybridRankMemories_AppliesMinScoreFilter()
    {
        var memories = new List<MemoryRecord> { Memory("m1", "completely unrelated content about gardening tools") };

        var ranked = Ranking.HybridRankMemories("quantum computing research", memories, new Ranking.HybridRankOptions { TopK = 5, MinScore = 0.9 });

        Assert.Empty(ranked);
    }

    [Fact]
    public void HybridRankMemories_BoostsPreferredTables()
    {
        var memories = new List<MemoryRecord>
        {
            Memory("episodic-1", "project deadline is next Friday", MemoryTables.Episodic),
            Memory("semantic-1", "project deadline is next Friday", MemoryTables.Semantic)
        };

        var rankedPreferSemantic = Ranking.HybridRankMemories("project deadline", memories, new Ranking.HybridRankOptions
        {
            TopK = 5,
            PreferredTables = new[] { MemoryTables.Semantic }
        });

        // suppressDuplicateContent collapses identical content, so we assert the table boost is
        // reflected in the score computation instead of survivorship.
        var semanticOnly = Ranking.HybridRankMemories("project deadline", new List<MemoryRecord> { memories[1] }, new Ranking.HybridRankOptions
        {
            TopK = 5,
            PreferredTables = new[] { MemoryTables.Semantic }
        }).Single();
        var episodicOnly = Ranking.HybridRankMemories("project deadline", new List<MemoryRecord> { memories[0] }, new Ranking.HybridRankOptions
        {
            TopK = 5,
            PreferredTables = new[] { MemoryTables.Semantic }
        }).Single();

        Assert.True(semanticOnly.Score > episodicOnly.Score);
        Assert.NotEmpty(rankedPreferSemantic);
    }

    [Fact]
    public void Tokenize_LowercasesAndDropsStopwordsAndShortTokens()
    {
        var tokens = Ranking.Tokenize("The Cats and Dogs are running with You");

        Assert.DoesNotContain("the", tokens);
        Assert.DoesNotContain("and", tokens);
        Assert.DoesNotContain("are", tokens);
        Assert.DoesNotContain("you", tokens);
        Assert.Contains("cat", tokens); // "cats" -> "cat" via plural stripping
        Assert.Contains("dog", tokens); // "dogs" -> "dog"
    }

    private static MemoryFactRecord Fact(
        string id, string subject, string predicate, string valueText,
        string? factType = null, double? confidence = 0.75, string? evidenceText = null) => new()
    {
        Id = id,
        UserId = "user-1",
        Subject = subject,
        Predicate = predicate,
        ValueText = valueText,
        FactType = factType,
        Confidence = confidence,
        EvidenceText = evidenceText
    };

    [Fact]
    public void RankFacts_ReturnsEmpty_WhenQueryHasNoTokens()
    {
        var facts = new List<MemoryFactRecord> { Fact("f1", "user", "likes", "pizza") };

        // "the" alone tokenizes to nothing (stopword) -- rankFacts short-circuits to [] rather
        // than dividing by a zero-length token count.
        var ranked = Ranking.RankFacts("the", facts, 5);

        Assert.Empty(ranked);
    }

    [Fact]
    public void RankFacts_RanksSubjectAndPredicateMatchHigherThanPredicateOnlyMatch()
    {
        // f1 matches both subject ("sam") and predicate ("favorite_language") tokens from the
        // query; f2 only shares the "favorite" predicate token. Both clear the 0.2 threshold, but
        // f1's subject+predicate hits (0.25 + 0.3) must put it strictly ahead.
        var facts = new List<MemoryFactRecord>
        {
            Fact("f1", "sam", "favorite_language", "python"),
            Fact("f2", "unrelated_person", "favorite_food", "sushi")
        };

        var ranked = Ranking.RankFacts("what is sam's favorite language", facts, 5);

        Assert.NotEmpty(ranked);
        Assert.Equal("f1", ranked[0].Id);
        Assert.True(ranked[0].Score > ranked.First(r => r.Id == "f2").Score);
    }

    [Fact]
    public void RankFacts_AppliesTemporalBoost_ForTemporalFactTypeOnTemporalQuestion()
    {
        // Note: "when" is itself a Ranking.Stopwords entry (stripped before factScore ever sees
        // it) -- exactly matching the original TS's shared tokenize(), which has the same quirk.
        // "year" is not a stopword, so it's the token that actually trips the temporal-question check.
        var facts = new List<MemoryFactRecord>
        {
            Fact("temporal", "sam", "started_job", "2024", factType: "temporal_fact"),
            Fact("plain", "sam", "started_job", "2024")
        };

        var ranked = Ranking.RankFacts("what year did sam start the job", facts, 5);

        Assert.Equal(2, ranked.Count);
        Assert.Equal("temporal", ranked[0].Id);
        Assert.True(ranked[0].Score > ranked[1].Score);
    }

    [Fact]
    public void RankFacts_FiltersBelowScoreThreshold()
    {
        var facts = new List<MemoryFactRecord> { Fact("f1", "unrelated", "unrelated_predicate", "unrelated_value") };

        var ranked = Ranking.RankFacts("completely different topic entirely", facts, 5);

        Assert.Empty(ranked);
    }

    [Fact]
    public void RankFacts_RespectsTopK()
    {
        var facts = Enumerable.Range(0, 10)
            .Select(i => Fact($"f{i}", "widget", "has_count", $"testing widget number {i}"))
            .ToList();

        var ranked = Ranking.RankFacts("testing widget", facts, 3);

        Assert.True(ranked.Count <= 3);
    }
}
