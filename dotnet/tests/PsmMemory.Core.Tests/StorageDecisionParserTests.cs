using PsmMemory.Core;
using PsmMemory.Core.Decisions;
using PsmMemory.Core.Models;
using Xunit;

namespace PsmMemory.Core.Tests;

public class StorageDecisionParserTests
{
    [Fact]
    public void ParseStorageDecision_ParsesValidJson()
    {
        var raw = """
            {"reasoning":"Durable fact about a pet.","action":"store_episodic","memory":{"content":"Sam adopted a puppy named Biscuit.","type":"episodic","confidence":0.9},"facts":[],"indexables":[]}
            """;

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "fallback content");

        Assert.Null(decision.ParseError);
        Assert.Equal(Actions.Kinds.StoreEpisodic, decision.Action);
        Assert.NotNull(decision.Memory);
        Assert.Equal("Sam adopted a puppy named Biscuit.", decision.Memory!.Content);
        Assert.Equal(0.9, decision.Confidence);
    }

    [Fact]
    public void ParseStorageDecision_ParsesIgnoreWithNullMemory()
    {
        var raw = """{"reasoning":"Nothing durable.","action":"ignore","memory":null,"facts":[]}""";

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "fallback");

        Assert.Null(decision.ParseError);
        Assert.Equal(Actions.Kinds.Ignore, decision.Action);
        Assert.Null(decision.Memory);
    }

    [Fact]
    public void ParseStorageDecision_ExtractsJsonEmbeddedInProseWithNoTrailingText()
    {
        // the parser locates the first '{' .. last '}', so it can recover JSON that has leading
        // prose but must not have trailing prose after the final closing brace.
        var raw = "Sure, here is the decision:\n{\"reasoning\":\"ok\",\"action\":\"ignore\",\"memory\":null}";

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "fallback");

        Assert.Null(decision.ParseError);
        Assert.Equal(Actions.Kinds.Ignore, decision.Action);
    }

    [Fact]
    public void ParseStorageDecision_MalformedJsonFallsBackToFailSafeShapeWithParseError()
    {
        var raw = "this is not json at all, no braces here";

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "the original assistant response");

        Assert.NotNull(decision.ParseError);
        Assert.Equal("parse_fallback", decision.Memory!.Tags!.Single());
        Assert.Equal("the original assistant response", decision.Memory!.Content);
    }

    [Fact]
    public void ParseStorageDecision_TruncatedJsonTriggersParseError()
    {
        var raw = """{"reasoning":"cut off","action":"store_episodic","memory":{"content":"oops""";

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "fallback content");

        Assert.NotNull(decision.ParseError);
    }

    [Fact]
    public void FailsafeDecision_IsAlwaysIgnoreWithNullMemory()
    {
        var failsafe = StorageDecisionParser.FailsafeDecision("garbage", "boom");

        Assert.Equal(Actions.Kinds.Ignore, failsafe.Action);
        Assert.Null(failsafe.Memory);
        Assert.Contains("unparseable", failsafe.Reasoning);
    }

    [Fact]
    public void ParseStorageDecision_DropsFactsWithoutEvidenceText()
    {
        var raw = """
            {"reasoning":"r","action":"store_episodic","memory":{"content":"c"},
             "facts":[{"subject":"Sam","predicate":"has_pet","value_text":"Biscuit","confidence":0.9,"inference_kind":"explicit"}]}
            """;

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "fallback");

        Assert.Empty(decision.Facts);
    }

    [Fact]
    public void ParseStorageDecision_KeepsValidExplicitFactWithEvidence()
    {
        var raw = """
            {"reasoning":"r","action":"store_episodic","memory":{"content":"c"},
             "facts":[{"subject":"Sam","predicate":"has_pet","value_text":"Biscuit","confidence":0.9,"inference_kind":"explicit","evidence_text":"Sam adopted Biscuit"}]}
            """;

        var decision = StorageDecisionParser.ParseStorageDecision(raw, "fallback");

        var fact = Assert.Single(decision.Facts);
        Assert.Equal("has_pet", fact.Predicate);
    }

    [Fact]
    public void ParseRecallPlan_ParsesValidPlan()
    {
        var raw = """{"intent":"recall","target_tables":["semantic","episodic"],"ranking_hints":["sam","dog"],"top_k":3}""";

        var plan = StorageDecisionParser.ParseRecallPlan(raw, "what is sam's dog's name", 5);

        Assert.Null(plan.ParseError);
        Assert.False(plan.PlanFallback);
        Assert.Equal(new[] { "semantic", "episodic" }, plan.TargetTables);
        Assert.Equal(3, plan.TopK);
    }

    [Fact]
    public void ParseRecallPlan_FallsBackOnMalformedJson()
    {
        var plan = StorageDecisionParser.ParseRecallPlan("not json", "question text", 5);

        Assert.True(plan.PlanFallback);
        Assert.NotNull(plan.ParseError);
        Assert.Equal(new[] { MemoryTables.Semantic, MemoryTables.Episodic }, plan.TargetTables);
    }

    [Fact]
    public void ParseConsolidationDecision_ParsesValidDecision()
    {
        var raw = """{"reasoning":"restates the same fact","action":"update_existing","target_memory_id":"mem-1","merged_content":"merged text"}""";

        var decision = StorageDecisionParser.ParseConsolidationDecision(raw);

        Assert.Null(decision.ParseError);
        Assert.Equal(Actions.Kinds.UpdateExisting, decision.Action);
        Assert.Equal("mem-1", decision.TargetMemoryId);
        Assert.Equal("merged text", decision.MergedContent);
    }

    [Fact]
    public void ParseConsolidationDecision_FallsBackToStoreEpisodicOnParseFailure()
    {
        var decision = StorageDecisionParser.ParseConsolidationDecision("garbage output");

        Assert.NotNull(decision.ParseError);
        Assert.Equal(Actions.Kinds.StoreEpisodic, decision.Action);
    }
}
