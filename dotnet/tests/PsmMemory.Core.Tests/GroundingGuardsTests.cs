using PsmMemory.Core;
using PsmMemory.Core.Models;
using Xunit;

namespace PsmMemory.Core.Tests;

public class GroundingGuardsTests
{
    private static StorageDecision Decision(string content, string action = "store_episodic") => new()
    {
        Action = action,
        Memory = new MemoryPayload { Content = content },
        Reasoning = "test",
        RawJson = "{}"
    };

    [Fact]
    public void ApplyStorageGuards_AllowsContentGroundedInSource()
    {
        var source = "Sam mentioned he adopted a golden retriever puppy named Biscuit last weekend.";
        var decision = Decision("Sam adopted a golden retriever puppy named Biscuit.");

        var result = GroundingGuards.ApplyStorageGuards(source, decision);

        Assert.False(result.Rejected);
    }

    [Fact]
    public void ApplyStorageGuards_RejectsUngroundedContent()
    {
        var source = "The weather today is sunny with a light breeze.";
        var decision = Decision("The user's favorite programming language is Rust and they work at a fintech startup.");

        var result = GroundingGuards.ApplyStorageGuards(source, decision);

        Assert.True(result.Rejected);
        Assert.Equal("grounding_reject", result.GuardRoute);
    }

    [Fact]
    public void ApplyStorageGuards_RejectsCurriculumBleed()
    {
        var source = "Let's check the runpod checkpoint before running the gate6 probe again.";
        var decision = Decision("Let's check the runpod checkpoint before running the gate6 probe again.");

        var result = GroundingGuards.ApplyStorageGuards(source, decision);

        Assert.True(result.Rejected);
        Assert.Equal("grounding_reject_bleed", result.GuardRoute);
    }

    [Fact]
    public void ApplyStorageGuards_DoesNotRejectIgnoreDecisions()
    {
        var decision = new StorageDecision { Action = "ignore", Memory = null, Reasoning = "nothing to store", RawJson = "{}" };

        var result = GroundingGuards.ApplyStorageGuards("anything at all", decision);

        Assert.False(result.Rejected);
    }

    [Theory]
    [InlineData("checkpoint saved to runpod", true)]
    [InlineData("Sam adopted a puppy named Biscuit", false)]
    public void HasCurriculumBleed_DetectsBlocklistedPhrases(string text, bool expected)
    {
        Assert.Equal(expected, GroundingGuards.HasCurriculumBleed(text));
    }

    [Fact]
    public void GroundingOverlapScore_TreatsEmptyInputAsGrounded()
    {
        var score = GroundingGuards.GroundingOverlapScore("", "some stored text");
        Assert.True(score.Grounded);
    }
}
