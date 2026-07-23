using System.Text.Json;
using PsmMemory.Core.Runtime.WarmHost;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Confirms <see cref="WarmHostRequest"/>/<see cref="WarmHostResponse"/> round-trip through
/// <see cref="System.Text.Json"/> using the same camelCase, case-insensitive options
/// (<c>JsonSerializerDefaults.Web</c>) that both <see cref="WarmHostServer"/> and
/// <see cref="WarmHostClient"/> use on the wire.
/// </summary>
public class WarmHostProtocolTests
{
    private static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);

    [Fact]
    public void WarmHostRequest_RoundTrips_ForAGenerateOperation()
    {
        var original = new WarmHostRequest
        {
            Operation = "storage-decision",
            Prompt = "some prompt text",
            Domain = "conversational",
        };

        var json = JsonSerializer.Serialize(original, Options);
        Assert.Contains("\"operation\"", json); // camelCase on the wire
        var roundTripped = JsonSerializer.Deserialize<WarmHostRequest>(json, Options);

        Assert.NotNull(roundTripped);
        Assert.Equal(original.Operation, roundTripped!.Operation);
        Assert.Equal(original.Prompt, roundTripped.Prompt);
        Assert.Equal(original.Domain, roundTripped.Domain);
        Assert.Null(roundTripped.Text);
    }

    [Fact]
    public void WarmHostRequest_RoundTrips_ForEmbedOperation()
    {
        var original = new WarmHostRequest { Operation = "embed", Text = "text to embed" };

        var json = JsonSerializer.Serialize(original, Options);
        var roundTripped = JsonSerializer.Deserialize<WarmHostRequest>(json, Options);

        Assert.NotNull(roundTripped);
        Assert.Equal("embed", roundTripped!.Operation);
        Assert.Equal("text to embed", roundTripped.Text);
        Assert.Null(roundTripped.Prompt);
        Assert.Equal("coding", roundTripped.Domain); // default when not overridden.
    }

    [Fact]
    public void WarmHostRequest_Domain_DefaultsToCoding_WhenOmittedFromWire()
    {
        var roundTripped = JsonSerializer.Deserialize<WarmHostRequest>("{\"operation\":\"embed\",\"text\":\"x\"}", Options);
        Assert.NotNull(roundTripped);
        Assert.Equal("coding", roundTripped!.Domain);
    }

    [Fact]
    public void WarmHostResponse_RoundTrips_ForAGenerateResult()
    {
        var original = new WarmHostResponse { Ok = true, Result = "{\"action\":\"store_episodic\"}" };

        var json = JsonSerializer.Serialize(original, Options);
        Assert.Contains("\"ok\"", json);
        var roundTripped = JsonSerializer.Deserialize<WarmHostResponse>(json, Options);

        Assert.NotNull(roundTripped);
        Assert.True(roundTripped!.Ok);
        Assert.Equal(original.Result, roundTripped.Result);
        Assert.Null(roundTripped.Embedding);
        Assert.Null(roundTripped.Error);
    }

    [Fact]
    public void WarmHostResponse_RoundTrips_ForAnEmbeddingResult()
    {
        var original = new WarmHostResponse { Ok = true, Embedding = new[] { 0.1f, -0.2f, 0.3f } };

        var json = JsonSerializer.Serialize(original, Options);
        var roundTripped = JsonSerializer.Deserialize<WarmHostResponse>(json, Options);

        Assert.NotNull(roundTripped);
        Assert.True(roundTripped!.Ok);
        Assert.Equal(original.Embedding, roundTripped.Embedding);
    }

    [Fact]
    public void WarmHostResponse_RoundTrips_ForAnErrorResult()
    {
        var original = new WarmHostResponse { Ok = false, Error = "unsupported_operation" };

        var json = JsonSerializer.Serialize(original, Options);
        var roundTripped = JsonSerializer.Deserialize<WarmHostResponse>(json, Options);

        Assert.NotNull(roundTripped);
        Assert.False(roundTripped!.Ok);
        Assert.Equal("unsupported_operation", roundTripped.Error);
        Assert.Null(roundTripped.Result);
        Assert.Null(roundTripped.Embedding);
    }
}
