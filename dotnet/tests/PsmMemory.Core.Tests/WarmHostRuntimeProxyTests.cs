using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Runtime.WarmHost;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Verifies <see cref="WarmHostPsmRuntime"/>/<see cref="WarmHostEmbeddingRuntime"/> are pure IPC
/// forwarders: the right JSON request goes out for each of the four warm-host operations and the
/// right value comes back parsed correctly. Uses the standard <see cref="HttpClient"/> testing
/// pattern (a fake <see cref="HttpMessageHandler"/>) -- no real <see cref="System.Net.HttpListener"/>
/// needed.
/// </summary>
public class WarmHostRuntimeProxyTests
{
    private static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);
    private static readonly Uri BaseAddress = new("http://127.0.0.1:12345/");

    /// <summary>Captures the outgoing request and returns a canned <see cref="WarmHostResponse"/>.</summary>
    private sealed class FakeHandler : HttpMessageHandler
    {
        public HttpRequestMessage? LastRequest { get; private set; }
        public WarmHostRequest? LastBody { get; private set; }
        public Func<WarmHostRequest, WarmHostResponse> RespondWith { get; init; } = _ => new WarmHostResponse { Ok = true };

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            LastRequest = request;
            if (request.Content is not null)
            {
                var json = await request.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                LastBody = JsonSerializer.Deserialize<WarmHostRequest>(json, Options);
            }

            var responseBody = RespondWith(LastBody!);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = JsonContent.Create(responseBody, options: Options),
            };
        }
    }

    [Fact]
    public async Task GenerateStorageDecisionAsync_SendsStorageDecisionOperation_AndReturnsResult()
    {
        var handler = new FakeHandler { RespondWith = _ => new WarmHostResponse { Ok = true, Result = "{\"action\":\"store_episodic\"}" } };
        var runtime = new WarmHostPsmRuntime(BaseAddress, new HttpClient(handler));

        var result = await runtime.GenerateStorageDecisionAsync("some prompt", PsmDomain.Conversational);

        Assert.Equal("{\"action\":\"store_episodic\"}", result);
        Assert.NotNull(handler.LastBody);
        Assert.Equal("storage-decision", handler.LastBody!.Operation);
        Assert.Equal("some prompt", handler.LastBody.Prompt);
        Assert.Equal("conversational", handler.LastBody.Domain);
        Assert.Equal(new Uri(BaseAddress, "v1"), handler.LastRequest!.RequestUri);
        Assert.Equal(HttpMethod.Post, handler.LastRequest.Method);
    }

    [Fact]
    public async Task GenerateRecallPlanAsync_SendsRecallPlanOperation_AndReturnsResult()
    {
        var handler = new FakeHandler { RespondWith = _ => new WarmHostResponse { Ok = true, Result = "{\"action\":\"recall\"}" } };
        var runtime = new WarmHostPsmRuntime(BaseAddress, new HttpClient(handler));

        var result = await runtime.GenerateRecallPlanAsync("recall prompt", PsmDomain.Coding);

        Assert.Equal("{\"action\":\"recall\"}", result);
        Assert.Equal("recall-plan", handler.LastBody!.Operation);
        Assert.Equal("coding", handler.LastBody.Domain);
    }

    [Fact]
    public async Task GenerateConsolidationDecisionAsync_SendsConsolidationDecisionOperation_AndReturnsResult()
    {
        var handler = new FakeHandler { RespondWith = _ => new WarmHostResponse { Ok = true, Result = "{\"action\":\"merge\"}" } };
        var runtime = new WarmHostPsmRuntime(BaseAddress, new HttpClient(handler));

        var result = await runtime.GenerateConsolidationDecisionAsync("consolidation prompt");

        Assert.Equal("{\"action\":\"merge\"}", result);
        Assert.Equal("consolidation-decision", handler.LastBody!.Operation);
        Assert.Equal("consolidation prompt", handler.LastBody.Prompt);
    }

    [Fact]
    public async Task EmbedAsync_SendsEmbedOperation_AndReturnsParsedEmbedding()
    {
        var expected = new[] { 0.1f, 0.2f, -0.3f };
        var handler = new FakeHandler { RespondWith = _ => new WarmHostResponse { Ok = true, Embedding = expected } };
        var runtime = new WarmHostEmbeddingRuntime(BaseAddress, new HttpClient(handler));

        var embedding = await runtime.EmbedAsync("text to embed");

        Assert.Equal(expected, embedding);
        Assert.Equal("embed", handler.LastBody!.Operation);
        Assert.Equal("text to embed", handler.LastBody.Text);
    }

    [Fact]
    public async Task GenerateStorageDecisionAsync_WhenServerReturnsError_ThrowsWithErrorMessage()
    {
        var handler = new FakeHandler { RespondWith = _ => new WarmHostResponse { Ok = false, Error = "missing_prompt" } };
        var runtime = new WarmHostPsmRuntime(BaseAddress, new HttpClient(handler));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => runtime.GenerateStorageDecisionAsync("prompt"));
        Assert.Contains("missing_prompt", ex.Message);
    }
}
