namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Pure IPC forwarder to a <see cref="WarmHostServer"/>'s embedding endpoint: no local model logic at
/// all. Implements <see cref="IEmbeddingRuntime"/> exactly like
/// <see cref="LlamaSharpEmbeddingRuntime"/> does.
/// </summary>
public sealed class WarmHostEmbeddingRuntime : IEmbeddingRuntime
{
    private readonly Uri _baseAddress;
    private readonly HttpClient? _httpClient;

    public WarmHostEmbeddingRuntime(Uri baseAddress) : this(baseAddress, null)
    {
    }

    /// <summary>Test seam: pass an <see cref="HttpClient"/> wired to a fake
    /// <see cref="HttpMessageHandler"/> to assert outgoing requests/responses without a real
    /// <see cref="System.Net.HttpListener"/>.</summary>
    public WarmHostEmbeddingRuntime(Uri baseAddress, HttpClient? httpClient)
    {
        _baseAddress = baseAddress;
        _httpClient = httpClient;
    }

    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        var request = new WarmHostRequest { Operation = "embed", Text = text };
        var response = await WarmHostClient.PostAsync(_baseAddress, request, ct, _httpClient).ConfigureAwait(false);
        return response.Embedding ?? Array.Empty<float>();
    }
}
