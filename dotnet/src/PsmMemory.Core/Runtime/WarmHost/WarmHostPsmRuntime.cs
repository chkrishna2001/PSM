namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Pure IPC forwarder to a <see cref="WarmHostServer"/>: no local model logic at all. Implements
/// <see cref="IPsmRuntime"/> exactly like <see cref="LlamaSharpPsmRuntime"/> does, so
/// <see cref="PsmService"/> (which only ever talks to runtimes through this interface) needs zero
/// changes to accept either one.
/// </summary>
public sealed class WarmHostPsmRuntime : IPsmRuntime
{
    private readonly Uri _baseAddress;
    private readonly HttpClient? _httpClient;

    public WarmHostPsmRuntime(Uri baseAddress) : this(baseAddress, null)
    {
    }

    /// <summary>Test seam: pass an <see cref="HttpClient"/> wired to a fake
    /// <see cref="HttpMessageHandler"/> to assert outgoing requests/responses without a real
    /// <see cref="System.Net.HttpListener"/>.</summary>
    public WarmHostPsmRuntime(Uri baseAddress, HttpClient? httpClient)
    {
        _baseAddress = baseAddress;
        _httpClient = httpClient;
    }

    public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
        SendAsync("storage-decision", prompt, domain, ct);

    public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
        SendAsync("recall-plan", prompt, domain, ct);

    public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
        SendAsync("consolidation-decision", prompt, domain, ct);

    private async Task<string> SendAsync(string operation, string prompt, PsmDomain domain, CancellationToken ct)
    {
        var request = new WarmHostRequest { Operation = operation, Prompt = prompt, Domain = domain.ToString().ToLowerInvariant() };
        var response = await WarmHostClient.PostAsync(_baseAddress, request, ct, _httpClient).ConfigureAwait(false);
        return response.Result ?? string.Empty;
    }
}
