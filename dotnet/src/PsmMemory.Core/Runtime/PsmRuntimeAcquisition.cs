using PsmMemory.Core.Runtime.WarmHost;

namespace PsmMemory.Core.Runtime;

/// <summary>
/// Holds whatever a runtime pair returned by <see cref="PsmRuntimeAcquisition.AcquireAsync"/> needs
/// disposed. When backed by a direct local load (<see cref="LlamaSharpPsmRuntime"/> +
/// <see cref="LlamaSharpEmbeddingRuntime"/>), that's the two real, unmanaged-resource-holding
/// runtimes. When backed by warm-host proxies (<see cref="WarmHostPsmRuntime"/> /
/// <see cref="WarmHostEmbeddingRuntime"/>), there is nothing local to dispose -- the actual model
/// lives in the separate warm-host process.
/// </summary>
public sealed class AcquiredRuntime : IAsyncDisposable
{
    public required IPsmRuntime Runtime { get; init; }
    public required IEmbeddingRuntime EmbeddingRuntime { get; init; }

    public IReadOnlyList<IDisposable> Disposables { get; init; } = Array.Empty<IDisposable>();

    public ValueTask DisposeAsync()
    {
        foreach (var disposable in Disposables) disposable.Dispose();
        return ValueTask.CompletedTask;
    }
}

/// <summary>
/// The single decision point a short-lived process's bootstrap goes through to get its
/// <see cref="IPsmRuntime"/>/<see cref="IEmbeddingRuntime"/> pair: reach an already-warm
/// <see cref="WarmHostServer"/> over loopback HTTP if one is configured and reachable, otherwise fall
/// back to loading the GGUF model directly in this process (the pre-warm-host behavior, unchanged).
/// A later wiring step (not part of this change) calls this from the CLI's bootstrap call sites.
/// </summary>
public static class PsmRuntimeAcquisition
{
    /// <param name="modelDir">Local GGUF model directory, used by the direct-load fallback (and by
    /// the warm host itself, once spawned).</param>
    /// <param name="warmHostOptions">Null means "daemon disabled for this call" -- go straight to a
    /// direct local load. Non-null means "try the warm host first."</param>
    /// <param name="daemonProcessArgs">Only used if <paramref name="warmHostOptions"/> is non-null
    /// (needed for the spawn-detached path inside <see cref="WarmHostClient.EnsureDaemonAsync"/>).</param>
    /// <param name="localLoader">Test seam: overrides the direct-local-load step so tests can verify
    /// the warm-host-unavailable fallback branch without needing a real GGUF model on disk. Production
    /// callers should omit this.</param>
    public static async Task<AcquiredRuntime> AcquireAsync(
        string modelDir,
        WarmHostOptions? warmHostOptions,
        string[]? daemonProcessArgs,
        CancellationToken ct = default,
        Func<string, CancellationToken, Task<AcquiredRuntime>>? localLoader = null)
    {
        if (warmHostOptions is not null)
        {
            try
            {
                var baseAddress = await WarmHostClient
                    .EnsureDaemonAsync(warmHostOptions, daemonProcessArgs ?? Array.Empty<string>(), ct)
                    .ConfigureAwait(false);
                return new AcquiredRuntime
                {
                    Runtime = new WarmHostPsmRuntime(baseAddress),
                    EmbeddingRuntime = new WarmHostEmbeddingRuntime(baseAddress),
                    // No local disposables -- the model lives in the warm-host process, not this one.
                };
            }
            catch (Exception ex)
            {
                await Console.Error.WriteLineAsync(
                        $"psm-memory: warm host unavailable ({ex.Message}), falling back to a direct local model load.")
                    .ConfigureAwait(false);
                // fall through to direct load below.
            }
        }

        var load = localLoader ?? DefaultLocalLoadAsync;
        return await load(modelDir, ct).ConfigureAwait(false);
    }

    private static async Task<AcquiredRuntime> DefaultLocalLoadAsync(string modelDir, CancellationToken ct)
    {
        var runtime = await LlamaSharpPsmRuntime.CreateAsync(modelDir, ct: ct).ConfigureAwait(false);
        var embeddingRuntime = await LlamaSharpEmbeddingRuntime.CreateAsync(modelDir, ct: ct).ConfigureAwait(false);
        return new AcquiredRuntime
        {
            Runtime = runtime,
            EmbeddingRuntime = embeddingRuntime,
            Disposables = new IDisposable[] { runtime, embeddingRuntime },
        };
    }
}
