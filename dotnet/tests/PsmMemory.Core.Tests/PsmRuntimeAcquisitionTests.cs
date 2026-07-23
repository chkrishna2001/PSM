using PsmMemory.Core.Runtime;
using PsmMemory.Core.Runtime.WarmHost;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Verifies <see cref="PsmRuntimeAcquisition.AcquireAsync"/>'s fallback branch: when the warm host is
/// unavailable, it logs and falls back to a direct local load rather than throwing the warm-host
/// failure back at the caller.
///
/// <see cref="LlamaSharpPsmRuntime.CreateAsync"/> can't run in this test environment without a real
/// GGUF model on disk (and, if the model directory is missing its adapters, it would attempt a real
/// network download from Hugging Face -- undesirable in a unit test). So this test uses
/// <see cref="PsmRuntimeAcquisition.AcquireAsync"/>'s <c>localLoader</c> test seam to observe that the
/// fallback CODE PATH is actually reached (a fake local loader is invoked, and the warm-host-failure
/// message is logged to stderr) rather than attempting a full, real fallback load.
///
/// The warm-host failure itself is made fast and side-effect-free (no process spawned, no network
/// call) by pre-creating a fresh (non-stale) lock file directly: this makes
/// <see cref="WarmHostClient.EnsureDaemonAsync"/> take its "someone else is spawning -- poll for them"
/// branch, which then times out against a tiny <see cref="WarmHostOptions.StartupTimeout"/> since no
/// real daemon ever writes daemon.json.
/// </summary>
public class PsmRuntimeAcquisitionTests
{
    private sealed class FakePsmRuntime : IPsmRuntime
    {
        public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult("fake-storage");
        public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult("fake-recall");
        public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default) =>
            Task.FromResult("fake-consolidation");
    }

    private sealed class FakeEmbeddingRuntime : IEmbeddingRuntime
    {
        public Task<float[]> EmbedAsync(string text, CancellationToken ct = default) => Task.FromResult(Array.Empty<float>());
    }

    [Fact]
    public async Task AcquireAsync_WarmHostUnavailable_FallsBackWithoutThrowing()
    {
        var stateDir = Path.Combine(Path.GetTempPath(), $"psm-warmhost-acquisition-{Guid.NewGuid():N}");
        Directory.CreateDirectory(stateDir);
        try
        {
            var options = new WarmHostOptions
            {
                ModelDir = "unused-in-this-test",
                StateDirectory = stateDir,
                StartupTimeout = TimeSpan.FromMilliseconds(200),
            };

            // Simulate "another process is currently spawning the daemon" with a fresh, non-stale
            // lock -- WarmHostLock.TryAcquire will see it held and not reclaim it, so
            // EnsureDaemonAsync goes into its "poll for them" branch and never spawns anything itself.
            File.WriteAllText(options.LockFilePath, DateTimeOffset.UtcNow.ToString("O"));

            var fallbackCalled = false;
            var fallbackModelDir = (string?)null;
            Func<string, CancellationToken, Task<AcquiredRuntime>> localLoader = (dir, _) =>
            {
                fallbackCalled = true;
                fallbackModelDir = dir;
                return Task.FromResult(new AcquiredRuntime
                {
                    Runtime = new FakePsmRuntime(),
                    EmbeddingRuntime = new FakeEmbeddingRuntime(),
                });
            };

            var originalError = Console.Error;
            var stderr = new StringWriter();
            Console.SetError(stderr);
            AcquiredRuntime acquired;
            try
            {
                acquired = await PsmRuntimeAcquisition.AcquireAsync(
                    options.ModelDir, options, daemonProcessArgs: null, ct: CancellationToken.None, localLoader: localLoader);
            }
            finally
            {
                Console.SetError(originalError);
            }

            Assert.True(fallbackCalled, "expected the fallback local-load path to be invoked after the warm host failed");
            Assert.Equal(options.ModelDir, fallbackModelDir);
            Assert.IsType<FakePsmRuntime>(acquired.Runtime);
            Assert.IsType<FakeEmbeddingRuntime>(acquired.EmbeddingRuntime);
            Assert.Contains("warm host unavailable", stderr.ToString(), StringComparison.OrdinalIgnoreCase);

            await acquired.DisposeAsync(); // should be a no-op (no local disposables from the fake loader) -- must not throw.
        }
        finally
        {
            Directory.Delete(stateDir, recursive: true);
        }
    }

    [Fact]
    public async Task AcquireAsync_NoWarmHostOptions_GoesStraightToLocalLoad_WithoutTryingTheDaemon()
    {
        var localLoaderCalled = false;
        Func<string, CancellationToken, Task<AcquiredRuntime>> localLoader = (dir, _) =>
        {
            localLoaderCalled = true;
            return Task.FromResult(new AcquiredRuntime { Runtime = new FakePsmRuntime(), EmbeddingRuntime = new FakeEmbeddingRuntime() });
        };

        var acquired = await PsmRuntimeAcquisition.AcquireAsync(
            "unused-model-dir", warmHostOptions: null, daemonProcessArgs: null, ct: CancellationToken.None, localLoader: localLoader);

        Assert.True(localLoaderCalled);
        Assert.IsType<FakePsmRuntime>(acquired.Runtime);
        await acquired.DisposeAsync();
    }
}
