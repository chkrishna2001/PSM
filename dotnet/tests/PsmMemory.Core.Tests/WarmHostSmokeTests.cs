using PsmMemory.Core.Runtime;
using PsmMemory.Core.Runtime.WarmHost;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Integration smoke tests: starts a real <see cref="WarmHostServer"/> against the real GGUF model at
/// psm-model/prod-memory/gguf-runtime/v1 (see <see cref="LlamaSharpSmokeTests.FindRepoRoot"/> for the
/// same repo-root-discovery approach) and drives it via <see cref="WarmHostPsmRuntime"/> over real
/// loopback HTTP. Skipped by default -- matches <see cref="LlamaSharpSmokeTests"/>'s convention -- run
/// manually to validate the warm-host end-to-end.
/// </summary>
public class WarmHostSmokeTests
{
    private static string RepoRoot() => LlamaSharpSmokeTests.FindRepoRoot();
    private static string ModelDir() => Path.Combine(RepoRoot(), "psm-model", "prod-memory", "gguf-runtime", "v1");

    [Fact(Skip = "manual -- requires the real GGUF model on disk; run manually to validate the warm-host end-to-end")]
    public async Task WarmHostServer_ServesARealGenerateCall_ThroughTheProxyRuntime()
    {
        var modelDir = ModelDir();
        Assert.True(Directory.Exists(modelDir), $"expected model directory at {modelDir}");

        var stateDir = Path.Combine(Path.GetTempPath(), $"psm-warmhost-smoke-{Guid.NewGuid():N}");
        Directory.CreateDirectory(stateDir);
        try
        {
            var options = new WarmHostOptions
            {
                ModelDir = modelDir,
                StateDirectory = stateDir,
                IdleTimeout = TimeSpan.FromMinutes(15), // long -- this test drives the server directly, not via idle timeout.
            };

            using var serverCts = new CancellationTokenSource();
            var serverTask = WarmHostServer.RunAsync(options, serverCts.Token);

            // Poll daemon.json/health the same way WarmHostClient.EnsureDaemonAsync does, since we're
            // driving the server directly here rather than going through EnsureDaemonAsync's spawn path.
            DaemonState? state = null;
            var deadline = DateTimeOffset.UtcNow + TimeSpan.FromSeconds(30);
            while (DateTimeOffset.UtcNow < deadline)
            {
                state = WarmHostState.Read(options.StateFilePath);
                if (state is not null) break;
                await Task.Delay(100);
            }
            Assert.NotNull(state);

            var baseAddress = new Uri($"http://{state!.Host}:{state.Port}/");
            var runtime = new WarmHostPsmRuntime(baseAddress);

            var result = await runtime.GenerateStorageDecisionAsync(
                "The user's favorite editor is Neovim.", PsmDomain.Coding);

            Assert.False(string.IsNullOrWhiteSpace(result));
        }
        finally
        {
            // Best-effort cleanup regardless of assertion outcome above.
        }
    }

    [Fact(Skip = "manual -- requires the real GGUF model on disk; run manually to validate the warm-host end-to-end")]
    public async Task WarmHostServer_ShutsItselfDown_AfterIdleTimeoutElapses()
    {
        var modelDir = ModelDir();
        Assert.True(Directory.Exists(modelDir), $"expected model directory at {modelDir}");

        var stateDir = Path.Combine(Path.GetTempPath(), $"psm-warmhost-idle-smoke-{Guid.NewGuid():N}");
        Directory.CreateDirectory(stateDir);
        try
        {
            var options = new WarmHostOptions
            {
                ModelDir = modelDir,
                StateDirectory = stateDir,
                IdleTimeout = TimeSpan.FromSeconds(2), // short, so this test doesn't have to wait long.
            };

            using var serverCts = new CancellationTokenSource();
            var serverTask = WarmHostServer.RunAsync(options, serverCts.Token);

            // Wait for it to come up.
            var upDeadline = DateTimeOffset.UtcNow + TimeSpan.FromSeconds(30);
            while (DateTimeOffset.UtcNow < upDeadline && WarmHostState.Read(options.StateFilePath) is null)
            {
                await Task.Delay(100);
            }
            Assert.NotNull(WarmHostState.Read(options.StateFilePath));

            // Now wait for it to shut itself down (well past the 2s idle timeout, no requests sent).
            var completed = await Task.WhenAny(serverTask, Task.Delay(TimeSpan.FromSeconds(30)));
            Assert.Same(serverTask, completed);
            Assert.Null(WarmHostState.Read(options.StateFilePath)); // state file cleaned up on shutdown.
        }
        finally
        {
            // Best-effort cleanup regardless of assertion outcome above.
        }
    }
}
