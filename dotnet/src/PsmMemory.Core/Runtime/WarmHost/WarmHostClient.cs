using System.Diagnostics;
using System.Net.Http.Json;

namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Client side of the warm-host mechanism: <see cref="EnsureDaemonAsync"/> gets a short-lived caller
/// (a CLI one-shot command) a working base <see cref="Uri"/> for an already-warm
/// <see cref="WarmHostServer"/>, spawning one if none is running yet; <see cref="PostAsync"/> sends one
/// request/response round trip to it. Ported from src/psm-cli/src/daemon.ts's <c>ensureDaemon()</c>,
/// but with a real cross-process lock (<see cref="WarmHostLock"/>) around the
/// check-health/decide-to-spawn/spawn sequence -- the TS original has no such lock, so two
/// near-simultaneous callers can both spawn a daemon there.
/// </summary>
public static class WarmHostClient
{
    private static readonly HttpClient DefaultHttp = new();

    private static readonly TimeSpan HealthCheckTimeout = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(250);

    /// <summary>
    /// Returns the base <see cref="Uri"/> of a healthy warm host, spawning one (via
    /// <paramref name="daemonProcessArgs"/>) if none is currently running. <paramref name="httpClient"/>
    /// is a test seam for injecting a fake-handler-backed client; production callers should omit it.
    /// </summary>
    public static async Task<Uri> EnsureDaemonAsync(
        WarmHostOptions options, string[] daemonProcessArgs, CancellationToken ct = default, HttpClient? httpClient = null)
    {
        var http = httpClient ?? DefaultHttp;

        // Common warm-path case: an already-healthy daemon is running -- never touches the lock.
        var existing = WarmHostState.Read(options.StateFilePath);
        if (existing is not null)
        {
            var existingAddress = BuildUri(existing.Host, existing.Port);
            if (await IsHealthyAsync(http, existingAddress, ct).ConfigureAwait(false))
            {
                return existingAddress;
            }
        }

        Directory.CreateDirectory(options.StateDirectory);
        using var lockHandle = WarmHostLock.TryAcquire(options.LockFilePath, options.LockStaleThreshold);
        if (lockHandle is null)
        {
            // Someone else is spawning right now (and their lock isn't stale) -- poll for THEM to
            // finish instead of racing a second spawn.
            return await PollUntilHealthyAsync(options, http, ct).ConfigureAwait(false);
        }

        SpawnDetached(daemonProcessArgs);
        return await PollUntilHealthyAsync(options, http, ct).ConfigureAwait(false);
    }

    private static async Task<Uri> PollUntilHealthyAsync(WarmHostOptions options, HttpClient http, CancellationToken ct)
    {
        var deadline = DateTimeOffset.UtcNow + options.StartupTimeout;
        while (DateTimeOffset.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            var state = WarmHostState.Read(options.StateFilePath);
            if (state is not null)
            {
                var baseAddress = BuildUri(state.Host, state.Port);
                if (await IsHealthyAsync(http, baseAddress, ct).ConfigureAwait(false))
                {
                    return baseAddress;
                }
            }
            await Task.Delay(PollInterval, ct).ConfigureAwait(false);
        }

        throw new TimeoutException($"PSM warm host did not become healthy within {options.StartupTimeout}.");
    }

    private static Uri BuildUri(string host, int port) => new($"http://{host}:{port}/");

    private static async Task<bool> IsHealthyAsync(HttpClient http, Uri baseAddress, CancellationToken ct)
    {
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(HealthCheckTimeout);
            using var response = await http.GetAsync(new Uri(baseAddress, "health"), cts.Token).ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            // Connection refused, timeout, DNS failure, etc. all just mean "not healthy yet".
            return false;
        }
    }

    /// <summary>
    /// Re-launches the CURRENT executable with <paramref name="daemonProcessArgs"/> (e.g.
    /// <c>["daemon-run", "--model-dir", modelDir, "--state-dir", stateDir]</c>, supplied by the CLI
    /// project so this Core class doesn't need to know about `daemon-run` as a concept) and does not
    /// wait for it to exit, so it outlives this process.
    ///
    /// NOTE for a human to manually verify: .NET has no direct Node-style `detached + unref()`
    /// primitive. Not calling <see cref="Process.WaitForExit()"/> and not redirecting stdio is the
    /// practical equivalent here -- the child's lifetime is not tied to this process's -- but real
    /// detached-process semantics (surviving parent-process-group teardown on Windows, console
    /// ownership, etc.) are platform-specific enough that this needs to actually be run and observed
    /// by a human, not just reasoned about in code review.
    /// </summary>
    private static void SpawnDetached(string[] daemonProcessArgs)
    {
        var exePath = Environment.ProcessPath ?? Process.GetCurrentProcess().MainModule?.FileName
            ?? throw new InvalidOperationException("Could not determine the current executable path to spawn the warm host.");

        var startInfo = new ProcessStartInfo
        {
            FileName = exePath,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = false,
            RedirectStandardError = false,
            RedirectStandardInput = false,
        };
        foreach (var arg in daemonProcessArgs) startInfo.ArgumentList.Add(arg);

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException($"Failed to start warm host process: {exePath}");
        // Deliberately no WaitForExit()/no awaiting -- see the doc comment above.
    }

    /// <summary>Sends one request/response round trip to an already-known-healthy warm host.
    /// <paramref name="httpClient"/> is a test seam (see <see cref="EnsureDaemonAsync"/>).</summary>
    public static async Task<WarmHostResponse> PostAsync(
        Uri baseAddress, WarmHostRequest request, CancellationToken ct = default, HttpClient? httpClient = null)
    {
        var http = httpClient ?? DefaultHttp;
        using var response = await http.PostAsJsonAsync(new Uri(baseAddress, "v1"), request, WarmHostJson.Options, ct)
            .ConfigureAwait(false);
        var result = await response.Content.ReadFromJsonAsync<WarmHostResponse>(WarmHostJson.Options, ct).ConfigureAwait(false);
        if (result is null)
            throw new InvalidOperationException("PSM warm host returned an empty response body.");
        if (!result.Ok)
            throw new InvalidOperationException($"PSM warm host request failed: {result.Error}");
        return result;
    }
}
