using System.Net;
using System.Net.Sockets;
using System.Text.Json;

namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// The warm host itself: loads one <see cref="LlamaSharpPsmRuntime"/> + one
/// <see cref="LlamaSharpEmbeddingRuntime"/> ONCE, then serves them over loopback HTTP for the
/// lifetime of this process (bounded by a sliding idle-expiration timeout, or by cancelling
/// <paramref name="ct"/> passed to <see cref="RunAsync"/>). Every short-lived caller (a CLI one-shot
/// command, especially `hook recall`) reaches this instead of paying the GGUF-load cost itself --
/// see <see cref="WarmHostClient"/>/<see cref="PsmRuntimeAcquisition"/> for that side.
///
/// Ported from src/psm-cli/src/daemon.ts's <c>GET /health</c> / <c>POST /v1</c> daemon, but over a
/// plain <see cref="HttpListener"/> (this project deliberately has no ASP.NET Core/Kestrel
/// dependency) instead of Node's http server.
/// </summary>
public static class WarmHostServer
{
    public static async Task RunAsync(WarmHostOptions options, CancellationToken ct)
    {
        Directory.CreateDirectory(options.StateDirectory);

        var port = GetFreePort(options.Host);

        using var psmRuntime = await LlamaSharpPsmRuntime.CreateAsync(options.ModelDir, options.HfRepoId, ct: ct)
            .ConfigureAwait(false);
        using var embeddingRuntime = await LlamaSharpEmbeddingRuntime.CreateAsync(options.ModelDir, options.HfRepoId, ct: ct)
            .ConfigureAwait(false);

        var listener = new HttpListener();
        listener.Prefixes.Add($"http://{options.Host}:{port}/");
        listener.Start();

        var now = DateTimeOffset.UtcNow;
        var state = new DaemonState
        {
            Pid = Environment.ProcessId,
            Host = options.Host,
            Port = port,
            StartedAt = now,
            LastSeenAt = now,
        };
        WarmHostState.Write(options.StateFilePath, state);

        using var stopCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        using var ctReg = ct.Register(() => TryStop(listener));

        var idleTask = IdleWatchAsync(options, listener, state, stopCts);

        try
        {
            await ServeLoopAsync(listener, options, state, psmRuntime, embeddingRuntime, stopCts.Token).ConfigureAwait(false);
        }
        finally
        {
            stopCts.Cancel();
            TryStop(listener);
            try { listener.Close(); } catch { /* already closed */ }
            try { File.Delete(options.StateFilePath); } catch { /* best effort */ }
            await idleTask.ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Binds a real <see cref="TcpListener"/> on port 0 to let the OS hand back a free ephemeral port,
    /// then immediately releases it so <see cref="HttpListener"/> can bind the same port. Accepted
    /// small race: another process could bind this exact port in the gap between releasing it here and
    /// <see cref="HttpListener.Start"/> below -- fine for a loopback-only, single local warm-host use
    /// case; not worth a bind-retry loop.
    /// </summary>
    private static int GetFreePort(string host)
    {
        var address = host == "127.0.0.1" || host == "localhost" ? IPAddress.Loopback : IPAddress.Parse(host);
        var probe = new TcpListener(address, 0);
        probe.Start();
        var port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        return port;
    }

    private static async Task ServeLoopAsync(
        HttpListener listener,
        WarmHostOptions options,
        DaemonState state,
        IPsmRuntime psmRuntime,
        IEmbeddingRuntime embeddingRuntime,
        CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            HttpListenerContext context;
            try
            {
                context = await listener.GetContextAsync().ConfigureAwait(false);
            }
            catch (Exception) when (!listener.IsListening || ct.IsCancellationRequested)
            {
                break; // listener was stopped (idle timeout or external cancellation) -- exit cleanly.
            }

            // Every request -- including /health -- refreshes LastSeenAt before it's handled; this
            // rewrite IS the sliding-expiration mechanism (mirrors daemon.ts's daemon.json rewrite).
            state.LastSeenAt = DateTimeOffset.UtcNow;
            WarmHostState.Write(options.StateFilePath, state);

            // Handled sequentially, not fire-and-forget: both LlamaSharpPsmRuntime and
            // LlamaSharpEmbeddingRuntime already self-serialize concurrent calls via their own
            // SemaphoreSlim(1,1), so there is no throughput to gain from overlapping requests here,
            // and serializing avoids concurrent writers racing on daemon.json's temp-file rename.
            await HandleRequestAsync(context, psmRuntime, embeddingRuntime, ct).ConfigureAwait(false);
        }
    }

    private static async Task HandleRequestAsync(
        HttpListenerContext context, IPsmRuntime psmRuntime, IEmbeddingRuntime embeddingRuntime, CancellationToken ct)
    {
        var request = context.Request;
        var response = context.Response;
        try
        {
            if (request.HttpMethod == "GET" && request.Url?.AbsolutePath == "/health")
            {
                await WriteJsonAsync(response, 200, new { ok = true, pid = Environment.ProcessId }).ConfigureAwait(false);
                return;
            }

            if (request.HttpMethod == "POST" && request.Url?.AbsolutePath == "/v1")
            {
                WarmHostRequest? body;
                try
                {
                    body = await JsonSerializer.DeserializeAsync<WarmHostRequest>(request.InputStream, WarmHostJson.Options, ct)
                        .ConfigureAwait(false);
                }
                catch (JsonException)
                {
                    await WriteJsonAsync(response, 400, new WarmHostResponse { Ok = false, Error = "invalid_request_body" })
                        .ConfigureAwait(false);
                    return;
                }

                if (body is null)
                {
                    await WriteJsonAsync(response, 400, new WarmHostResponse { Ok = false, Error = "empty_request_body" })
                        .ConfigureAwait(false);
                    return;
                }

                WarmHostResponse result;
                try
                {
                    result = await DispatchAsync(body, psmRuntime, embeddingRuntime, ct).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    // Never crash the server on a single bad request.
                    await WriteJsonAsync(response, 500, new WarmHostResponse { Ok = false, Error = ex.Message }).ConfigureAwait(false);
                    return;
                }

                await WriteJsonAsync(response, result.Ok ? 200 : 400, result).ConfigureAwait(false);
                return;
            }

            await WriteJsonAsync(response, 404, new WarmHostResponse { Ok = false, Error = "not_found" }).ConfigureAwait(false);
        }
        finally
        {
            try { response.Close(); } catch { /* client may have already disconnected */ }
        }
    }

    private static async Task<WarmHostResponse> DispatchAsync(
        WarmHostRequest request, IPsmRuntime psmRuntime, IEmbeddingRuntime embeddingRuntime, CancellationToken ct)
    {
        switch (request.Operation)
        {
            case "storage-decision":
            case "recall-plan":
            case "consolidation-decision":
                if (string.IsNullOrEmpty(request.Prompt))
                    return new WarmHostResponse { Ok = false, Error = "missing_prompt" };

                var domain = ParseDomain(request.Domain);
                var result = request.Operation switch
                {
                    "storage-decision" => await psmRuntime.GenerateStorageDecisionAsync(request.Prompt, domain, ct).ConfigureAwait(false),
                    "recall-plan" => await psmRuntime.GenerateRecallPlanAsync(request.Prompt, domain, ct).ConfigureAwait(false),
                    _ => await psmRuntime.GenerateConsolidationDecisionAsync(request.Prompt, domain, ct).ConfigureAwait(false),
                };
                return new WarmHostResponse { Ok = true, Result = result };

            case "embed":
                if (string.IsNullOrEmpty(request.Text))
                    return new WarmHostResponse { Ok = false, Error = "missing_text" };

                var embedding = await embeddingRuntime.EmbedAsync(request.Text, ct).ConfigureAwait(false);
                return new WarmHostResponse { Ok = true, Embedding = embedding };

            default:
                return new WarmHostResponse { Ok = false, Error = "unsupported_operation" };
        }
    }

    private static PsmDomain ParseDomain(string domain) =>
        Enum.TryParse<PsmDomain>(domain, ignoreCase: true, out var parsed)
            ? parsed
            : throw new ArgumentException($"Unknown domain '{domain}' (expected 'coding' or 'conversational').", nameof(domain));

    private static async Task WriteJsonAsync(HttpListenerResponse response, int statusCode, object payload)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(payload, WarmHostJson.Options);
        response.StatusCode = statusCode;
        response.ContentType = "application/json";
        response.ContentLength64 = bytes.Length;
        await response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
    }

    private static async Task IdleWatchAsync(WarmHostOptions options, HttpListener listener, DaemonState state, CancellationTokenSource stopCts)
    {
        var pollSeconds = Math.Clamp(options.IdleTimeout.TotalSeconds / 4, 1, 60);
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(pollSeconds));
        try
        {
            while (await timer.WaitForNextTickAsync(stopCts.Token).ConfigureAwait(false))
            {
                if (DateTimeOffset.UtcNow - state.LastSeenAt >= options.IdleTimeout)
                {
                    TryStop(listener);
                    stopCts.Cancel();
                    return;
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown path (external ct cancelled, or the serve loop itself stopped us).
        }
    }

    private static void TryStop(HttpListener listener)
    {
        try { listener.Stop(); } catch { /* already stopped/closed */ }
    }
}
