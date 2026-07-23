using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;

namespace PsmMemory.Core;

/// <summary>
/// The single Core-level entry point for turning a raw LLM response into stored memory. Every host
/// (CLI, MCP, tests, any future integration) goes through this one class -- there is exactly one
/// processing pipeline: enqueue a durable row, split it into semantically-sensible pieces via
/// <see cref="TextSegmenter"/> (a no-op split for the common short-response case), then call the
/// existing, unmodified <see cref="PsmService.RememberAsync"/> once per piece. The only thing that
/// varies per caller is whether they wait for that processing to finish:
/// <list type="bullet">
/// <item><see cref="Enqueue"/> -- fire-and-forget: durably logs the request and returns instantly.
/// Used by callers (e.g. an MCP tool) that don't need the result.</item>
/// <item><see cref="RememberAndWaitAsync"/> -- enqueues via the exact same path, then immediately
/// drains that one row and returns the resulting decision(s). Used by callers (CLI's single-shot
/// `remember`, `serve`'s NDJSON remember command, a benchmark harness) that need a synchronous
/// result. This is composition on top of the fire-and-forget primitive, not a separate code path --
/// the request is still durably logged first, so even a "synchronous" caller's request survives a
/// crash mid-processing.</item>
/// <item><see cref="DrainOnceAsync"/> -- batch background draining, used by a continuously-polling
/// worker (e.g. an MCP host's <c>BackgroundService</c>) and by manual CLI testing.</item>
/// </list>
/// No TS equivalent; the fire-and-forget queue itself is a new addition for this port (see
/// <see cref="PendingRememberRequest"/>'s doc comment).
/// </summary>
public sealed class RememberQueueDrainer
{
    private readonly MemoryStore _store;
    private readonly PsmService _service;

    public RememberQueueDrainer(MemoryStore store, PsmService service)
    {
        _store = store;
        _service = service;
    }

    /// <summary>Durably logs a remember request and returns immediately with its pending id -- no
    /// LLM call, no chunking, just one row write. The request is processed later by whatever drains
    /// this queue (<see cref="DrainOnceAsync"/> or <see cref="RememberAndWaitAsync"/>). Static +
    /// store-only (no <see cref="PsmService"/> needed) so a caller that doesn't want to load a model
    /// at all -- e.g. a CLI command purely for queueing -- can still use the exact same mapping from
    /// <see cref="RememberRequest"/> to the stored row that every other path uses.</summary>
    public static string Enqueue(MemoryStore store, RememberRequest request) => store.InsertPendingRememberRequest(
        userId: request.UserId,
        llmResponse: request.LlmResponse,
        userMessage: request.UserMessage,
        includeExistingMemories: request.IncludeExistingMemories,
        extraTags: request.ExtraTags,
        sourceKind: request.Source?.SourceKind,
        sourceId: request.Source?.SourceId,
        sourceTimestamp: request.Source?.SourceTimestamp,
        sourceLabel: request.Source?.SourceLabel,
        domain: request.Domain.ToString());

    /// <summary>Instance convenience for <see cref="Enqueue(MemoryStore, RememberRequest)"/> using
    /// this drainer's own store.</summary>
    public string Enqueue(RememberRequest request) => Enqueue(_store, request);

    /// <summary>Enqueues (see <see cref="Enqueue(MemoryStore, RememberRequest)"/>) and immediately
    /// drains that exact row, returning the resulting decision(s) -- one per chunk if the response
    /// was long enough to split, one element in the common case. This is the synchronous-result
    /// composition atop the same durable queue every other caller uses; it does not bypass the queue
    /// or duplicate its logic.</summary>
    public async Task<List<RememberResult>> RememberAndWaitAsync(RememberRequest request, CancellationToken ct = default)
    {
        var domain = request.Domain.ToString();
        var id = Enqueue(request);

        // We already have every field the row would have if we re-read it from the DB -- no need
        // for a wasted round-trip SELECT just to reconstruct what we already know.
        var row = new PendingRememberRequest(
            id, request.UserId, request.LlmResponse, request.UserMessage,
            request.IncludeExistingMemories, request.ExtraTags,
            request.Source?.SourceKind, request.Source?.SourceId,
            request.Source?.SourceTimestamp, request.Source?.SourceLabel, domain);

        return await DrainRowAsync(row, ct).ConfigureAwait(false);
    }

    /// <summary>Drains up to <paramref name="batchSize"/> pending rows. Returns how many rows were
    /// attempted (processed or failed) -- a full batch means there's likely more backlog to drain.
    /// Each row is isolated so one bad row (a malformed response, a runtime error) never blocks the
    /// rest of the batch.</summary>
    public async Task<int> DrainOnceAsync(int batchSize = 10, CancellationToken ct = default)
    {
        var rows = _store.SelectPendingRememberRequests(batchSize);
        foreach (var row in rows)
        {
            try
            {
                await DrainRowAsync(row, ct).ConfigureAwait(false);
            }
            catch
            {
                // Already recorded via MarkPendingRememberRequestFailed inside DrainRowAsync --
                // swallow here so one bad row doesn't stop the rest of the batch.
            }
        }
        return rows.Count;
    }

    /// <summary>Processes one specific pending row right now and returns the resulting decision(s).
    /// Shared by <see cref="DrainOnceAsync"/>'s batch loop and <see cref="RememberAndWaitAsync"/>'s
    /// synchronous-wait composition -- this is the one place row processing actually happens.</summary>
    public async Task<List<RememberResult>> DrainRowAsync(PendingRememberRequest row, CancellationToken ct = default)
    {
        try
        {
            var results = await ProcessRowAsync(row, ct).ConfigureAwait(false);
            _store.MarkPendingRememberRequestProcessed(row.Id);
            return results;
        }
        catch (Exception ex)
        {
            _store.MarkPendingRememberRequestFailed(row.Id, ex.Message);
            throw;
        }
    }

    private async Task<List<RememberResult>> ProcessRowAsync(PendingRememberRequest row, CancellationToken ct)
    {
        var domain = Enum.Parse<PsmDomain>(row.Domain, ignoreCase: true);
        var segments = TextSegmenter.SegmentLlmResponse(row.LlmResponse);
        var hasSource = row.SourceKind is not null || row.SourceId is not null
            || row.SourceTimestamp is not null || row.SourceLabel is not null;

        var results = new List<RememberResult>(segments.Count);
        foreach (var segment in segments)
        {
            // Only tag a chunk-specific source id when the response actually got split into more
            // than one piece -- the common single-segment case passes the original source id
            // through completely unchanged, so today's non-chunked behavior is unaffected.
            var sourceId = segments.Count > 1 && row.SourceId is not null
                ? TextSegmenter.ChunkSourceId(row.SourceId, segment.Index)
                : row.SourceId;

            var result = await _service.RememberAsync(new RememberRequest
            {
                LlmResponse = segment.Text,
                UserMessage = row.UserMessage,
                UserId = row.UserId,
                IncludeExistingMemories = row.IncludeExistingMemories,
                ExtraTags = row.ExtraTags,
                Domain = domain,
                Source = hasSource
                    ? new MemorySourceMetadata
                    {
                        SourceKind = row.SourceKind,
                        SourceId = sourceId,
                        SourceTimestamp = row.SourceTimestamp,
                        SourceLabel = row.SourceLabel,
                    }
                    : null,
            }, ct).ConfigureAwait(false);
            results.Add(result);
        }
        return results;
    }
}
