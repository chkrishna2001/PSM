using PsmMemory.Core.Decisions;
using PsmMemory.Core.Models;
using PsmMemory.Core.Prompts;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;

namespace PsmMemory.Core;

/// <summary>
/// Ported from psm-core/src/service.ts's PsmService class (remember/recall/context orchestration),
/// adapted to the 3-adapter in-process ONNX runtime. See class-level remarks on each method for
/// what was ported faithfully vs. simplified/adapted for this port.
///
/// Embedding-based vector recall (service.ts's optional EmbeddingRuntime path -- embedWrittenMemories
/// on remember, contextCandidates on recall/context) IS ported: see <see cref="EmbedWrittenMemoriesAsync"/>
/// and <see cref="ContextCandidatesAsync"/>. It activates automatically whenever an
/// <see cref="IEmbeddingRuntime"/> is supplied to the constructor, and is entirely independent of
/// <see cref="PsmDomain"/> -- the same embedding model backs vector recall for both Coding and
/// Conversational callers, since embeddings are computed from raw memory content, never from a
/// domain-specific adapter.
///
/// Remaining simplification vs. service.ts (documented once here): no LLM context-render step (no
/// adapter for it exists in the 3-adapter runtime — service.ts's renderContext/
/// buildContextRenderPrompt/parseContextRender have no port). memory_facts/indexables ranking IS
/// ported (see <see cref="PlanAndRankAsync"/> / <see cref="Ranking.RankFacts"/> /
/// <see cref="Indexables.RankIndexables"/>), and automatic indexable synthesis from stored content
/// IS ported (<see cref="Indexables.BuildIndexablesForRemember"/>, invoked from
/// <see cref="RememberAsync"/> exactly where service.ts's remember() calls it).
/// </summary>
public sealed class PsmService
{
    private readonly MemoryStore _store;
    private readonly IPsmRuntime _runtime;
    private readonly IEmbeddingRuntime? _embeddingRuntime;
    private readonly string _embeddingModel;

    /// <summary>
    /// Minimum hybrid-ranking score for an existing memory to be considered "close enough" to the
    /// new storage decision's content to trigger a consolidation-adapter call. No TS equivalent —
    /// psm-core never had a consolidation step; this two-step remember flow is new for this port
    /// (task brief: "two-step consolidation" using Ranking.HybridRankMemories for the nearest-neighbor
    /// lookup). Exposed so callers can tune it without a fork.
    /// </summary>
    public double ConsolidationCandidateMinScore { get; init; } = 0.3;

    /// <summary>
    /// Minimum fraction of the NEW decision content's own significant tokens that must survive
    /// into a consolidation adapter's "update_existing" merge for the merge to be accepted. Found
    /// via a real probe (2026-07-20, coding-agent-cx-store-10): the consolidation adapter matched a
    /// new, distinct finding against an unrelated existing memory that merely shared boilerplate
    /// experiment-log phrasing ("documenting that... not a new mechanism"), then merged toward the
    /// OLD memory's content, silently discarding the new one's actual substance (specific ratios/
    /// numbers). GroundingGuards.IsGroundedInSource's ~10% threshold (tuned to catch pure
    /// hallucination) does NOT catch this — the shared boilerplate alone clears it. 0.5 requires the
    /// merge to retain most of the new content's own vocabulary, not just its connective phrasing.
    /// </summary>
    public double ConsolidationMergeRetentionMinRatio { get; init; } = 0.5;

    /// <summary>
    /// Minimum cosine similarity (embedding space) required between the new decision's content and
    /// the nearest lexical-match candidate before <see cref="ConsolidateAsync"/> will even call the
    /// consolidation adapter -- only enforced when an <see cref="IEmbeddingRuntime"/> is available
    /// (no-op in pure-lexical mode). Found via a real probe (2026-07-21): two completely unrelated
    /// facts about the same person ("Melanie got a cat" / "Melanie ran a charity race") scored 0.60
    /// on Ranking.HybridRankMemories -- past the 0.3 candidate threshold -- almost entirely from the
    /// shared-entity boost (Ranking.cs's entity term is tuned for query-vs-memory RETRIEVAL
    /// relevance, where "same person" is a legitimate positive signal; it is not evidence these are
    /// "the same fact" for consolidation purposes). Measured cosine similarity for that same
    /// unrelated pair was 0.35, vs. 0.81 for a genuine same-fact update ("favorite language is
    /// Python" -> "...now Rust") -- a clean separation. 0.5 sits between the two with margin on
    /// both sides.
    /// </summary>
    public double ConsolidationSemanticMinScore { get; init; } = 0.5;

    /// <summary>
    /// A final "ignore" outcome is logged to ignored_decisions for later reprocessing when its
    /// confidence is below this threshold (or has no confidence at all, e.g. parse fail-safe /
    /// guard rejection). No TS equivalent — new side table, see MemoryStore remarks.
    /// </summary>
    public double LowConfidenceIgnoreThreshold { get; init; } = 0.5;

    private const double ContextMinScore = 0.15;
    private const double RecallMinScore = 0.35;

    public PsmService(
        MemoryStore store,
        IPsmRuntime runtime,
        IEmbeddingRuntime? embeddingRuntime = null,
        string embeddingModel = LlamaSharpEmbeddingRuntime.ModelName)
    {
        _store = store;
        _runtime = runtime;
        _embeddingRuntime = embeddingRuntime;
        _embeddingModel = embeddingModel;
    }

    /// <summary>
    /// Ported from service.ts's remember(). Flow: fetch existing memories -> build storage prompt
    /// (PromptBuilder) -> runtime.GenerateStorageDecisionAsync -> parse (StorageDecisionParser),
    /// with one repair-retry attempt on parse failure and a fail-safe `ignore` if that also fails
    /// -> apply source metadata overrides -> GroundingGuards.ApplyStorageGuards -> (new) two-step
    /// consolidation against the nearest existing memory when one is close enough -> MemoryStore.ApplyDecision.
    /// Never throws on model/parse failure; a low-confidence final ignore is logged to
    /// ignored_decisions for later async reprocessing.
    /// </summary>
    public async Task<RememberResult> RememberAsync(RememberRequest request, CancellationToken ct = default)
    {
        var existing = request.IncludeExistingMemories
            ? _store.SelectMemories(request.UserId, new[] { MemoryTables.Semantic, MemoryTables.Episodic }, 50)
            : new List<MemoryRecord>();

        var storagePrompt = PromptBuilder.BuildStoragePrompt(request.LlmResponse);
        var raw = await _runtime.GenerateStorageDecisionAsync(storagePrompt, request.Domain, ct).ConfigureAwait(false);
        var decision = StorageDecisionParser.ParseStorageDecision(raw, request.LlmResponse);

        var repairAttempted = false;
        if (decision.ParseError is not null)
        {
            repairAttempted = true;
            var repairPrompt = PromptBuilder.BuildStorageRepairPrompt(request.LlmResponse, decision.RawJson);
            var repairedRaw = await _runtime.GenerateStorageDecisionAsync(repairPrompt, request.Domain, ct).ConfigureAwait(false);
            decision = StorageDecisionParser.ParseStorageDecision(repairedRaw, request.LlmResponse);
        }

        if (decision.ParseError is not null)
        {
            var failsafe = StorageDecisionParser.FailsafeDecision(decision.RawJson, decision.ParseError);
            LogIfLowConfidenceIgnore(request, "parse_error_noop", failsafe, null);
            return new RememberResult
            {
                UserId = request.UserId,
                Action = Actions.Kinds.Ignore,
                Route = "parse_error_noop",
                Written = new List<string>(),
                Memory = null,
                Reasoning = failsafe.Reasoning,
                RawModelJson = failsafe.RawJson,
                RepairAttempted = repairAttempted,
                ParseError = decision.ParseError
            };
        }

        ApplySourceOverrides(decision, request.Source);

        // Ported from service.ts's remember() call site: normalize relative-time expressions
        // ("yesterday", "last month", ...) detected in the stored content/facts into resolved,
        // rankable date strings before the grounding guards / consolidation / persistence steps.
        var temporalAnchor = decision.Memory?.SourceTimestamp ?? request.Source?.SourceTimestamp;
        if (decision.Memory is not null)
        {
            TemporalNormalizer.NormalizeMemoryTemporalFields(decision.Memory, temporalAnchor);
        }
        foreach (var fact in decision.Facts)
        {
            TemporalNormalizer.NormalizeFactTemporalFields(fact, temporalAnchor);
        }

        var guarded = GroundingGuards.ApplyStorageGuards(request.LlmResponse, decision);
        if (guarded.Rejected)
        {
            LogIfLowConfidenceIgnore(request, guarded.GuardRoute ?? "grounding_reject", decision, guarded.GuardReason);
            return new RememberResult
            {
                UserId = request.UserId,
                Action = Actions.Kinds.Ignore,
                Route = guarded.GuardRoute ?? "grounding_reject",
                Written = new List<string>(),
                Memory = null,
                Reasoning = guarded.GuardReason ?? "Storage guard rejected ungrounded output.",
                RawModelJson = decision.RawJson,
                RepairAttempted = repairAttempted,
                ParseError = decision.ParseError,
                GuardRejected = true
            };
        }

        (string Id, string Table)? conflictAgainst = null;
        if (WouldStoreDecision(decision))
        {
            (decision, conflictAgainst) = await ConsolidateAsync(decision, existing, request.Domain, ct).ConfigureAwait(false);
        }

        // Ported from service.ts's remember() call site (service.ts:176-186): synthesize
        // indexables deterministically (no LLM call) when this decision will actually write a
        // memory and the model didn't already emit its own indexables.
        if (WouldStoreDecision(decision) && decision.Indexables.Count == 0)
        {
            var indexableTags = new List<string>();
            if (decision.Memory?.Tags is not null) indexableTags.AddRange(decision.Memory.Tags);
            if (request.ExtraTags is not null) indexableTags.AddRange(request.ExtraTags);

            decision.Indexables = Indexables.BuildIndexablesForRemember(new Indexables.BuildIndexablesInput
            {
                LlmResponse = request.LlmResponse,
                MemoryContent = decision.Memory?.Content?.Trim() ?? request.LlmResponse,
                Tags = indexableTags,
                Facts = decision.Facts
            });
        }

        var source = request.Source?.SourceId ?? "llm-response";
        var result = _store.ApplyDecision(request.UserId, source, decision, request.ExtraTags, conflictAgainst);
        await EmbedWrittenMemoriesAsync(request.UserId, result.MemoryRefs, ct).ConfigureAwait(false);

        LogIfLowConfidenceIgnore(request, result.Route, decision, null, result);

        return new RememberResult
        {
            UserId = request.UserId,
            Action = result.Action,
            Route = result.Route,
            Written = result.Written,
            Memory = decision.Memory,
            Reasoning = decision.Reasoning,
            RawModelJson = decision.RawJson,
            RepairAttempted = repairAttempted,
            ParseError = decision.ParseError,
            Indexables = decision.Indexables
        };
    }

    /// <summary>
    /// Ported from service.ts's recall(): runtime.GenerateRecallPlanAsync is the primary path;
    /// on any exception it falls back to DeterministicPlanFallback (never lets a runtime failure
    /// propagate to the caller). Existing memories are then ranked via Ranking.HybridRankMemories.
    /// </summary>
    public Task<RecallResult> RecallAsync(RecallRequest request, CancellationToken ct = default) =>
        PlanAndRankAsync(request.Question, request.UserId, request.TopK ?? 5, isContext: false, request.Domain, ct);

    /// <summary>
    /// Ported from service.ts's context(): same recall-plan-first-then-rank flow as RecallAsync,
    /// using the context-plan prompt and a lower minimum relevance score (matching service.ts's
    /// contextMinScore=0.15 vs recallMinScore=0.35). No LLM context-render step — see class remarks.
    /// </summary>
    public Task<RecallResult> ContextAsync(ContextRequest request, CancellationToken ct = default) =>
        PlanAndRankAsync(request.Prompt, request.UserId, request.TopK ?? 5, isContext: true, request.Domain, ct);

    private async Task<RecallResult> PlanAndRankAsync(string query, string userId, int topK, bool isContext, PsmDomain domain, CancellationToken ct)
    {
        var prompt = isContext
            ? PromptBuilder.BuildContextPlanPrompt(query, topK)
            : PromptBuilder.BuildRecallPlanPrompt(query, topK);

        RecallPlan plan;
        try
        {
            var raw = await _runtime.GenerateRecallPlanAsync(prompt, domain, ct).ConfigureAwait(false);
            plan = StorageDecisionParser.ParseRecallPlan(raw, query, topK);
        }
        catch
        {
            plan = DeterministicPlanFallback.BuildPlan(query, topK);
        }

        var limit = Math.Max(100, plan.TopK * 10);
        var (candidates, vectorScores) = await ContextCandidatesAsync(userId, query, limit, ct).ConfigureAwait(false);
        var ranked = Ranking.HybridRankMemories(query, candidates, new Ranking.HybridRankOptions
        {
            TopK = plan.TopK,
            VectorScores = vectorScores,
            PreferredTables = plan.TargetTables,
            MinScore = isContext ? ContextMinScore : RecallMinScore
        });
        _store.UpdateAccess(ranked);

        // Ported from service.ts's recall()/context(): both rank memory_facts (rankFacts), but
        // only recall() ranks indexables/workflows (rankIndexables) -- context() never surfaces
        // them in the original TS source.
        var facts = _store.SelectMemoryFacts(userId, limit);
        var rankedFacts = Ranking.RankFacts(query, facts, plan.TopK);

        var rankedIndexables = new List<Indexables.ScoredIndexable>();
        if (!isContext)
        {
            var indexableRows = _store.SelectIndexables(userId, 100);
            rankedIndexables = Indexables.RankIndexables(query, indexableRows, plan.TopK);
        }

        return new RecallResult
        {
            UserId = userId,
            Query = query,
            Plan = plan,
            PlanFallback = plan.PlanFallback,
            Memories = ranked,
            Facts = rankedFacts,
            Indexables = rankedIndexables,
            Workflows = rankedIndexables.Where(row => row.Kind == IndexableKinds.Workflow).ToList()
        };
    }

    /// <summary>
    /// Ported from service.ts's <c>embedWrittenMemories</c>: embeds and stores one embedding per
    /// memory actually written by this remember() call. No-op when no <see cref="IEmbeddingRuntime"/>
    /// was supplied (pure-lexical mode, the pre-fix default). Runs for every <see cref="PsmDomain"/>
    /// identically -- embedding a memory's content never depends on which domain's adapter produced
    /// the storage decision.
    /// </summary>
    private async Task EmbedWrittenMemoriesAsync(string userId, List<WrittenMemoryRef> refs, CancellationToken ct)
    {
        if (_embeddingRuntime is null) return;
        foreach (var reference in refs)
        {
            var embedding = await _embeddingRuntime.EmbedAsync(reference.Content, ct).ConfigureAwait(false);
            _store.UpsertMemoryEmbedding(reference, userId, _embeddingModel, embedding);
        }
    }

    /// <summary>
    /// Ported from service.ts's <c>contextCandidates</c>: without an <see cref="IEmbeddingRuntime"/>,
    /// falls back to the pre-fix pure-lexical behavior but widens the candidate pool to at least
    /// 10,000 rows (psm-core's "ponytail" comment: a 200-row cap silently misses LoCoMo-scale
    /// evidence when there's no vector search to fall back on). With an embedding runtime, embeds
    /// the query, scores every stored embedding for this user via cosine similarity, fetches the
    /// top-<paramref name="limit"/> memories by vector score, and merges them with the lexical
    /// candidate set (lexical-first, deduped by table:id) so <see cref="Ranking.HybridRankMemories"/>
    /// can blend both signals. Identical behavior regardless of <see cref="PsmDomain"/>.
    /// </summary>
    private async Task<(List<MemoryRecord> Candidates, Dictionary<string, double>? VectorScores)> ContextCandidatesAsync(
        string userId, string query, int limit, CancellationToken ct)
    {
        var lexicalLimit = _embeddingRuntime is not null ? limit : Math.Max(limit, 10_000);
        var lexicalCandidates = _store.SelectMemories(userId, MemoryTables.All, lexicalLimit);
        if (_embeddingRuntime is null) return (lexicalCandidates, null);

        var queryEmbedding = await _embeddingRuntime.EmbedAsync(query, ct).ConfigureAwait(false);
        var scored = _store.SelectEmbeddingRows(userId, _embeddingModel)
            .Select(row => (row.MemoryTable, row.MemoryId, Score: CosineSimilarity(queryEmbedding, row.Embedding)))
            .OrderByDescending(row => row.Score)
            .Take(limit)
            .ToList();

        var vectorScores = new Dictionary<string, double>();
        var vectorMemories = new List<MemoryRecord>();
        foreach (var row in scored)
        {
            vectorScores[$"{row.MemoryTable}:{row.MemoryId}"] = row.Score;
            var memory = _store.GetMemory(row.MemoryTable, row.MemoryId);
            if (memory is not null) vectorMemories.Add(memory);
        }

        return (MergeMemories(lexicalCandidates, vectorMemories), vectorScores);
    }

    private static List<MemoryRecord> MergeMemories(params IEnumerable<MemoryRecord>[] groups)
    {
        var seen = new HashSet<string>();
        var result = new List<MemoryRecord>();
        foreach (var group in groups)
        {
            foreach (var memory in group)
            {
                var key = $"{memory.Table}:{memory.Id}";
                if (!seen.Add(key)) continue;
                result.Add(memory);
            }
        }
        return result;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        var length = Math.Min(a.Length, b.Length);
        if (length == 0) return 0;
        double dot = 0, aNorm = 0, bNorm = 0;
        for (var i = 0; i < length; i++)
        {
            dot += (double)a[i] * b[i];
            aNorm += (double)a[i] * a[i];
            bNorm += (double)b[i] * b[i];
        }
        if (aNorm == 0 || bNorm == 0) return 0;
        return dot / (Math.Sqrt(aNorm) * Math.Sqrt(bNorm));
    }

    /// <summary>
    /// New two-step consolidation (no TS equivalent): finds the nearest existing memory to the new
    /// decision's content; if it's close enough, asks the consolidation adapter whether to store
    /// independently, merge (update_existing), or flag a conflict, and folds that verdict back into
    /// the storage decision. On any consolidation failure (parse error or runtime exception), the
    /// original storage decision is kept as-is — consolidation is a best-effort enrichment step, never
    /// a blocker.
    /// </summary>
    private async Task<(StorageDecision Decision, (string Id, string Table)? ConflictAgainst)> ConsolidateAsync(
        StorageDecision decision, List<MemoryRecord> existing, PsmDomain domain, CancellationToken ct)
    {
        var content = decision.Memory?.Content?.Trim();
        if (string.IsNullOrEmpty(content) || existing.Count == 0) return (decision, null);

        var nearest = Ranking.HybridRankMemories(content, existing, new Ranking.HybridRankOptions { TopK = 1 }).FirstOrDefault();
        if (nearest is null || nearest.Score < ConsolidationCandidateMinScore) return (decision, null);

        if (_embeddingRuntime is not null)
        {
            var newEmbedding = await _embeddingRuntime.EmbedAsync(content, ct).ConfigureAwait(false);
            var nearestEmbedding = await _embeddingRuntime.EmbedAsync(nearest.Content, ct).ConfigureAwait(false);
            if (CosineSimilarity(newEmbedding, nearestEmbedding) < ConsolidationSemanticMinScore)
            {
                // Lexically similar (shared entity/table/confidence terms) but not semantically --
                // not a genuine consolidation candidate, just two unrelated facts about the same
                // subject. See ConsolidationSemanticMinScore remarks for the measured evidence.
                return (decision, null);
            }
        }

        try
        {
            var prompt = PromptBuilder.BuildConsolidationPrompt(content, nearest.Id, nearest.Content);
            var raw = await _runtime.GenerateConsolidationDecisionAsync(prompt, domain, ct).ConfigureAwait(false);
            var consolidation = StorageDecisionParser.ParseConsolidationDecision(raw);
            if (consolidation.ParseError is not null) return (decision, null);

            if (consolidation.Action == Actions.Kinds.UpdateExisting
                && !string.IsNullOrWhiteSpace(consolidation.MergedContent)
                && !MergeRetainsNewContent(content, consolidation.MergedContent))
            {
                // The merge would discard the new content's own substance (see
                // ConsolidationMergeRetentionMinRatio remarks) -- treat this exactly like "no
                // consolidation candidate was close enough" rather than silently losing the new
                // information, instead of applying an untrustworthy merge.
                return (decision, null);
            }

            var updated = decision.Clone();
            updated.Action = consolidation.Action;

            if (consolidation.Action == Actions.Kinds.UpdateExisting
                && !string.IsNullOrWhiteSpace(consolidation.MergedContent)
                && updated.Memory is not null)
            {
                updated.Memory.Content = consolidation.MergedContent;
            }

            (string Id, string Table)? conflictAgainst = consolidation.Action == Actions.Kinds.FlagConflict
                ? (nearest.Id, nearest.Table)
                : null;

            return (updated, conflictAgainst);
        }
        catch
        {
            return (decision, null);
        }
    }

    /// <summary>
    /// True if <paramref name="mergedContent"/> retains at least <see cref="ConsolidationMergeRetentionMinRatio"/>
    /// of <paramref name="newContent"/>'s own significant tokens. Deliberately stricter than
    /// <see cref="GroundingGuards.IsGroundedInSource"/> (which only requires ~10% overlap, or 1-2
    /// tokens) -- that threshold is tuned to catch pure hallucination, and passes right through a
    /// merge that keeps only boilerplate connective phrasing shared between two topically-unrelated
    /// memories while discarding the new content's actual specifics (see
    /// ConsolidationMergeRetentionMinRatio remarks for the concrete case this was found from).
    /// </summary>
    private bool MergeRetainsNewContent(string newContent, string mergedContent)
    {
        var newTokens = GroundingGuards.SignificantTokens(newContent);
        if (newTokens.Count == 0) return true;
        var mergedSet = new HashSet<string>(GroundingGuards.SignificantTokens(mergedContent));
        var overlap = newTokens.Count(t => mergedSet.Contains(t));
        return overlap >= Math.Ceiling(newTokens.Count * ConsolidationMergeRetentionMinRatio);
    }

    private static void ApplySourceOverrides(StorageDecision decision, MemorySourceMetadata? source)
    {
        if (decision.Memory is null || source is null) return;
        decision.Memory.SourceKind = source.SourceKind ?? decision.Memory.SourceKind;
        decision.Memory.SourceId = source.SourceId ?? decision.Memory.SourceId;
        decision.Memory.SourceTimestamp = source.SourceTimestamp ?? decision.Memory.SourceTimestamp;
        decision.Memory.SourceLabel = source.SourceLabel ?? decision.Memory.SourceLabel;
    }

    private static bool WouldStoreDecision(StorageDecision decision)
    {
        var route = Actions.RouteForAction(decision.Action);
        if (route is Actions.Routes.Ignore or Actions.Routes.RecallOnly) return false;
        return !string.IsNullOrWhiteSpace(decision.Memory?.Content);
    }

    private void LogIfLowConfidenceIgnore(
        RememberRequest request,
        string route,
        StorageDecision decision,
        string? guardReason,
        ApplyDecisionResult? applied = null)
    {
        var finalAction = applied?.Action ?? Actions.Kinds.Ignore;
        var wroteNothing = applied is null || applied.Written.Count == 0;
        if (finalAction != Actions.Kinds.Ignore || !wroteNothing) return;
        if (decision.Confidence.HasValue && decision.Confidence.Value >= LowConfidenceIgnoreThreshold) return;

        _store.InsertIgnoredDecision(
            request.UserId,
            request.Source?.SourceId,
            request.LlmResponse,
            guardReason ?? decision.Reasoning,
            decision.Confidence,
            decision.RawJson);
    }
}
