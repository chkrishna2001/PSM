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
/// Simplifications vs. service.ts (documented once here, not per-method): no embedding-based
/// vector recall (psm-core's optional EmbeddingRuntime path), no LLM context-render step (no
/// adapter for it exists in the 3-adapter ONNX runtime — service.ts's renderContext/buildContextRenderPrompt/
/// parseContextRender have no port), no memory_facts/indexables ranking in recall results, and no
/// automatic indexable synthesis from stored content (indexables.ts's buildIndexablesForRemember).
/// Facts/indexables the model itself emits in a storage decision are still parsed, guarded, and
/// persisted via MemoryStore.ApplyDecision.
/// </summary>
public sealed class PsmService
{
    private readonly MemoryStore _store;
    private readonly IPsmRuntime _runtime;

    /// <summary>
    /// Minimum hybrid-ranking score for an existing memory to be considered "close enough" to the
    /// new storage decision's content to trigger a consolidation-adapter call. No TS equivalent —
    /// psm-core never had a consolidation step; this two-step remember flow is new for this port
    /// (task brief: "two-step consolidation" using Ranking.HybridRankMemories for the nearest-neighbor
    /// lookup). Exposed so callers can tune it without a fork.
    /// </summary>
    public double ConsolidationCandidateMinScore { get; init; } = 0.3;

    /// <summary>
    /// A final "ignore" outcome is logged to ignored_decisions for later reprocessing when its
    /// confidence is below this threshold (or has no confidence at all, e.g. parse fail-safe /
    /// guard rejection). No TS equivalent — new side table, see MemoryStore remarks.
    /// </summary>
    public double LowConfidenceIgnoreThreshold { get; init; } = 0.5;

    private const double ContextMinScore = 0.15;
    private const double RecallMinScore = 0.35;

    public PsmService(MemoryStore store, IPsmRuntime runtime)
    {
        _store = store;
        _runtime = runtime;
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

        var source = request.Source?.SourceId ?? "llm-response";
        var result = _store.ApplyDecision(request.UserId, source, decision, request.ExtraTags, conflictAgainst);

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

        var candidates = _store.SelectMemories(userId, MemoryTables.All, Math.Max(100, plan.TopK * 10));
        var ranked = Ranking.HybridRankMemories(query, candidates, new Ranking.HybridRankOptions
        {
            TopK = plan.TopK,
            PreferredTables = plan.TargetTables,
            MinScore = isContext ? ContextMinScore : RecallMinScore
        });
        _store.UpdateAccess(ranked);

        return new RecallResult
        {
            UserId = userId,
            Query = query,
            Plan = plan,
            PlanFallback = plan.PlanFallback,
            Memories = ranked
        };
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

        try
        {
            var prompt = PromptBuilder.BuildConsolidationPrompt(content, nearest.Id, nearest.Content);
            var raw = await _runtime.GenerateConsolidationDecisionAsync(prompt, domain, ct).ConfigureAwait(false);
            var consolidation = StorageDecisionParser.ParseConsolidationDecision(raw);
            if (consolidation.ParseError is not null) return (decision, null);

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
