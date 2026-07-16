using PsmMemory.Core.Models;

namespace PsmMemory.Core.Runtime;

/// <summary>
/// Ported from psm-core/src/deterministic-plan-runtime.ts's recall/context-plan stub (the
/// `operation === "context_plan" || operation === "recall_plan"` branch of
/// DeterministicPlanRuntime.generateJson). Used ONLY as a fallback when the real
/// retrieval_plan adapter call throws — never as a primary path, per PsmService's remember/recall
/// orchestration.
/// </summary>
public static class DeterministicPlanFallback
{
    public static RecallPlan BuildPlan(string query, int topK) => new()
    {
        Intent = "recall",
        TargetTables = new List<string> { MemoryTables.Episodic, MemoryTables.Semantic, MemoryTables.Archival },
        Filters = new Dictionary<string, object?>(),
        RankingHints = new List<string>(),
        TemporalIntent = null,
        TopK = topK,
        RawJson = "{\"plan_fallback\":true}",
        PlanFallback = true
    };
}
