namespace PsmMemory.Core.Models;

/// <summary>
/// Ported from psm-core/src/types.ts StorageDecision. This is the parsed result of the
/// storage adapter's JSON output (or the fail-safe/parse-fallback decision when it can't be parsed).
/// </summary>
public sealed class StorageDecision
{
    public required string Action { get; set; }
    public MemoryPayload? Memory { get; set; }
    public List<MemoryFactPayload> Facts { get; set; } = new();
    public List<IndexablePayload> Indexables { get; set; } = new();
    public required string Reasoning { get; set; }
    public double? Confidence { get; set; }
    public double? EmotionalWeight { get; set; }
    public double? ContradictionScore { get; set; }
    public required string RawJson { get; set; }
    public string? ParseError { get; set; }

    public StorageDecision Clone() => new()
    {
        Action = Action,
        Memory = Memory?.Clone(),
        Facts = new List<MemoryFactPayload>(Facts),
        Indexables = new List<IndexablePayload>(Indexables),
        Reasoning = Reasoning,
        Confidence = Confidence,
        EmotionalWeight = EmotionalWeight,
        ContradictionScore = ContradictionScore,
        RawJson = RawJson,
        ParseError = ParseError
    };
}

/// <summary>Ported from psm-core/src/types.ts RecallPlan (the recall/context planner adapter's parsed output).</summary>
public sealed class RecallPlan
{
    public required string Intent { get; set; }
    public required List<string> TargetTables { get; set; }
    public Dictionary<string, object?> Filters { get; set; } = new();
    public List<string> RankingHints { get; set; } = new();
    public string? TemporalIntent { get; set; }
    public required int TopK { get; set; }
    public required string RawJson { get; set; }
    public bool PlanFallback { get; set; }
    public string? ParseError { get; set; }
}

/// <summary>
/// New: the parsed output of the consolidation adapter (retrieval_plan/consolidation two-step
/// remember flow). No TS equivalent exists — psm-core never had a dedicated consolidation adapter.
/// Schema matches psm-model/prod-memory/prod_memory/hf_prompts.py's compact_consolidation_json:
/// {"reasoning","action","target_memory_id","merged_content"}.
/// </summary>
public sealed class ConsolidationDecision
{
    public required string Action { get; set; }
    public string? TargetMemoryId { get; set; }
    public string? MergedContent { get; set; }
    public required string Reasoning { get; set; }
    public required string RawJson { get; set; }
    public string? ParseError { get; set; }
}

/// <summary>Kind of an indexable row (mnemonic | fact_anchor | workflow). Ported from types.ts IndexableKind.</summary>
public static class IndexableKinds
{
    public const string Mnemonic = "mnemonic";
    public const string FactAnchor = "fact_anchor";
    public const string Workflow = "workflow";
}

/// <summary>Ported from psm-core/src/types.ts IndexablePayload.</summary>
public sealed class IndexablePayload
{
    public required string Kind { get; set; }
    public required string Key { get; set; }
    public string? TargetMemoryTable { get; set; }
    public string? TargetMemoryId { get; set; }
    public List<string>? Steps { get; set; }
    public double? Salience { get; set; }
    public string? ReconstructiveHint { get; set; }
    public string? EvidenceText { get; set; }
    public List<string>? Tags { get; set; }
}

/// <summary>Ported from psm-core/src/types.ts IndexableRecord.</summary>
public sealed class IndexableRecord
{
    public required string Id { get; set; }
    public required string UserId { get; set; }
    public required string Kind { get; set; }
    public required string Key { get; set; }
    public string? TargetMemoryTable { get; set; }
    public string? TargetMemoryId { get; set; }
    public List<string> Steps { get; set; } = new();
    public double Salience { get; set; } = 0.8;
    public string? ReconstructiveHint { get; set; }
    public string? EvidenceText { get; set; }
    public List<string> Tags { get; set; } = new();
    public string? CreatedAt { get; set; }
}
