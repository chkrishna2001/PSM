using PsmMemory.Core.Runtime;

namespace PsmMemory.Core.Models;

/// <summary>Ported from psm-core/src/types.ts RememberRequest.</summary>
public sealed class RememberRequest
{
    public required string LlmResponse { get; set; }
    public string? UserMessage { get; set; }
    public required string UserId { get; set; }
    public MemorySourceMetadata? Source { get; set; }
    public List<string>? ExtraTags { get; set; }
    public bool IncludeExistingMemories { get; set; } = true;

    /// <summary>Which trained-adapter domain to use. Default Coding for backward compatibility.</summary>
    public PsmDomain Domain { get; set; } = PsmDomain.Coding;
}

/// <summary>Ported from psm-core/src/types.ts RecallRequest.</summary>
public sealed class RecallRequest
{
    public required string Question { get; set; }
    public required string UserId { get; set; }
    public int? TopK { get; set; }
    public PsmDomain Domain { get; set; } = PsmDomain.Coding;
}

/// <summary>Ported from psm-core/src/types.ts ContextRequest.</summary>
public sealed class ContextRequest
{
    public required string Prompt { get; set; }
    public required string UserId { get; set; }
    public int? TopK { get; set; }
    public PsmDomain Domain { get; set; } = PsmDomain.Coding;
}

/// <summary>Ported from psm-core/src/types.ts ContextItem (a single grounded context row returned to a caller).</summary>
public sealed class ContextItem
{
    public string? Id { get; set; }
    public string? MemoryId { get; set; }
    public required string Table { get; set; }
    public required string Content { get; set; }
    public string? Reason { get; set; }
    public string? SourceKind { get; set; }
    public string? SourceId { get; set; }
    public string? SourceTimestamp { get; set; }
    public string? SourceLabel { get; set; }
    public string? SavedAt { get; set; }
    public string? TemporalExpression { get; set; }
    public string? ResolvedTime { get; set; }
    public double? ResolvedTimeConfidence { get; set; }
    public double? Score { get; set; }
}

/// <summary>
/// Result of <see cref="PsmMemory.Core.PsmService.RememberAsync"/>. Mirrors the shape of
/// service.ts's remember() plain-object return value (action/route/written/memory/reasoning/...).
/// </summary>
public sealed class RememberResult
{
    public required string UserId { get; set; }
    public required string Action { get; set; }
    public required string Route { get; set; }
    public List<string> Written { get; set; } = new();
    public MemoryPayload? Memory { get; set; }
    public required string Reasoning { get; set; }
    public string? RawModelJson { get; set; }
    public bool RepairAttempted { get; set; }
    public string? ParseError { get; set; }
    public bool GuardRejected { get; set; }
    public List<IndexablePayload> Indexables { get; set; } = new();
}

/// <summary>
/// Result of <see cref="PsmMemory.Core.PsmService.RecallAsync"/> / ContextAsync. Mirrors the
/// shape of service.ts's recall()/context() plain-object return values, minus the embedding-based
/// vector recall and LLM context-render step (no adapter exists for context-render in the ONNX
/// 3-adapter runtime — see PsmService remarks).
/// </summary>
public sealed class RecallResult
{
    public required string UserId { get; set; }
    public required string Query { get; set; }
    public required RecallPlan Plan { get; set; }
    public bool PlanFallback { get; set; }
    public List<RankedMemory> Memories { get; set; } = new();
}
