namespace PsmMemory.Core.Models;

/// <summary>Ported from psm-core/src/types.ts MemoryFactPayload (a fact as extracted from a model decision).</summary>
public sealed class MemoryFactPayload
{
    public string? Subject { get; set; }
    public string? Predicate { get; set; }
    public string? Object { get; set; }
    public object? Value { get; set; }
    public string? ValueText { get; set; }
    public object? ValueJson { get; set; }
    public string? FactType { get; set; }
    public double? Confidence { get; set; }
    public string? InferenceKind { get; set; }
    public string? EvidenceText { get; set; }
    public string? TemporalExpression { get; set; }
    public string? ResolvedTime { get; set; }
    public double? ResolvedTimeConfidence { get; set; }
}

/// <summary>Ported from psm-core/src/types.ts MemoryFactRecord (a fact row read back from the store).</summary>
public sealed class MemoryFactRecord
{
    public required string Id { get; set; }
    public required string UserId { get; set; }
    public required string Subject { get; set; }
    public required string Predicate { get; set; }
    public string? Object { get; set; }
    public required string ValueText { get; set; }
    public string? ValueJson { get; set; }
    public string? FactType { get; set; }
    public double? Confidence { get; set; }
    public string? InferenceKind { get; set; }
    public string? EvidenceText { get; set; }
    public string? SourceMemoryTable { get; set; }
    public string? SourceMemoryId { get; set; }
    public string? SourceId { get; set; }
    public string? SourceTimestamp { get; set; }
    public string? TemporalExpression { get; set; }
    public string? ResolvedTime { get; set; }
    public double? ResolvedTimeConfidence { get; set; }
    public string? CreatedAt { get; set; }
    public string? UpdatedAt { get; set; }
}
