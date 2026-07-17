namespace PsmMemory.Core.Models;

/// <summary>
/// The three memory tiers a <see cref="MemoryRecord"/> can live in.
/// Mirrors the "episodic" | "semantic" | "archival" union in psm-core's types.ts.
/// </summary>
public static class MemoryTables
{
    public const string Episodic = "episodic";
    public const string Semantic = "semantic";
    public const string Archival = "archival";

    public static readonly IReadOnlyList<string> All = new[] { Episodic, Semantic, Archival };
}

/// <summary>Ported from psm-core/src/types.ts MemoryPayload.</summary>
public sealed class MemoryPayload
{
    public string? Content { get; set; }
    public string? Type { get; set; }
    public double? Strength { get; set; }
    public double? DecayRate { get; set; }
    public double? EmotionalWeight { get; set; }
    public double? Confidence { get; set; }
    public List<string>? Tags { get; set; }
    public List<string>? SourceEpisodes { get; set; }
    public string? SourceKind { get; set; }
    public string? SourceId { get; set; }
    public string? SourceTimestamp { get; set; }
    public string? SourceLabel { get; set; }
    public string? TemporalExpression { get; set; }
    public string? ResolvedTime { get; set; }
    public double? ResolvedTimeConfidence { get; set; }

    public MemoryPayload Clone() => new()
    {
        Content = Content,
        Type = Type,
        Strength = Strength,
        DecayRate = DecayRate,
        EmotionalWeight = EmotionalWeight,
        Confidence = Confidence,
        Tags = Tags is null ? null : new List<string>(Tags),
        SourceEpisodes = SourceEpisodes is null ? null : new List<string>(SourceEpisodes),
        SourceKind = SourceKind,
        SourceId = SourceId,
        SourceTimestamp = SourceTimestamp,
        SourceLabel = SourceLabel,
        TemporalExpression = TemporalExpression,
        ResolvedTime = ResolvedTime,
        ResolvedTimeConfidence = ResolvedTimeConfidence
    };
}

/// <summary>Ported from psm-core/src/types.ts MemoryRecord (a row read back from the store).</summary>
public sealed class MemoryRecord
{
    public required string Id { get; set; }
    public required string UserId { get; set; }
    public required string Content { get; set; }
    public double? Strength { get; set; }
    public double? DecayRate { get; set; }
    public double? EmotionalWeight { get; set; }
    public double? Confidence { get; set; }
    public string? Tags { get; set; }
    public string? SourceEpisodes { get; set; }
    public string? SourceKind { get; set; }
    public string? SourceId { get; set; }
    public string? SourceTimestamp { get; set; }
    public string? SourceLabel { get; set; }
    public string? TemporalExpression { get; set; }
    public string? ResolvedTime { get; set; }
    public double? ResolvedTimeConfidence { get; set; }
    public required string Table { get; set; }
    public string? CreatedAt { get; set; }
    public string? LastAccessed { get; set; }
}

/// <summary>Ported from psm-core/src/types.ts RankedMemory (MemoryRecord + hybrid ranking score/metadata).</summary>
public sealed class RankedMemory
{
    public required MemoryRecord Memory { get; init; }
    public required double Score { get; init; }
    public required Dictionary<string, object?> Metadata { get; init; }

    public string Id => Memory.Id;
    public string Table => Memory.Table;
    public string Content => Memory.Content;
}

/// <summary>Ported from psm-core/src/types.ts WrittenMemoryRef.</summary>
public sealed record WrittenMemoryRef(string Table, string Id, string Content);

/// <summary>Ported from psm-core/src/types.ts MemorySourceMetadata.</summary>
public sealed class MemorySourceMetadata
{
    public string? SourceKind { get; set; }
    public string? SourceId { get; set; }
    public string? SourceTimestamp { get; set; }
    public string? SourceLabel { get; set; }
}
