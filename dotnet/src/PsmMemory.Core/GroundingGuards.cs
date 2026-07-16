using System.Text.RegularExpressions;
using PsmMemory.Core.Models;

namespace PsmMemory.Core;

public sealed class GroundingOverlapScore
{
    public required int Overlap { get; init; }
    public required int Required { get; init; }
    public required bool Grounded { get; init; }
}

public sealed class StorageGuardResult
{
    public required StorageDecision Decision { get; init; }
    public required bool Rejected { get; init; }
    public string? GuardRoute { get; init; }
    public string? GuardReason { get; init; }
}

/// <summary>
/// Ported directly from psm-core/src/grounding-guards.ts: applyStorageGuards(), hasCurriculumBleed(),
/// groundingOverlapScore(). Rejects (downgrades to ignore) any storage decision whose content isn't
/// grounded in the source text, or that matches the curriculum-bleed blocklist.
/// </summary>
public static partial class GroundingGuards
{
    [GeneratedRegex(
        "checkpoint|powershell|gate datasets|nvidia-smi|direct probe|token budget|runpod|fact parser|malformed parser|constoursated|gate6|expanded probe|gate-?\\d",
        RegexOptions.IgnoreCase)]
    private static partial Regex BleedPattern();

    [GeneratedRegex(@"^\d+$")]
    private static partial Regex AllDigitsRegex();

    public static bool HasCurriculumBleed(string text) => BleedPattern().IsMatch(text);

    public static List<string> SignificantTokens(string text) =>
        Ranking.Tokenize(text).Where(token => token.Length >= 3 && !AllDigitsRegex().IsMatch(token)).ToList();

    public static GroundingOverlapScore GroundingOverlapScore(string rememberTarget, string storedText)
    {
        var inputTokens = SignificantTokens(rememberTarget);
        if (inputTokens.Count == 0)
        {
            return new GroundingOverlapScore { Overlap = 0, Required = 0, Grounded = true };
        }
        var storedSet = new HashSet<string>(SignificantTokens(storedText));
        var overlap = inputTokens.Count(t => storedSet.Contains(t));
        var required = Math.Min(2, Math.Max(1, (int)Math.Ceiling(inputTokens.Count * 0.1)));
        return new GroundingOverlapScore { Overlap = overlap, Required = required, Grounded = overlap >= required };
    }

    public static bool IsGroundedInSource(string rememberTarget, string storedText) =>
        GroundingOverlapScore(rememberTarget, storedText).Grounded;

    private static string StoredTextFromDecision(StorageDecision decision)
    {
        var content = decision.Memory?.Content?.Trim() ?? "";
        var factParts = decision.Facts.SelectMany(fact => new[]
        {
            fact.Subject ?? "",
            fact.Predicate ?? "",
            fact.ValueText ?? "",
            fact.EvidenceText ?? ""
        });
        return string.Join(" ", new[] { content }.Concat(factParts).Where(s => !string.IsNullOrEmpty(s)));
    }

    private static bool WouldPersist(StorageDecision decision)
    {
        if (decision.ParseError is not null) return false;
        var route = Actions.RouteForAction(decision.Action);
        if (route is Actions.Routes.Ignore or Actions.Routes.RecallOnly) return false;
        return !string.IsNullOrWhiteSpace(StoredTextFromDecision(decision));
    }

    public static StorageGuardResult ApplyStorageGuards(string rememberTarget, StorageDecision decision)
    {
        if (!WouldPersist(decision))
        {
            return new StorageGuardResult { Decision = decision, Rejected = false };
        }
        var storedText = StoredTextFromDecision(decision);
        if (HasCurriculumBleed(storedText))
        {
            return new StorageGuardResult
            {
                Decision = decision,
                Rejected = true,
                GuardRoute = "grounding_reject_bleed",
                GuardReason = "Stored content matches curriculum bleed blocklist."
            };
        }
        if (!IsGroundedInSource(rememberTarget, storedText))
        {
            return new StorageGuardResult
            {
                Decision = decision,
                Rejected = true,
                GuardRoute = "grounding_reject",
                GuardReason = "Stored content is not grounded in remember_target tokens."
            };
        }
        return new StorageGuardResult { Decision = decision, Rejected = false };
    }
}
