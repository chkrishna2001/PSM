using System.Text.RegularExpressions;
using PsmMemory.Core.Models;

namespace PsmMemory.Core;

/// <summary>
/// Ported directly from psm-core/src/temporal.ts: normalizeMemoryTemporalFields(),
/// normalizeFactTemporalFields(), isSupportedTemporalExpression(), detectRelativeExpression(),
/// resolveRelativeTime(). Populates TemporalExpression/ResolvedTime/ResolvedTimeConfidence on a
/// storage decision's memory/facts so Ranking.cs's existing (previously dormant) temporal-signal
/// scoring has something to read.
/// </summary>
public static partial class TemporalNormalizer
{
    [GeneratedRegex(
        @"\b(yesterday|today|tomorrow|this week|last week|next week|this month|last month|next month|this year|last year|next year)\b",
        RegexOptions.IgnoreCase)]
    private static partial Regex RelativePattern();

    [GeneratedRegex(@"\b\d{4}\b")]
    private static partial Regex BareYearPattern();

    [GeneratedRegex(
        @"\b\d{1,2}\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
        RegexOptions.IgnoreCase)]
    private static partial Regex DayMonthPattern();

    [GeneratedRegex(
        @"\b(\d{1,2})\s+(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December),?\s+(\d{4})\b",
        RegexOptions.IgnoreCase)]
    private static partial Regex DayMonthYearPattern();

    private static readonly string[] MonthNames =
    {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    };

    /// <summary>Ported from temporal.ts's normalizeMemoryTemporalFields(). Mutates and returns <paramref name="memory"/>.</summary>
    public static MemoryPayload NormalizeMemoryTemporalFields(MemoryPayload memory, string? sourceTimestamp)
    {
        var temporalExpression = memory.TemporalExpression ?? DetectRelativeExpression(memory.Content);
        if (temporalExpression is null) return memory;

        if (!IsSupportedTemporalExpression(temporalExpression))
        {
            memory.TemporalExpression = null;
            memory.ResolvedTime = null;
            memory.ResolvedTimeConfidence = null;
            return memory;
        }

        var resolved = !string.IsNullOrEmpty(sourceTimestamp)
            ? ResolveRelativeTime(temporalExpression, sourceTimestamp)
            : null;
        if (resolved is null) return memory;

        memory.TemporalExpression ??= temporalExpression;
        memory.ResolvedTime = resolved;
        memory.ResolvedTimeConfidence = Math.Max(memory.ResolvedTimeConfidence ?? 0, 0.9);
        return memory;
    }

    /// <summary>Ported from temporal.ts's normalizeFactTemporalFields(). Mutates and returns <paramref name="fact"/>.</summary>
    public static MemoryFactPayload NormalizeFactTemporalFields(MemoryFactPayload fact, string? sourceTimestamp)
    {
        var text = string.Join(" ", new[]
        {
            fact.EvidenceText,
            fact.ValueText,
            fact.Value as string
        }.Where(s => !string.IsNullOrEmpty(s)));

        var temporalExpression = fact.TemporalExpression ?? DetectRelativeExpression(text);
        if (temporalExpression is null) return fact;

        if (!IsSupportedTemporalExpression(temporalExpression))
        {
            fact.TemporalExpression = null;
            fact.ResolvedTime = null;
            fact.ResolvedTimeConfidence = null;
            return fact;
        }

        var resolved = !string.IsNullOrEmpty(sourceTimestamp)
            ? ResolveRelativeTime(temporalExpression, sourceTimestamp)
            : null;
        if (resolved is null) return fact;

        fact.TemporalExpression ??= temporalExpression;
        fact.ResolvedTime = resolved;
        fact.ResolvedTimeConfidence = Math.Max(fact.ResolvedTimeConfidence ?? 0, 0.9);
        return fact;
    }

    /// <summary>Ported from temporal.ts's detectRelativeExpression().</summary>
    public static string? DetectRelativeExpression(string? text)
    {
        if (string.IsNullOrEmpty(text)) return null;
        var match = RelativePattern().Match(text);
        return match.Success ? match.Groups[1].Value.ToLowerInvariant() : null;
    }

    private static bool IsSupportedTemporalExpression(string value)
    {
        var normalized = value.Trim().ToLowerInvariant();
        return RelativePattern().IsMatch(normalized)
            || BareYearPattern().IsMatch(normalized)
            || DayMonthPattern().IsMatch(normalized);
    }

    /// <summary>Ported from temporal.ts's resolveRelativeTime().</summary>
    public static string? ResolveRelativeTime(string expression, string sourceTimestamp)
    {
        var anchor = ParseSourceDate(sourceTimestamp);
        if (anchor is null) return null;

        var normalized = expression.ToLowerInvariant();
        var date = anchor.Value;
        return normalized switch
        {
            "today" => FormatDate(date),
            "yesterday" => FormatDate(date.AddDays(-1)),
            "tomorrow" => FormatDate(date.AddDays(1)),
            "this week" => $"week of {FormatDate(date)}",
            "last week" => $"week before {FormatDate(date)}",
            "next week" => $"week after {FormatDate(date)}",
            "this month" => FormatMonth(date),
            "last month" => FormatMonth(AddMonths(date, -1)),
            "next month" => FormatMonth(AddMonths(date, 1)),
            "this year" => date.Year.ToString(),
            "last year" => (date.Year - 1).ToString(),
            "next year" => (date.Year + 1).ToString(),
            _ => null
        };
    }

    private static DateTime? ParseSourceDate(string value)
    {
        if (DateTime.TryParse(
                value,
                System.Globalization.CultureInfo.InvariantCulture,
                System.Globalization.DateTimeStyles.RoundtripKind
                    | System.Globalization.DateTimeStyles.AllowWhiteSpaces,
                out var direct))
        {
            var utc = direct.Kind switch
            {
                DateTimeKind.Utc => direct,
                DateTimeKind.Local => direct.ToUniversalTime(),
                _ => direct
            };
            return new DateTime(utc.Year, utc.Month, utc.Day, 0, 0, 0, DateTimeKind.Utc);
        }

        var match = DayMonthYearPattern().Match(value);
        if (!match.Success) return null;

        var day = int.Parse(match.Groups[1].Value);
        var month = MonthIndex(match.Groups[2].Value);
        var year = int.Parse(match.Groups[3].Value);
        if (month < 0) return null;

        try
        {
            return new DateTime(year, month + 1, day, 0, 0, 0, DateTimeKind.Utc);
        }
        catch (ArgumentOutOfRangeException)
        {
            return null;
        }
    }

    private static int MonthIndex(string value)
    {
        var normalized = value.ToLowerInvariant();
        var prefix = normalized.Length >= 3 ? normalized[..3] : normalized;
        return Array.FindIndex(MonthNames, month => month.ToLowerInvariant().StartsWith(prefix, StringComparison.Ordinal));
    }

    private static DateTime AddMonths(DateTime date, int months)
    {
        var totalMonths = date.Month - 1 + months;
        var year = date.Year + (int)Math.Floor(totalMonths / 12.0);
        var month = ((totalMonths % 12) + 12) % 12;
        return new DateTime(year, month + 1, 1, 0, 0, 0, DateTimeKind.Utc);
    }

    private static string FormatDate(DateTime date) => $"{date.Day} {MonthNames[date.Month - 1]} {date.Year}";

    private static string FormatMonth(DateTime date) => $"{MonthNames[date.Month - 1]} {date.Year}";
}
