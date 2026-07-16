using System.Text.Json;
using System.Text.RegularExpressions;
using PsmMemory.Core.Models;

namespace PsmMemory.Core;

/// <summary>Ported directly from psm-core/src/ranking.ts: hybridRankMemories() and tokenize().</summary>
public static partial class Ranking
{
    private static readonly HashSet<string> Stopwords = new()
    {
        "the", "and", "for", "that", "this", "with", "you", "your", "what", "when", "where", "why", "how", "are",
        "was", "were", "has", "have", "had", "from", "about", "into", "onto", "then", "than", "they", "them",
        "does", "did", "doing", "done", "will", "would", "could", "should", "their", "there", "here", "also"
    };

    public sealed class HybridRankOptions
    {
        public required int TopK { get; init; }
        public Dictionary<string, double>? VectorScores { get; init; }
        public IReadOnlyCollection<string>? PreferredTables { get; init; }
        public double MinScore { get; init; } = 0;
    }

    public static List<RankedMemory> RankMemories(string query, IEnumerable<MemoryRecord> memories, int topK) =>
        HybridRankMemories(query, memories, new HybridRankOptions { TopK = topK });

    public static List<RankedMemory> HybridRankMemories(string query, IEnumerable<MemoryRecord> memories, HybridRankOptions options)
    {
        var qTokens = Tokenize(query);
        var qNumbers = Numbers(query);
        var qEntities = EntityTokens(query);
        var temporalQuestion = TemporalQuestionRegex().IsMatch(query);
        var preferredTables = new HashSet<string>(options.PreferredTables ?? Array.Empty<string>());

        var ranked = new List<RankedMemory>();
        foreach (var memory in Dedupe(memories))
        {
            var tags = ParseJsonStringArray(memory.Tags);
            var sourceEpisodes = ParseJsonStringArray(memory.SourceEpisodes);
            var searchable = string.Join(" ", new[]
            {
                memory.Content,
                string.Join(" ", tags),
                memory.SourceKind ?? "",
                memory.SourceId ?? "",
                memory.SourceLabel ?? "",
                memory.TemporalExpression ?? "",
                memory.ResolvedTime ?? "",
                memory.SourceTimestamp ?? ""
            });
            var memoryTokens = Tokenize(searchable);
            var memorySet = new HashSet<string>(memoryTokens);
            var vectorScore = options.VectorScores is not null && options.VectorScores.TryGetValue(MemoryKey(memory), out var vs) ? vs : 0;
            var exactCoverage = qTokens.Count == 0 ? 0 : (double)qTokens.Count(t => memorySet.Contains(t)) / qTokens.Count;
            var rareExact = qTokens.Count(t => t.Length >= 5 && memorySet.Contains(t));
            var numberScore = OverlapRatio(qNumbers, Numbers(searchable));
            var temporalScore = temporalQuestion ? TemporalSignal(searchable) : 0;
            var entityScore = OverlapRatio(qEntities, memoryTokens);
            var tagScore = LexicalScore(qTokens, Tokenize(string.Join(" ", tags)));
            var tableBoost = preferredTables.Contains(memory.Table) ? 0.08 : 0;
            var lexical = LexicalScore(qTokens, memoryTokens);

            var score =
                0.9 * lexical +
                0.75 * exactCoverage +
                0.18 * rareExact +
                0.55 * numberScore +
                0.25 * temporalScore +
                0.4 * entityScore +
                0.25 * tagScore +
                0.8 * vectorScore +
                tableBoost +
                0.12 * (memory.Confidence ?? 0.5) +
                0.08 * (memory.Strength ?? 0.5) +
                0.04 * (memory.Table == MemoryTables.Semantic ? 1 : 0);

            var metadata = new Dictionary<string, object?>
            {
                ["tags"] = tags,
                ["source_episodes"] = sourceEpisodes,
                ["ranking"] = new Dictionary<string, object?>
                {
                    ["lexical"] = Math.Round(lexical, 6),
                    ["exact_coverage"] = Math.Round(exactCoverage, 6),
                    ["rare_exact"] = rareExact,
                    ["number"] = Math.Round(numberScore, 6),
                    ["temporal"] = Math.Round(temporalScore, 6),
                    ["entity"] = Math.Round(entityScore, 6),
                    ["tag"] = Math.Round(tagScore, 6),
                    ["vector"] = Math.Round(vectorScore, 6),
                    ["preferred_table"] = tableBoost > 0
                }
            };
            ranked.Add(new RankedMemory { Memory = memory, Score = Math.Round(score, 6), Metadata = metadata });
        }

        var sorted = ranked.Where(m => m.Score >= options.MinScore).OrderByDescending(m => m.Score).ToList();
        return SuppressDuplicateContent(sorted).Take(options.TopK).ToList();
    }

    public static List<string> Tokenize(string text)
    {
        var matches = TokenRegex().Matches(text.ToLowerInvariant());
        var result = new List<string>();
        foreach (Match m in matches)
        {
            var token = NormalizeToken(m.Value);
            if (token.Length > 2 && !Stopwords.Contains(token)) result.Add(token);
        }
        return result;
    }

    private static double LexicalScore(List<string> queryTokens, List<string> memoryTokens)
    {
        if (queryTokens.Count == 0 || memoryTokens.Count == 0) return 0;
        var memorySet = new HashSet<string>(memoryTokens);
        var overlap = queryTokens.Count(t => memorySet.Contains(t));
        return overlap / Math.Sqrt(queryTokens.Count * (double)memoryTokens.Count);
    }

    private static double OverlapRatio(List<string> queryValues, List<string> memoryValues)
    {
        if (queryValues.Count == 0 || memoryValues.Count == 0) return 0;
        var memorySet = new HashSet<string>(memoryValues);
        return (double)queryValues.Count(v => memorySet.Contains(v)) / queryValues.Count;
    }

    private static List<string> ParseJsonStringArray(string? value)
    {
        if (string.IsNullOrEmpty(value)) return new List<string>();
        try
        {
            var parsed = JsonSerializer.Deserialize<List<string>>(value);
            return parsed ?? new List<string>();
        }
        catch
        {
            return new List<string>();
        }
    }

    private static string NormalizeToken(string token)
    {
        if (token.EndsWith("ies") && token.Length > 4) return token[..^3] + "y";
        if (token.EndsWith("es") && token.Length > 4) return token[..^2];
        if (token.EndsWith("s") && token.Length > 3) return token[..^1];
        return token;
    }

    private static List<string> Numbers(string text) =>
        NumberRegex().Matches(text).Select(m => m.Value).ToList();

    private static double TemporalSignal(string text)
    {
        if (FourDigitYearRegex().IsMatch(text)) return 1;
        if (DayMonthRegex().IsMatch(text)) return 0.9;
        if (MonthOnlyRegex().IsMatch(text)) return 0.7;
        if (RelativeTimeRegex().IsMatch(text)) return 0.45;
        return 0;
    }

    private static List<string> EntityTokens(string text) =>
        EntityRegex().Matches(text).Select(m => NormalizeToken(m.Value.ToLowerInvariant())).ToList();

    private static string MemoryKey(MemoryRecord memory) => $"{memory.Table}:{memory.Id}";

    private static List<MemoryRecord> Dedupe(IEnumerable<MemoryRecord> memories)
    {
        var seen = new HashSet<string>();
        var result = new List<MemoryRecord>();
        foreach (var memory in memories)
        {
            var key = MemoryKey(memory);
            if (!seen.Add(key)) continue;
            result.Add(memory);
        }
        return result;
    }

    private static List<RankedMemory> SuppressDuplicateContent(List<RankedMemory> memories)
    {
        var seen = new HashSet<string>();
        var result = new List<RankedMemory>();
        foreach (var memory in memories)
        {
            var key = DuplicateContentKey(memory.Content);
            if (!seen.Add(key)) continue;
            result.Add(memory);
        }
        return result;
    }

    private static string DuplicateContentKey(string content) =>
        string.Join(" ", Tokenize(content).Take(28));

    [GeneratedRegex("[a-z0-9]+")]
    private static partial Regex TokenRegex();

    [GeneratedRegex(@"\b\d{2,4}\b")]
    private static partial Regex NumberRegex();

    [GeneratedRegex(@"\bwhen\b|\bdate\b|\byear\b|\bmonth\b|\bday\b|\btime\b", RegexOptions.IgnoreCase)]
    private static partial Regex TemporalQuestionRegex();

    [GeneratedRegex(@"\b\d{4}\b")]
    private static partial Regex FourDigitYearRegex();

    [GeneratedRegex(@"\b\d{1,2}\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b", RegexOptions.IgnoreCase)]
    private static partial Regex DayMonthRegex();

    [GeneratedRegex(@"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b", RegexOptions.IgnoreCase)]
    private static partial Regex MonthOnlyRegex();

    [GeneratedRegex(@"\b(yesterday|today|tomorrow|last year|last week|last month|next year|next week|next month)\b", RegexOptions.IgnoreCase)]
    private static partial Regex RelativeTimeRegex();

    [GeneratedRegex(@"\b[A-Z][a-zA-Z]{2,}\b")]
    private static partial Regex EntityRegex();
}
