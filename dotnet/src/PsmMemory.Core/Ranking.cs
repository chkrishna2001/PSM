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

    private static readonly HashSet<string> TemporalQuestionWords = new() { "when", "date", "year", "month", "time" };

    /// <summary>A <see cref="MemoryFactRecord"/> scored against a query by <see cref="RankFacts"/>.</summary>
    public sealed class ScoredFact
    {
        public required MemoryFactRecord Fact { get; init; }
        public required double Score { get; init; }

        public string Id => Fact.Id;
        public string Subject => Fact.Subject;
        public string Predicate => Fact.Predicate;
        public string? Object => Fact.Object;
        public string ValueText => Fact.ValueText;
        public string? FactType => Fact.FactType;
        public double? Confidence => Fact.Confidence;
        public string? InferenceKind => Fact.InferenceKind;
        public string? EvidenceText => Fact.EvidenceText;
        public string? SourceMemoryTable => Fact.SourceMemoryTable;
        public string? SourceMemoryId => Fact.SourceMemoryId;
        public string? SourceId => Fact.SourceId;
        public string? SourceTimestamp => Fact.SourceTimestamp;
        public string? TemporalExpression => Fact.TemporalExpression;
        public string? ResolvedTime => Fact.ResolvedTime;
        public double? ResolvedTimeConfidence => Fact.ResolvedTimeConfidence;
    }

    /// <summary>
    /// Ported directly from psm-core/src/service.ts's rankFacts() (a sibling of
    /// <see cref="HybridRankMemories"/>, not an extension of it -- TS never unified the two: facts
    /// and memories are scored by entirely separate heuristics). No candidate-pool filtering happens
    /// here beyond the &gt;=0.2 score threshold; callers select which facts to rank.
    /// </summary>
    public static List<ScoredFact> RankFacts(string query, IEnumerable<MemoryFactRecord> facts, int topK)
    {
        var qTokens = Tokenize(query);
        if (qTokens.Count == 0) return new List<ScoredFact>();
        return facts
            .Select(fact => new ScoredFact { Fact = fact, Score = FactScore(qTokens, fact) })
            .Where(scored => scored.Score >= 0.2)
            .OrderByDescending(scored => scored.Score)
            .Take(topK)
            .ToList();
    }

    /// <summary>Ported directly from psm-core/src/service.ts's factScore().</summary>
    private static double FactScore(List<string> queryTokens, MemoryFactRecord fact)
    {
        var searchable = string.Join(" ", new[]
        {
            fact.Subject, fact.Predicate, fact.Object ?? "", fact.ValueText, fact.FactType ?? "",
            fact.EvidenceText ?? "", fact.TemporalExpression ?? "", fact.ResolvedTime ?? "", fact.SourceId ?? ""
        });
        var memoryTokens = new HashSet<string>(Tokenize(searchable));
        var overlap = queryTokens.Count(t => memoryTokens.Contains(t));
        var coverage = (double)overlap / queryTokens.Count;
        var predicateTokens = Tokenize(fact.Predicate);
        var predicateHit = queryTokens.Any(t => predicateTokens.Contains(t)) ? 0.3 : 0;
        var subjectTokens = Tokenize(fact.Subject);
        var subjectHit = queryTokens.Any(t => subjectTokens.Contains(t)) ? 0.25 : 0;
        var temporalQuestion = queryTokens.Any(t => TemporalQuestionWords.Contains(t));
        var temporalBoost = temporalQuestion && fact.FactType == "temporal_fact" ? 0.4 : 0;
        return Math.Round(coverage + predicateHit + subjectHit + 0.1 * (fact.Confidence ?? 0.75) + temporalBoost, 6);
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
