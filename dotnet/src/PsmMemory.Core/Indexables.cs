using System.Text.RegularExpressions;
using PsmMemory.Core.Models;

namespace PsmMemory.Core;

/// <summary>
/// Ported directly from psm-core/src/indexables.ts: buildIndexablesForRemember() and its helpers,
/// plus rankIndexables()/normalizeRecallKey(). Pure deterministic synthesis -- no LLM call involved
/// (see PsmService.RememberAsync remarks for the call site: this only ever runs when a storage
/// decision would actually write a memory and didn't already carry explicit indexables). Style
/// mirrors <see cref="TextSegmenter"/>: a static partial class of source-generated regexes plus
/// small deterministic string-transform helpers.
/// </summary>
public static partial class Indexables
{
    // Exact hardcoded table from indexables.ts's WORKFLOW_HEADER_PATTERNS -- do not add entries
    // without a corresponding change in the original TS source; this port must stay 1:1 with it.
    [GeneratedRegex(@"review.*pull request|pull request review", RegexOptions.IgnoreCase)]
    private static partial Regex ReviewPrRegex();

    [GeneratedRegex(@"runpod.*train|gpu train|train.*runpod", RegexOptions.IgnoreCase)]
    private static partial Regex RunpodTrainRegex();

    [GeneratedRegex(@"grounding bar|promotion bar", RegexOptions.IgnoreCase)]
    private static partial Regex GroundingBarRegex();

    [GeneratedRegex(@"^workflow:([a-z0-9-]+)$", RegexOptions.IgnoreCase)]
    private static partial Regex WorkflowTagRegex();

    // Deliberately a single '#' (not '#{1,3}' like TextSegmenter's header splitter) -- indexables.ts
    // only treats a top-level H1 as a workflow title.
    [GeneratedRegex(@"^#\s+(.+)$", RegexOptions.Multiline)]
    private static partial Regex H1HeaderRegex();

    [GeneratedRegex(@"^\s*\d+\.\s+(.+?)\s*$")]
    private static partial Regex NumberedStepRegex();

    [GeneratedRegex("`([^`]+)`")]
    private static partial Regex BacktickRegex();

    [GeneratedRegex("[^a-z0-9]+")]
    private static partial Regex NonAlnumRunRegex();

    [GeneratedRegex("[^a-z0-9-]+")]
    private static partial Regex NonAlnumOrHyphenRunRegex();

    [GeneratedRegex("[a-z0-9]+")]
    private static partial Regex AlnumTokenRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRunRegex();

    [GeneratedRegex(@"^(.+?[.!?])(?:\s|$)")]
    private static partial Regex FirstSentenceRegex();

    [GeneratedRegex(@"\b\d{4}\b|yesterday|last week|workflow|review|procedure")]
    private static partial Regex SalienceSituationalRegex();

    [GeneratedRegex(@"decision|prefer|constraint|indexable|mnemonic|recall")]
    private static partial Regex SalienceImportanceRegex();

    [GeneratedRegex(@"review|workflow|procedure|how do i", RegexOptions.IgnoreCase)]
    private static partial Regex WorkflowQueryRegex();

    // A separate stopword list from Ranking.Stopwords by design (see task brief) -- meaningfulTokens
    // in the original TS never stems tokens the way ranking.ts's tokenize() does, and the two lists
    // were tuned independently for different purposes (key-building vs. relevance ranking).
    private static readonly HashSet<string> MeaningfulTokenStopwords = new()
    {
        "the", "and", "for", "that", "this", "with", "from", "into", "said", "user", "memory"
    };

    /// <summary>Ported from indexables.ts's BuildIndexablesInput.</summary>
    public sealed class BuildIndexablesInput
    {
        public required string LlmResponse { get; init; }
        public required string MemoryContent { get; init; }
        public List<string>? Tags { get; init; }
        public string? MemoryTable { get; init; }
        public string? MemoryId { get; init; }
        public List<MemoryFactPayload>? Facts { get; init; }
        public List<IndexablePayload>? ExplicitIndexables { get; init; }
    }

    /// <summary>An <see cref="IndexableRecord"/> scored against a query by <see cref="RankIndexables"/>.</summary>
    public sealed class ScoredIndexable
    {
        public required IndexableRecord Record { get; init; }
        public required double Score { get; init; }

        public string Id => Record.Id;
        public string Kind => Record.Kind;
        public string Key => Record.Key;
        public string? TargetMemoryTable => Record.TargetMemoryTable;
        public string? TargetMemoryId => Record.TargetMemoryId;
        public List<string> Steps => Record.Steps;
        public double Salience => Record.Salience;
        public string? ReconstructiveHint => Record.ReconstructiveHint;
        public string? EvidenceText => Record.EvidenceText;
        public List<string> Tags => Record.Tags;
    }

    /// <summary>
    /// Ported directly from indexables.ts's buildIndexablesForRemember(). Deterministic, no LLM
    /// call: (1) explicit indexables the model already emitted just get normalized/passed through;
    /// (2) otherwise, a workflow row is synthesized when the source text has BOTH a recognized
    /// workflow header/tag AND &gt;=2 numbered steps; (3) otherwise a mnemonic row is always produced,
    /// plus an additional fact_anchor row when the supplied facts key off to something distinct from
    /// the mnemonic key.
    /// </summary>
    public static List<IndexablePayload> BuildIndexablesForRemember(BuildIndexablesInput input)
    {
        if (input.ExplicitIndexables is { Count: > 0 })
        {
            var normalized = new List<IndexablePayload>();
            foreach (var row in input.ExplicitIndexables)
            {
                var result = NormalizeIndexable(row, input);
                if (result is not null) normalized.Add(result);
            }
            return normalized;
        }

        var sourceText = input.LlmResponse.Trim();
        if (sourceText.Length == 0) sourceText = input.MemoryContent.Trim();
        if (sourceText.Length == 0) return new List<IndexablePayload>();

        var tags = input.Tags ?? new List<string>();

        var workflowKey = InferWorkflowKey(sourceText, tags);
        var steps = ExtractWorkflowSteps(sourceText);
        if (workflowKey is not null && steps.Count >= 2)
        {
            return new List<IndexablePayload>
            {
                new()
                {
                    Kind = IndexableKinds.Workflow,
                    Key = workflowKey,
                    TargetMemoryTable = input.MemoryTable,
                    TargetMemoryId = input.MemoryId,
                    Steps = steps,
                    Salience = 0.95,
                    ReconstructiveHint = ReconstructiveHint(sourceText),
                    EvidenceText = sourceText.Length > 500 ? sourceText[..500] : sourceText,
                    Tags = UniqueTags(new[] { $"workflow:{workflowKey}", "workflow" }.Concat(tags))
                }
            };
        }

        var content = CleanText(!string.IsNullOrEmpty(input.MemoryContent) ? input.MemoryContent : sourceText);
        var mnemonicKey = BuildMnemonicKey(content, tags);
        var rows = new List<IndexablePayload>
        {
            new()
            {
                Kind = IndexableKinds.Mnemonic,
                Key = mnemonicKey,
                TargetMemoryTable = input.MemoryTable,
                TargetMemoryId = input.MemoryId,
                Salience = SalienceFor(content, tags),
                ReconstructiveHint = ReconstructiveHint(content),
                EvidenceText = content,
                Tags = UniqueTags(tags).Take(6).ToList()
            }
        };

        var factKey = BuildFactAnchorKey(input.Facts);
        if (factKey.Length > 0 && factKey != mnemonicKey)
        {
            rows.Add(new IndexablePayload
            {
                Kind = IndexableKinds.FactAnchor,
                Key = factKey,
                TargetMemoryTable = input.MemoryTable,
                TargetMemoryId = input.MemoryId,
                Salience = Math.Max(rows[0].Salience ?? 0.8, 0.82),
                ReconstructiveHint = ReconstructiveHint(content),
                EvidenceText = content,
                Tags = UniqueTags(tags).Take(6).ToList()
            });
        }

        return rows;
    }

    /// <summary>Ported from indexables.ts's inferWorkflowKey().</summary>
    public static string? InferWorkflowKey(string text, IReadOnlyList<string>? tags)
    {
        foreach (var tag in tags ?? Array.Empty<string>())
        {
            var match = WorkflowTagRegex().Match(tag);
            if (match.Success) return match.Groups[1].Value.ToLowerInvariant();
        }

        var headerMatch = H1HeaderRegex().Match(text);
        var header = headerMatch.Success ? headerMatch.Groups[1].Value : "";
        var haystack = header + "\n" + (text.Length > 240 ? text[..240] : text);

        if (ReviewPrRegex().IsMatch(haystack)) return "review-pr";
        if (RunpodTrainRegex().IsMatch(haystack)) return "runpod-gpu-train";
        if (GroundingBarRegex().IsMatch(haystack)) return "grounding-bar";
        return null;
    }

    /// <summary>Ported from indexables.ts's extractWorkflowSteps().</summary>
    public static List<string> ExtractWorkflowSteps(string text)
    {
        var steps = new List<string>();
        foreach (var line in text.Split('\n'))
        {
            var match = NumberedStepRegex().Match(line);
            if (!match.Success || match.Groups[1].Value.Length == 0) continue;
            steps.Add(StepToId(match.Groups[1].Value));
        }
        return Unique(steps);
    }

    /// <summary>Ported from indexables.ts's stepToId().</summary>
    public static string StepToId(string step)
    {
        var lowered = step.ToLowerInvariant();
        var noBackticks = BacktickRegex().Replace(lowered, "$1");
        var cleaned = NonAlnumRunRegex().Replace(noBackticks, "_").Trim('_');
        var truncated = cleaned.Length > 48 ? cleaned[..48] : cleaned;
        return truncated.Length > 0 ? truncated : "step";
    }

    /// <summary>Ported from indexables.ts's normalizeRecallKey().</summary>
    public static string NormalizeRecallKey(string query)
    {
        var lowered = query.ToLowerInvariant();
        var replaced = NonAlnumRunRegex().Replace(lowered, "-");
        return replaced.Trim('-');
    }

    /// <summary>
    /// Ported directly from indexables.ts's rankIndexables(). Called only from the recall() call
    /// path -- context() never ranks/surfaces indexables in the original TS source (confirmed via
    /// grep: no `rankIndexables` call anywhere in context()).
    /// </summary>
    public static List<ScoredIndexable> RankIndexables(string query, IEnumerable<IndexableRecord> indexables, int topK = 5)
    {
        var normalized = NormalizeRecallKey(query);
        var tokens = Ranking.Tokenize(query);
        return indexables
            .Select(row => new ScoredIndexable { Record = row, Score = IndexableScore(normalized, tokens, row) })
            .Where(scored => scored.Score >= 0.35)
            .OrderByDescending(scored => scored.Score)
            .Take(topK)
            .ToList();
    }

    /// <summary>Ported from indexables.ts's indexableScore().</summary>
    private static double IndexableScore(string normalizedQuery, List<string> queryTokens, IndexableRecord row)
    {
        var score = row.Salience;
        var key = row.Key.ToLowerInvariant();
        if (key.Length > 0 && (normalizedQuery == key || normalizedQuery.Contains(key) || key.Contains(normalizedQuery)))
        {
            score += 0.55;
        }
        if (row.Kind == IndexableKinds.Workflow && WorkflowQueryRegex().IsMatch(normalizedQuery.Replace("-", " ")))
        {
            score += 0.1;
        }
        var searchable = string.Join(" ", new[] { row.Key, row.ReconstructiveHint ?? "" }.Concat(row.Tags).Concat(row.Steps));
        var haystack = new HashSet<string>(Ranking.Tokenize(searchable));
        var overlap = queryTokens.Count(t => haystack.Contains(t));
        if (queryTokens.Count > 0) score += (double)overlap / queryTokens.Count * 0.35;
        return Math.Round(score, 6);
    }

    /// <summary>Ported from indexables.ts's normalizeIndexable() (the explicitIndexables branch).</summary>
    private static IndexablePayload? NormalizeIndexable(IndexablePayload row, BuildIndexablesInput input)
    {
        var key = CleanKey(row.Key);
        if (key.Length == 0) return null;
        return new IndexablePayload
        {
            Kind = string.IsNullOrEmpty(row.Kind) ? IndexableKinds.Mnemonic : row.Kind,
            Key = key,
            TargetMemoryTable = row.TargetMemoryTable ?? input.MemoryTable,
            TargetMemoryId = row.TargetMemoryId ?? input.MemoryId,
            Steps = row.Steps?.Select(StepToId).ToList(),
            Salience = Clamp01(row.Salience ?? 0.85),
            ReconstructiveHint = row.ReconstructiveHint ?? ReconstructiveHint(input.MemoryContent),
            EvidenceText = row.EvidenceText ?? input.MemoryContent,
            Tags = UniqueTags(row.Tags ?? input.Tags ?? new List<string>())
        };
    }

    /// <summary>Ported from indexables.ts's buildMnemonicKey().</summary>
    private static string BuildMnemonicKey(string content, IReadOnlyList<string> tags)
    {
        var tagTokens = tags.SelectMany(tag => MeaningfulTokens(tag.Replace("_", " ")));
        var contentTokens = MeaningfulTokens(content);
        var tokens = Unique(tagTokens.Concat(contentTokens)).Take(4).ToList();
        return tokens.Count > 0 ? string.Join("-", tokens) : "memory-anchor";
    }

    /// <summary>Ported from indexables.ts's buildFactAnchorKey().</summary>
    private static string BuildFactAnchorKey(IReadOnlyList<MemoryFactPayload>? facts)
    {
        var fact = (facts ?? Array.Empty<MemoryFactPayload>()).FirstOrDefault(f =>
            !string.IsNullOrEmpty(f.Subject) && !string.IsNullOrEmpty(f.Predicate) && !string.IsNullOrEmpty(f.ValueText));
        if (fact is null) return "";

        var tokens = Unique(
                MeaningfulTokens(fact.Subject ?? "")
                    .Concat(MeaningfulTokens(fact.Predicate ?? ""))
                    .Concat(MeaningfulTokens(fact.ValueText ?? "")))
            .Take(4)
            .ToList();
        return string.Join("-", tokens);
    }

    /// <summary>Ported from indexables.ts's salienceFor().</summary>
    private static double SalienceFor(string content, IReadOnlyList<string> tags)
    {
        var lower = content.ToLowerInvariant();
        var score = 0.68;
        if (SalienceSituationalRegex().IsMatch(lower)) score += 0.08;
        if (SalienceImportanceRegex().IsMatch(lower)) score += 0.12;
        if (tags.Count > 0) score += 0.04;
        return Math.Round(Math.Min(score, 0.98), 2);
    }

    /// <summary>Ported from indexables.ts's reconstructiveHint().</summary>
    private static string ReconstructiveHint(string content)
    {
        var match = FirstSentenceRegex().Match(content);
        var sentence = match.Success ? match.Groups[1].Value : content;
        return sentence.Length <= 160 ? sentence : sentence[..157].Trim() + "...";
    }

    /// <summary>Ported from indexables.ts's meaningfulTokens(). Deliberately does NOT stem tokens
    /// the way Ranking.Tokenize does -- the original TS never shared that logic between the two.</summary>
    private static List<string> MeaningfulTokens(string text)
    {
        var cleaned = CleanText(text).ToLowerInvariant();
        var result = new List<string>();
        foreach (Match m in AlnumTokenRegex().Matches(cleaned))
        {
            if (m.Value.Length > 2 && !MeaningfulTokenStopwords.Contains(m.Value)) result.Add(m.Value);
        }
        return result;
    }

    /// <summary>Ported from indexables.ts's cleanText().</summary>
    private static string CleanText(string value) => WhitespaceRunRegex().Replace(value.Trim(), " ");

    /// <summary>Ported from indexables.ts's cleanKey().</summary>
    private static string CleanKey(string value) =>
        NonAlnumOrHyphenRunRegex().Replace(value.Trim().ToLowerInvariant(), "-").Trim('-');

    /// <summary>Ported from indexables.ts's uniqueTags().</summary>
    private static List<string> UniqueTags(IEnumerable<string> tags)
    {
        var seen = new HashSet<string>();
        var result = new List<string>();
        foreach (var raw in tags)
        {
            var trimmed = raw?.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            var normalized = WhitespaceRunRegex().Replace(trimmed, "_");
            if (seen.Add(normalized)) result.Add(normalized);
        }
        return result;
    }

    private static List<string> Unique(IEnumerable<string> values)
    {
        var seen = new HashSet<string>();
        var result = new List<string>();
        foreach (var v in values)
        {
            if (seen.Add(v)) result.Add(v);
        }
        return result;
    }

    private static double Clamp01(double value) => Math.Max(0, Math.Min(1, value));
}
