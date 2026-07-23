using System.Text.RegularExpressions;

namespace PsmMemory.Core;

/// <summary>Ported from psm-core/src/segment-remember.ts's SegmentSplitReason union.</summary>
public enum SegmentSplitReason
{
    Single,
    MarkdownHeader,
    NumberedBlock,
    Paragraph,
    HardMax,
}

/// <summary>Ported from psm-core/src/segment-remember.ts's TextSegment.</summary>
public sealed record TextSegment(int Index, string Text, int EstimatedTokens, SegmentSplitReason SplitReason);

/// <summary>
/// Ported directly from psm-core/src/segment-remember.ts: segmentLlmResponse() and its helpers.
/// Pure deterministic text splitting -- no LLM call involved. Used by PsmService's chunked-remember
/// path to break a long assistant response into semantically-sensible pieces (respecting markdown
/// headers, numbered-list blocks, and paragraph boundaries) before each piece goes through its own
/// independent storage-decision call, so a single long response doesn't force the model to pick one
/// action/one memory.content for content that actually covers multiple distinct facts.
/// </summary>
public static partial class TextSegmenter
{
    private const int DefaultMaxChunkTokens = 1200;
    private const int DefaultMinChunkTokens = 200;

    public static int EstimateTextTokens(string text)
    {
        var trimmed = text.Trim();
        if (trimmed.Length == 0) return 0;
        return Math.Max(1, (int)Math.Ceiling(trimmed.Length / 4.0));
    }

    public static List<TextSegment> SegmentLlmResponse(string text, int? maxChunkTokens = null, int? minChunkTokens = null)
    {
        var maxTokens = maxChunkTokens ?? DefaultMaxChunkTokens;
        var minTokens = minChunkTokens ?? DefaultMinChunkTokens;
        var trimmed = text.Trim();
        if (trimmed.Length == 0) return new List<TextSegment>();

        if (EstimateTextTokens(trimmed) <= maxTokens)
        {
            return new List<TextSegment> { MakeSegment(0, trimmed, SegmentSplitReason.Single) };
        }

        var headerSections = SplitByMarkdownHeaders(trimmed);
        var units = new List<(string Text, SegmentSplitReason Reason)>();
        foreach (var section in headerSections)
        {
            var reason = headerSections.Count > 1 ? SegmentSplitReason.MarkdownHeader : SegmentSplitReason.Paragraph;
            foreach (var unit in SplitPreservingNumberedBlocks(section))
            {
                var unitReason = IsNumberedWorkflowBlock(unit) ? SegmentSplitReason.NumberedBlock : reason;
                if (EstimateTextTokens(unit) <= maxTokens)
                {
                    units.Add((unit, unitReason));
                }
                else
                {
                    foreach (var piece in HardSplitByTokens(unit, maxTokens))
                    {
                        units.Add((piece, SegmentSplitReason.HardMax));
                    }
                }
            }
        }

        var merged = MergeSmallSegments(units, minTokens, maxTokens);
        var result = new List<TextSegment>();
        for (var i = 0; i < merged.Count; i++)
        {
            result.Add(MakeSegment(i, merged[i].Text, merged[i].Reason));
        }
        return result;
    }

    public static string ChunkSourceId(string baseSourceId, int chunkIndex) => $"{baseSourceId}:chunk-{chunkIndex}";

    private static TextSegment MakeSegment(int index, string text, SegmentSplitReason reason) =>
        new(index, text, EstimateTextTokens(text), reason);

    private static List<string> SplitByMarkdownHeaders(string text)
    {
        var lines = text.Split('\n');
        var sections = new List<string>();
        var current = new List<string>();

        void Flush()
        {
            var joined = string.Join("\n", current).Trim();
            if (joined.Length > 0) sections.Add(joined);
            current = new List<string>();
        }

        foreach (var line in lines)
        {
            if (HeaderLineRegex().IsMatch(line) && current.Count > 0)
            {
                Flush();
            }
            current.Add(line);
        }
        Flush();
        return sections.Count > 0 ? sections : new List<string> { text.Trim() };
    }

    private static List<string> SplitPreservingNumberedBlocks(string text)
    {
        var lines = text.Split('\n');
        var units = new List<string>();
        var buffer = new List<string>();
        var numberedCount = 0;

        void Flush()
        {
            var joined = string.Join("\n", buffer).Trim();
            if (joined.Length > 0) units.Add(joined);
            buffer = new List<string>();
            numberedCount = 0;
        }

        foreach (var line in lines)
        {
            var isNumbered = NumberedLineRegex().IsMatch(line);
            var isHeader = HeaderLineRegex().IsMatch(line);

            if (isHeader)
            {
                Flush();
                buffer.Add(line);
                continue;
            }

            if (isNumbered)
            {
                if (numberedCount == 0 && buffer.Count > 0 && !IsNumberedWorkflowBlock(string.Join("\n", buffer)))
                {
                    Flush();
                }
                numberedCount++;
                buffer.Add(line);
                continue;
            }

            if (numberedCount >= 2 && line.Trim().Length > 0 && !isHeader)
            {
                buffer.Add(line);
                continue;
            }

            if (line.Trim().Length == 0)
            {
                if (numberedCount >= 2)
                {
                    Flush();
                }
                else if (buffer.Count > 0)
                {
                    Flush();
                }
                continue;
            }

            if (numberedCount is > 0 and < 2)
            {
                numberedCount = 0;
            }

            if (buffer.Count > 0 && numberedCount == 0)
            {
                Flush();
            }
            buffer.Add(line);
        }

        Flush();
        return units.Count > 0 ? units : new List<string> { text.Trim() };
    }

    private static bool IsNumberedWorkflowBlock(string text)
    {
        var numbered = text.Split('\n').Count(line => NumberedLineRegex().IsMatch(line));
        return numbered >= 2;
    }

    private static List<string> HardSplitByTokens(string text, int maxChunkTokens)
    {
        var paragraphs = MultiNewlineRegex().Split(text).Select(p => p.Trim()).Where(p => p.Length > 0).ToList();
        var chunks = new List<string>();
        var current = "";

        void Flush()
        {
            if (current.Trim().Length > 0) chunks.Add(current.Trim());
            current = "";
        }

        foreach (var paragraph in paragraphs)
        {
            var candidate = current.Length > 0 ? $"{current}\n\n{paragraph}" : paragraph;
            if (EstimateTextTokens(candidate) <= maxChunkTokens)
            {
                current = candidate;
                continue;
            }
            if (current.Length > 0) Flush();
            if (EstimateTextTokens(paragraph) <= maxChunkTokens)
            {
                current = paragraph;
                continue;
            }
            var sentences = SentenceRegex().Matches(paragraph).Select(m => m.Value).ToList();
            if (sentences.Count == 0) sentences.Add(paragraph);
            foreach (var sentence in sentences)
            {
                var next = current.Length > 0 ? $"{current} {sentence.Trim()}" : sentence.Trim();
                if (EstimateTextTokens(next) <= maxChunkTokens)
                {
                    current = next;
                }
                else
                {
                    Flush();
                    current = sentence.Trim();
                }
            }
        }
        Flush();
        return chunks.Count > 0 ? chunks : new List<string> { text.Trim() };
    }

    private static List<(string Text, SegmentSplitReason Reason)> MergeSmallSegments(
        List<(string Text, SegmentSplitReason Reason)> units, int minChunkTokens, int maxChunkTokens)
    {
        if (units.Count <= 1) return units;
        var merged = new List<(string Text, SegmentSplitReason Reason)>();
        (string Text, SegmentSplitReason Reason)? pending = null;

        foreach (var unit in units)
        {
            if (pending is null)
            {
                pending = unit;
                continue;
            }
            var combined = $"{pending.Value.Text}\n\n{unit.Text}";
            var pendingSmall = EstimateTextTokens(pending.Value.Text) < minChunkTokens;
            var unitSmall = EstimateTextTokens(unit.Text) < minChunkTokens;
            if ((pendingSmall || unitSmall) && EstimateTextTokens(combined) <= maxChunkTokens)
            {
                pending = (combined, pending.Value.Reason);
                continue;
            }
            merged.Add(pending.Value);
            pending = unit;
        }
        if (pending is not null) merged.Add(pending.Value);
        return merged;
    }

    [GeneratedRegex(@"^#{1,3}\s+")]
    private static partial Regex HeaderLineRegex();

    [GeneratedRegex(@"^\s*\d+\.\s+")]
    private static partial Regex NumberedLineRegex();

    [GeneratedRegex(@"\n{2,}")]
    private static partial Regex MultiNewlineRegex();

    [GeneratedRegex(@"[^.!?]+[.!?]+|[^.!?]+$")]
    private static partial Regex SentenceRegex();
}
