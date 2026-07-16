using System.Text.Json;
using PsmMemory.Core.Models;

namespace PsmMemory.Core.Decisions;

/// <summary>
/// Ported from psm-core/src/json.ts: parseStorageDecision, parseRecallPlan, plus the fact/indexable
/// normalization helpers. Also ports the fail-safe decision shape from
/// psm-model/src/psm_model/storage_decision_repair.py's FAILSAFE_DECISION — used only when both the
/// first parse AND the repair retry fail, so a parse failure can never corrupt the store.
/// </summary>
public static class StorageDecisionParser
{
    /// <summary>
    /// psm-model/src/psm_model/storage_decision_repair.py: FAILSAFE_DECISION.
    /// action=ignore, memory=null — this is the only decision PsmService will accept without any
    /// grounding/parse validation, precisely because it can never write anything to the store.
    /// </summary>
    public static StorageDecision FailsafeDecision(string rawJson, string? parseError = null) => new()
    {
        Action = Actions.Kinds.Ignore,
        Memory = null,
        Facts = new List<MemoryFactPayload>(),
        Reasoning = "fail-safe: model output unparseable; storing nothing",
        RawJson = rawJson,
        ParseError = parseError ?? "unparseable"
    };

    public static string? ExtractJsonObject(string text)
    {
        var start = text.IndexOf('{');
        var end = text.LastIndexOf('}');
        if (start < 0 || end <= start) return null;
        return text[start..(end + 1)];
    }

    public static StorageDecision ParseStorageDecision(string rawText, string fallbackContent, string fallbackAction = "store_episodic")
    {
        var rawJson = ExtractJsonObject(rawText) ?? rawText.Trim();
        try
        {
            using var doc = JsonDocument.Parse(rawJson);
            var root = doc.RootElement;
            var memory = NormalizeMemory(TryGetProperty(root, "memory"), fallbackContent);
            return new StorageDecision
            {
                Action = Actions.NormalizeAction(TryGetString(root, "action") ?? fallbackAction),
                Memory = memory,
                Facts = NormalizeFacts(TryGetProperty(root, "facts")),
                Indexables = NormalizeIndexables(TryGetProperty(root, "indexables")),
                Reasoning = TryGetString(root, "reasoning") is { Length: > 0 } r ? r : "Model output missing explicit reasoning; applied parser defaults.",
                Confidence = TryGetDouble(root, "confidence") ?? memory?.Confidence,
                EmotionalWeight = TryGetDouble(root, "emotional_weight") ?? memory?.EmotionalWeight,
                ContradictionScore = TryGetDouble(root, "contradiction_score"),
                RawJson = rawJson
            };
        }
        catch (Exception error)
        {
            return new StorageDecision
            {
                Action = Actions.NormalizeAction(fallbackAction),
                Memory = new MemoryPayload
                {
                    Content = fallbackContent,
                    Type = "episodic",
                    Confidence = 0.5,
                    EmotionalWeight = 0.1,
                    Tags = new List<string> { "parse_fallback" }
                },
                Reasoning = $"Model returned invalid JSON; stored fallback content. {error.Message}",
                Confidence = 0.5,
                EmotionalWeight = 0.1,
                RawJson = rawJson,
                ParseError = error.Message
            };
        }
    }

    public static ConsolidationDecision ParseConsolidationDecision(string rawText)
    {
        var rawJson = ExtractJsonObject(rawText) ?? rawText.Trim();
        try
        {
            using var doc = JsonDocument.Parse(rawJson);
            var root = doc.RootElement;
            var action = TryGetString(root, "action");
            if (string.IsNullOrWhiteSpace(action)) throw new JsonException("consolidation decision missing action");
            return new ConsolidationDecision
            {
                Action = Actions.NormalizeAction(action),
                TargetMemoryId = TryGetString(root, "target_memory_id"),
                MergedContent = TryGetString(root, "merged_content"),
                Reasoning = TryGetString(root, "reasoning") is { Length: > 0 } r ? r : "PSM consolidator did not provide reasoning.",
                RawJson = rawJson
            };
        }
        catch (Exception error)
        {
            return new ConsolidationDecision
            {
                Action = Actions.Kinds.StoreEpisodic,
                Reasoning = $"Consolidation output unparseable; keeping storage decision as-is. {error.Message}",
                RawJson = rawJson,
                ParseError = error.Message
            };
        }
    }

    public static RecallPlan ParseRecallPlan(string rawText, string query, int topK = 5)
    {
        var rawJson = ExtractJsonObject(rawText) ?? rawText.Trim();
        try
        {
            using var doc = JsonDocument.Parse(rawJson);
            var root = doc.RootElement;
            var (tables, fallback) = NormalizeTables(TryGetProperty(root, "target_tables"));
            return new RecallPlan
            {
                Intent = TryGetString(root, "intent") is { Length: > 0 } intent ? intent : "recall",
                TargetTables = tables,
                Filters = new Dictionary<string, object?>(),
                RankingHints = StringArray(TryGetProperty(root, "ranking_hints")),
                TemporalIntent = TryGetString(root, "temporal_intent"),
                TopK = Math.Min(PositiveInt(TryGetProperty(root, "top_k"), topK), topK),
                RawJson = rawJson,
                PlanFallback = fallback
            };
        }
        catch (Exception error)
        {
            return new RecallPlan
            {
                Intent = "recall",
                TargetTables = new List<string> { MemoryTables.Semantic, MemoryTables.Episodic },
                RankingHints = Keywords(query),
                TopK = topK,
                RawJson = rawJson,
                PlanFallback = true,
                ParseError = error.Message
            };
        }
    }

    // --- normalization helpers (ported from json.ts) --------------------------------------------

    private static MemoryPayload? NormalizeMemory(JsonElement? value, string fallbackContent)
    {
        if (value is null || value.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined) return null;
        var el = value.Value;
        if (el.ValueKind == JsonValueKind.String)
        {
            var s = el.GetString();
            return string.IsNullOrWhiteSpace(s) ? null : new MemoryPayload { Content = s };
        }
        if (el.ValueKind != JsonValueKind.Object) return null;
        var content = TryGetString(el, "content");
        if (string.IsNullOrWhiteSpace(content)) return null;
        return new MemoryPayload
        {
            Content = content,
            Type = TryGetString(el, "type"),
            Strength = TryGetDouble(el, "strength"),
            DecayRate = TryGetDouble(el, "decay_rate"),
            EmotionalWeight = TryGetDouble(el, "emotional_weight"),
            Confidence = TryGetDouble(el, "confidence"),
            Tags = StringArray(TryGetProperty(el, "tags")),
            SourceEpisodes = StringArray(TryGetProperty(el, "source_episodes")),
            SourceKind = TryGetString(el, "source_kind"),
            SourceId = TryGetString(el, "source_id"),
            SourceTimestamp = TryGetString(el, "source_timestamp"),
            SourceLabel = TryGetString(el, "source_label"),
            TemporalExpression = TryGetString(el, "temporal_expression"),
            ResolvedTime = TryGetString(el, "resolved_time"),
            ResolvedTimeConfidence = TryGetDouble(el, "resolved_time_confidence")
        };
    }

    private static List<MemoryFactPayload> NormalizeFacts(JsonElement? value)
    {
        var result = new List<MemoryFactPayload>();
        if (value is not { ValueKind: JsonValueKind.Array } arr) return result;
        foreach (var item in arr.EnumerateArray())
        {
            var fact = NormalizeFact(item);
            if (fact is not null) result.Add(fact);
        }
        return result;
    }

    private static readonly HashSet<string> GenericFactSubjects = new(StringComparer.OrdinalIgnoreCase)
    {
        "person", "conversation context", "context", "current conversation", "current turn"
    };

    private static bool IsInvalidEvidenceText(string value)
    {
        var normalized = value.ToLowerInvariant();
        return normalized.Contains("benchmark dataset")
            || normalized.Contains("conversation-memory input")
            || normalized.Contains("extraction guidance")
            || normalized.Contains("current turn to remember")
            || normalized.Contains("previous context")
            || normalized.Contains("source id:");
    }

    private static MemoryFactPayload? NormalizeFact(JsonElement value)
    {
        if (value.ValueKind != JsonValueKind.Object) return null;
        var subject = TryGetString(value, "subject");
        var predicate = NormalizePredicate(TryGetString(value, "predicate"));
        var valueText = TryGetString(value, "value_text") ?? ValueToText(TryGetProperty(value, "value"));
        var inferenceKind = TryGetString(value, "inference_kind");
        var evidenceText = TryGetString(value, "evidence_text");
        if (string.IsNullOrWhiteSpace(subject) || string.IsNullOrWhiteSpace(predicate) || string.IsNullOrWhiteSpace(valueText)) return null;
        if (GenericFactSubjects.Contains(subject.Trim())) return null;
        if (!string.IsNullOrEmpty(inferenceKind) && inferenceKind != "explicit") return null;
        if (string.IsNullOrWhiteSpace(evidenceText)) return null;
        if (IsInvalidEvidenceText(evidenceText)) return null;
        return new MemoryFactPayload
        {
            Subject = subject,
            Predicate = predicate,
            Object = TryGetString(value, "object"),
            ValueText = valueText,
            FactType = TryGetString(value, "fact_type"),
            Confidence = TryGetDouble(value, "confidence"),
            InferenceKind = string.IsNullOrEmpty(inferenceKind) ? "explicit" : inferenceKind,
            EvidenceText = evidenceText,
            TemporalExpression = TryGetString(value, "temporal_expression"),
            ResolvedTime = TryGetString(value, "resolved_time"),
            ResolvedTimeConfidence = TryGetDouble(value, "resolved_time_confidence")
        };
    }

    private static List<IndexablePayload> NormalizeIndexables(JsonElement? value)
    {
        var result = new List<IndexablePayload>();
        if (value is not { ValueKind: JsonValueKind.Array } arr) return result;
        foreach (var item in arr.EnumerateArray())
        {
            var row = NormalizeIndexable(item);
            if (row is not null) result.Add(row);
        }
        return result;
    }

    private static IndexablePayload? NormalizeIndexable(JsonElement value)
    {
        if (value.ValueKind != JsonValueKind.Object) return null;
        var kind = TryGetString(value, "kind")?.Trim().ToLowerInvariant();
        if (kind is not (IndexableKinds.Mnemonic or IndexableKinds.FactAnchor or IndexableKinds.Workflow)) return null;
        var key = TryGetString(value, "key");
        if (string.IsNullOrWhiteSpace(key)) return null;
        var normalizedKey = System.Text.RegularExpressions.Regex.Replace(key.Trim().ToLowerInvariant(), "[^a-z0-9-]+", "-").Trim('-');
        return new IndexablePayload
        {
            Kind = kind,
            Key = normalizedKey,
            TargetMemoryTable = TryGetString(value, "target_memory_table"),
            TargetMemoryId = TryGetString(value, "target_memory_id"),
            Steps = StringArray(TryGetProperty(value, "steps")),
            Salience = TryGetDouble(value, "salience"),
            ReconstructiveHint = TryGetString(value, "reconstructive_hint"),
            EvidenceText = TryGetString(value, "evidence_text"),
            Tags = StringArray(TryGetProperty(value, "tags"))
        };
    }

    private static (List<string> Tables, bool Fallback) NormalizeTables(JsonElement? value)
    {
        var allowed = new HashSet<string> { MemoryTables.Semantic, MemoryTables.Episodic, MemoryTables.Archival };
        var tables = StringArray(value).Where(allowed.Contains).ToList();
        return tables.Count > 0 ? (tables, false) : (new List<string> { MemoryTables.Semantic, MemoryTables.Episodic }, true);
    }

    private static string? NormalizePredicate(string? value)
    {
        if (string.IsNullOrEmpty(value)) return null;
        var cleaned = System.Text.RegularExpressions.Regex.Replace(value.Trim().ToLowerInvariant(), "[^a-z0-9]+", "_").Trim('_');
        return string.IsNullOrEmpty(cleaned) ? null : cleaned;
    }

    private static string? ValueToText(JsonElement? value)
    {
        if (value is null) return null;
        var el = value.Value;
        return el.ValueKind switch
        {
            JsonValueKind.String => el.GetString(),
            JsonValueKind.Number => el.GetRawText(),
            JsonValueKind.True or JsonValueKind.False => el.GetBoolean().ToString().ToLowerInvariant(),
            _ => null
        };
    }

    private static List<string> Keywords(string text) =>
        System.Text.RegularExpressions.Regex.Matches(text.ToLowerInvariant(), "[a-z0-9]{3,}")
            .Select(m => m.Value).Take(12).ToList();

    // --- JsonElement helpers -----------------------------------------------------------------

    private static JsonElement? TryGetProperty(JsonElement el, string name) =>
        el.ValueKind == JsonValueKind.Object && el.TryGetProperty(name, out var v) ? v : null;

    private static string? TryGetString(JsonElement el, string name)
    {
        var prop = TryGetProperty(el, name);
        if (prop is null) return null;
        var value = prop.Value;
        var s = value.ValueKind == JsonValueKind.String ? value.GetString() : null;
        return string.IsNullOrWhiteSpace(s) ? null : s;
    }

    private static double? TryGetDouble(JsonElement el, string name)
    {
        var prop = TryGetProperty(el, name);
        if (prop is null) return null;
        var value = prop.Value;
        if (value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var d)) return d;
        if (value.ValueKind == JsonValueKind.String && double.TryParse(value.GetString(), out var ds)) return ds;
        return null;
    }

    private static int PositiveInt(JsonElement? value, int fallback)
    {
        if (value is null) return fallback;
        var el = value.Value;
        if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var i) && i > 0) return i;
        return fallback;
    }

    private static List<string> StringArray(JsonElement? value)
    {
        var result = new List<string>();
        if (value is not { ValueKind: JsonValueKind.Array } arr) return result;
        foreach (var item in arr.EnumerateArray())
        {
            var s = item.ValueKind == JsonValueKind.String ? item.GetString() : item.ToString();
            if (!string.IsNullOrWhiteSpace(s)) result.Add(s);
        }
        return result;
    }
}
