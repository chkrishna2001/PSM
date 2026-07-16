using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Data.Sqlite;
using PsmMemory.Core.Models;

namespace PsmMemory.Core.Store;

/// <summary>Result of <see cref="MemoryStore.ApplyDecision"/>. Ported from store.ts's applyDecision return shape.</summary>
public sealed class ApplyDecisionResult
{
    public required string Action { get; init; }
    public required string Route { get; init; }
    public List<string> Written { get; init; } = new();
    public List<WrittenMemoryRef> MemoryRefs { get; init; } = new();
}

/// <summary>
/// Ported from psm-core/src/store.ts: MemoryStore. Schema (episodic/semantic/archival/conflicts/
/// decay_schedule/decisions/memory_embeddings/memory_facts/indexables) and applyDecision() routing
/// logic are ported faithfully. One addition not present in the TS version: `ignored_decisions`, a
/// side table PsmService uses to log low-confidence ignore outcomes for later async reprocessing
/// (called out explicitly in this port's task brief; store.ts has no equivalent).
/// </summary>
public sealed partial class MemoryStore : IDisposable
{
    private readonly SqliteConnection _db;

    public string DbPath { get; }

    public MemoryStore(string dbPath)
    {
        DbPath = dbPath;
        _db = new SqliteConnection($"Data Source={dbPath}");
        _db.Open();
        Exec("PRAGMA foreign_keys = ON;");
    }

    public void InitializeSchema()
    {
        Exec(SchemaSql);
        EnsureMemoryMetadataColumns();
    }

    private const string SchemaSql = """
        CREATE TABLE IF NOT EXISTS schema_version (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT OR IGNORE INTO schema_version(version) VALUES (1);
        CREATE TABLE IF NOT EXISTS episodic (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          content TEXT NOT NULL,
          strength REAL NOT NULL,
          decay_rate REAL NOT NULL,
          emotional_weight REAL NOT NULL,
          confidence REAL NOT NULL,
          tags TEXT,
          source_kind TEXT,
          source_id TEXT,
          source_timestamp TEXT,
          source_label TEXT,
          temporal_expression TEXT,
          resolved_time TEXT,
          resolved_time_confidence REAL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          last_accessed TEXT,
          promoted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS semantic (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          content TEXT NOT NULL,
          strength REAL NOT NULL,
          decay_rate REAL NOT NULL,
          emotional_weight REAL NOT NULL,
          confidence REAL NOT NULL,
          tags TEXT,
          source_episodes TEXT,
          source_kind TEXT,
          source_id TEXT,
          source_timestamp TEXT,
          source_label TEXT,
          temporal_expression TEXT,
          resolved_time TEXT,
          resolved_time_confidence REAL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          last_accessed TEXT
        );
        CREATE TABLE IF NOT EXISTS archival (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          content TEXT NOT NULL,
          summary TEXT,
          original_type TEXT,
          source_id TEXT,
          archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS conflicts (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          existing_memory_id TEXT,
          existing_memory_type TEXT,
          conflicting_content TEXT NOT NULL,
          conflict_reason TEXT,
          status TEXT NOT NULL DEFAULT 'unresolved',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS decay_schedule (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          memory_key TEXT NOT NULL,
          next_decay TEXT NOT NULL,
          decay_rate REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decisions (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          source TEXT NOT NULL,
          action TEXT NOT NULL,
          route TEXT NOT NULL,
          reasoning TEXT,
          raw_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS memory_embeddings (
          memory_table TEXT NOT NULL,
          memory_id TEXT NOT NULL,
          user_id TEXT NOT NULL,
          model TEXT NOT NULL,
          dimensions INTEGER NOT NULL,
          embedding_json TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (memory_table, memory_id, model)
        );
        CREATE TABLE IF NOT EXISTS memory_facts (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          subject TEXT NOT NULL,
          predicate TEXT NOT NULL,
          object TEXT,
          value_text TEXT NOT NULL,
          value_json TEXT,
          fact_type TEXT,
          confidence REAL,
          inference_kind TEXT,
          evidence_text TEXT,
          source_memory_table TEXT,
          source_memory_id TEXT,
          source_id TEXT,
          source_timestamp TEXT,
          temporal_expression TEXT,
          resolved_time TEXT,
          resolved_time_confidence REAL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_episodic_user_created ON episodic(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_semantic_user_created ON semantic(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_conflicts_status_created ON conflicts(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_decisions_user_created ON decisions(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_embeddings_user_model ON memory_embeddings(user_id, model);
        CREATE INDEX IF NOT EXISTS idx_memory_facts_user_predicate ON memory_facts(user_id, predicate);
        CREATE INDEX IF NOT EXISTS idx_memory_facts_user_subject ON memory_facts(user_id, subject);
        CREATE INDEX IF NOT EXISTS idx_memory_facts_source_memory ON memory_facts(source_memory_table, source_memory_id);
        CREATE TABLE IF NOT EXISTS indexables (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          key TEXT NOT NULL,
          target_memory_table TEXT,
          target_memory_id TEXT,
          steps_json TEXT NOT NULL DEFAULT '[]',
          salience REAL NOT NULL,
          reconstructive_hint TEXT,
          evidence_text TEXT,
          tags TEXT,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(user_id, key)
        );
        CREATE INDEX IF NOT EXISTS idx_indexables_user_key ON indexables(user_id, key);
        CREATE TABLE IF NOT EXISTS ignored_decisions (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          source TEXT,
          content TEXT,
          reasoning TEXT,
          confidence REAL,
          raw_json TEXT,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_ignored_decisions_user_created ON ignored_decisions(user_id, created_at DESC);
        """;

    public ApplyDecisionResult ApplyDecision(
        string userId,
        string source,
        StorageDecision decision,
        IReadOnlyList<string>? extraTags = null,
        (string Id, string Table)? conflictAgainst = null)
    {
        extraTags ??= Array.Empty<string>();
        var route = Actions.RouteForAction(decision.Action);
        InsertDecision(userId, source, decision.Action, route, decision.Reasoning, decision.RawJson);

        var memory = decision.Memory is null ? null : WithExtraTags(CloneWithSourceId(decision.Memory, source), extraTags);
        var content = memory?.Content?.Trim();
        var written = new List<string>();
        var memoryRefs = new List<WrittenMemoryRef>();

        if (!string.IsNullOrEmpty(content) && HasDuplicateMemoryContent(userId, content, source))
        {
            return new ApplyDecisionResult { Action = Actions.Kinds.Ignore, Route = "dedupe_skip", Written = written, MemoryRefs = memoryRefs };
        }

        switch (route)
        {
            case Actions.Routes.Ignore:
            case Actions.Routes.RecallOnly:
                return new ApplyDecisionResult { Action = decision.Action, Route = route, Written = written, MemoryRefs = memoryRefs };

            case Actions.Routes.SemanticUpsert:
            case Actions.Routes.UpdateWithSupersede:
                if (memory is null || string.IsNullOrEmpty(content))
                    return new ApplyDecisionResult { Action = Actions.Kinds.Ignore, Route = Actions.Routes.Ignore, Written = written, MemoryRefs = memoryRefs };
                memoryRefs.Add(new WrittenMemoryRef(MemoryTables.Semantic, InsertSemantic(userId, content, memory, new List<string> { source }), content));
                written.Add(MemoryTables.Semantic);
                InsertDecisionFacts(userId, decision, memoryRefs);
                InsertDecisionIndexables(userId, decision, memoryRefs, memory);
                return new ApplyDecisionResult { Action = decision.Action, Route = route, Written = written, MemoryRefs = memoryRefs };

            case Actions.Routes.DecayExistingThenInsert:
                if (memory is null || string.IsNullOrEmpty(content))
                    return new ApplyDecisionResult { Action = Actions.Kinds.Ignore, Route = Actions.Routes.Ignore, Written = written, MemoryRefs = memoryRefs };
                InsertDecaySchedule(userId, content, memory.DecayRate ?? 0.03);
                memoryRefs.Add(new WrittenMemoryRef(MemoryTables.Episodic, InsertEpisodic(userId, content, memory), content));
                written.Add("decay_schedule");
                written.Add(MemoryTables.Episodic);
                InsertDecisionFacts(userId, decision, memoryRefs);
                InsertDecisionIndexables(userId, decision, memoryRefs, memory);
                return new ApplyDecisionResult { Action = decision.Action, Route = route, Written = written, MemoryRefs = memoryRefs };

            case Actions.Routes.ConflictLogAndHold:
                if (memory is null || string.IsNullOrEmpty(content))
                    return new ApplyDecisionResult { Action = Actions.Kinds.Ignore, Route = Actions.Routes.Ignore, Written = written, MemoryRefs = memoryRefs };
                InsertConflict(
                    userId,
                    content,
                    string.IsNullOrWhiteSpace(decision.Reasoning) ? "PSM flagged potential conflict" : decision.Reasoning,
                    conflictAgainst?.Id,
                    conflictAgainst?.Table);
                written.Add("conflicts");
                if (decision.Action == Actions.Kinds.FlagAndStore)
                {
                    memoryRefs.Add(new WrittenMemoryRef(MemoryTables.Episodic, InsertEpisodic(userId, content, memory), content));
                    written.Add(MemoryTables.Episodic);
                    InsertDecisionFacts(userId, decision, memoryRefs);
                    InsertDecisionIndexables(userId, decision, memoryRefs, memory);
                }
                return new ApplyDecisionResult { Action = decision.Action, Route = route, Written = written, MemoryRefs = memoryRefs };

            default: // episodic_insert
                if (memory is null || string.IsNullOrEmpty(content))
                    return new ApplyDecisionResult { Action = Actions.Kinds.Ignore, Route = Actions.Routes.Ignore, Written = written, MemoryRefs = memoryRefs };
                memoryRefs.Add(new WrittenMemoryRef(MemoryTables.Episodic, InsertEpisodic(userId, content, memory), content));
                written.Add(MemoryTables.Episodic);
                InsertDecisionFacts(userId, decision, memoryRefs);
                InsertDecisionIndexables(userId, decision, memoryRefs, memory);
                return new ApplyDecisionResult { Action = decision.Action, Route = route, Written = written, MemoryRefs = memoryRefs };
        }
    }

    /// <summary>
    /// New: logs an "ignore" outcome to a side table for later async reprocessing (no TS
    /// equivalent — see class remarks). Deliberately separate from `decisions` (which already
    /// records every raw decision) so a reprocessing worker can select a narrow, purpose-built queue.
    /// </summary>
    public string InsertIgnoredDecision(string userId, string? source, string? content, string reasoning, double? confidence, string rawJson)
    {
        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO ignored_decisions (id, user_id, source, content, reasoning, confidence, raw_json)
            VALUES ($id, $userId, $source, $content, $reasoning, $confidence, $rawJson)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$source", (object?)source ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$content", (object?)content ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$reasoning", reasoning);
        cmd.Parameters.AddWithValue("$confidence", (object?)confidence ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$rawJson", rawJson);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string InsertEpisodic(string userId, string content, MemoryPayload? memory = null)
    {
        memory ??= new MemoryPayload();
        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO episodic (
              id, user_id, content, strength, decay_rate, emotional_weight, confidence, tags,
              source_kind, source_id, source_timestamp, source_label, temporal_expression, resolved_time, resolved_time_confidence,
              promoted
            )
            VALUES ($id, $userId, $content, $strength, $decayRate, $emotionalWeight, $confidence, $tags,
              $sourceKind, $sourceId, $sourceTimestamp, $sourceLabel, $temporalExpression, $resolvedTime, $resolvedTimeConfidence, 0)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$content", content);
        cmd.Parameters.AddWithValue("$strength", memory.Strength ?? 0.75);
        cmd.Parameters.AddWithValue("$decayRate", memory.DecayRate ?? 0.02);
        cmd.Parameters.AddWithValue("$emotionalWeight", memory.EmotionalWeight ?? 0.2);
        cmd.Parameters.AddWithValue("$confidence", memory.Confidence ?? 0.8);
        cmd.Parameters.AddWithValue("$tags", JsonSerializer.Serialize(memory.Tags ?? new List<string>()));
        cmd.Parameters.AddWithValue("$sourceKind", (object?)memory.SourceKind ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceId", (object?)memory.SourceId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceTimestamp", (object?)memory.SourceTimestamp ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceLabel", (object?)memory.SourceLabel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$temporalExpression", (object?)memory.TemporalExpression ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$resolvedTime", (object?)memory.ResolvedTime ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$resolvedTimeConfidence", (object?)memory.ResolvedTimeConfidence ?? DBNull.Value);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string InsertSemantic(string userId, string content, MemoryPayload? memory = null, IReadOnlyList<string>? sourceEpisodes = null)
    {
        memory ??= new MemoryPayload();
        sourceEpisodes ??= Array.Empty<string>();
        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO semantic (
              id, user_id, content, strength, decay_rate, emotional_weight, confidence, tags, source_episodes,
              source_kind, source_id, source_timestamp, source_label, temporal_expression, resolved_time, resolved_time_confidence
            )
            VALUES ($id, $userId, $content, $strength, $decayRate, $emotionalWeight, $confidence, $tags, $sourceEpisodes,
              $sourceKind, $sourceId, $sourceTimestamp, $sourceLabel, $temporalExpression, $resolvedTime, $resolvedTimeConfidence)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$content", content);
        cmd.Parameters.AddWithValue("$strength", memory.Strength ?? 0.85);
        cmd.Parameters.AddWithValue("$decayRate", memory.DecayRate ?? 0.005);
        cmd.Parameters.AddWithValue("$emotionalWeight", memory.EmotionalWeight ?? 0.2);
        cmd.Parameters.AddWithValue("$confidence", memory.Confidence ?? 0.85);
        cmd.Parameters.AddWithValue("$tags", JsonSerializer.Serialize(memory.Tags ?? new List<string>()));
        cmd.Parameters.AddWithValue("$sourceEpisodes", JsonSerializer.Serialize(memory.SourceEpisodes ?? sourceEpisodes));
        cmd.Parameters.AddWithValue("$sourceKind", (object?)memory.SourceKind ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceId", (object?)memory.SourceId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceTimestamp", (object?)memory.SourceTimestamp ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceLabel", (object?)memory.SourceLabel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$temporalExpression", (object?)memory.TemporalExpression ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$resolvedTime", (object?)memory.ResolvedTime ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$resolvedTimeConfidence", (object?)memory.ResolvedTimeConfidence ?? DBNull.Value);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string InsertConflict(string userId, string content, string reason, string? existingMemoryId = null, string? existingMemoryType = null)
    {
        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO conflicts (id, user_id, existing_memory_id, existing_memory_type, conflicting_content, conflict_reason, status)
            VALUES ($id, $userId, $existingMemoryId, $existingMemoryType, $content, $reason, 'unresolved')
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$existingMemoryId", (object?)existingMemoryId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$existingMemoryType", (object?)existingMemoryType ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$content", content);
        cmd.Parameters.AddWithValue("$reason", reason);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string InsertDecaySchedule(string userId, string memoryKey, double decayRate)
    {
        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO decay_schedule (id, user_id, memory_key, next_decay, decay_rate)
            VALUES ($id, $userId, $memoryKey, datetime('now', '+1 day'), $decayRate)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$memoryKey", memoryKey);
        cmd.Parameters.AddWithValue("$decayRate", decayRate);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string InsertDecision(string userId, string source, string action, string route, string reasoning, string rawJson)
    {
        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO decisions (id, user_id, source, action, route, reasoning, raw_json)
            VALUES ($id, $userId, $source, $action, $route, $reasoning, $rawJson)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$source", source);
        cmd.Parameters.AddWithValue("$action", action);
        cmd.Parameters.AddWithValue("$route", route);
        cmd.Parameters.AddWithValue("$reasoning", (object?)reasoning ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$rawJson", rawJson);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string? InsertMemoryFact(string userId, MemoryFactPayload fact, WrittenMemoryRef? sourceMemory = null, MemoryPayload? source = null)
    {
        var subject = fact.Subject?.Trim();
        var predicate = NormalizePredicate(fact.Predicate);
        var valueText = !string.IsNullOrWhiteSpace(fact.ValueText) ? fact.ValueText.Trim() : ValueToText(fact.Value);
        var confidence = fact.Confidence ?? 0.75;
        if (string.IsNullOrEmpty(subject) || string.IsNullOrEmpty(predicate) || string.IsNullOrEmpty(valueText) || confidence < 0.35) return null;

        var id = Guid.NewGuid().ToString();
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO memory_facts (
              id, user_id, subject, predicate, object, value_text, value_json, fact_type, confidence,
              inference_kind, evidence_text, source_memory_table, source_memory_id, source_id, source_timestamp,
              temporal_expression, resolved_time, resolved_time_confidence
            )
            VALUES ($id, $userId, $subject, $predicate, $object, $valueText, $valueJson, $factType, $confidence,
              $inferenceKind, $evidenceText, $sourceMemoryTable, $sourceMemoryId, $sourceId, $sourceTimestamp,
              $temporalExpression, $resolvedTime, $resolvedTimeConfidence)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$subject", subject);
        cmd.Parameters.AddWithValue("$predicate", predicate);
        cmd.Parameters.AddWithValue("$object", (object?)fact.Object ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$valueText", valueText);
        cmd.Parameters.AddWithValue("$valueJson", fact.ValueJson is null ? (object)DBNull.Value : JsonSerializer.Serialize(fact.ValueJson));
        cmd.Parameters.AddWithValue("$factType", (object?)fact.FactType ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$confidence", confidence);
        cmd.Parameters.AddWithValue("$inferenceKind", (object?)fact.InferenceKind ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$evidenceText", (object?)fact.EvidenceText ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceMemoryTable", (object?)sourceMemory?.Table ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceMemoryId", (object?)sourceMemory?.Id ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceId", (object?)source?.SourceId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sourceTimestamp", (object?)source?.SourceTimestamp ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$temporalExpression", (object?)(fact.TemporalExpression ?? source?.TemporalExpression) ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$resolvedTime", (object?)(fact.ResolvedTime ?? source?.ResolvedTime) ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$resolvedTimeConfidence", (object?)(fact.ResolvedTimeConfidence ?? source?.ResolvedTimeConfidence) ?? DBNull.Value);
        cmd.ExecuteNonQuery();
        return id;
    }

    public string UpsertIndexable(string userId, IndexablePayload payload)
    {
        var id = Guid.NewGuid().ToString();
        var key = payload.Key.Trim().ToLowerInvariant();
        using (var cmd = _db.CreateCommand())
        {
            cmd.CommandText = """
                INSERT INTO indexables (
                  id, user_id, kind, key, target_memory_table, target_memory_id, steps_json, salience,
                  reconstructive_hint, evidence_text, tags
                )
                VALUES ($id, $userId, $kind, $key, $targetTable, $targetId, $steps, $salience, $hint, $evidence, $tags)
                ON CONFLICT(user_id, key) DO UPDATE SET
                  kind = excluded.kind,
                  target_memory_table = excluded.target_memory_table,
                  target_memory_id = excluded.target_memory_id,
                  steps_json = excluded.steps_json,
                  salience = excluded.salience,
                  reconstructive_hint = excluded.reconstructive_hint,
                  evidence_text = excluded.evidence_text,
                  tags = excluded.tags
                """;
            cmd.Parameters.AddWithValue("$id", id);
            cmd.Parameters.AddWithValue("$userId", userId);
            cmd.Parameters.AddWithValue("$kind", payload.Kind);
            cmd.Parameters.AddWithValue("$key", key);
            cmd.Parameters.AddWithValue("$targetTable", (object?)payload.TargetMemoryTable ?? DBNull.Value);
            cmd.Parameters.AddWithValue("$targetId", (object?)payload.TargetMemoryId ?? DBNull.Value);
            cmd.Parameters.AddWithValue("$steps", JsonSerializer.Serialize(payload.Steps ?? new List<string>()));
            cmd.Parameters.AddWithValue("$salience", payload.Salience ?? 0.8);
            cmd.Parameters.AddWithValue("$hint", (object?)payload.ReconstructiveHint ?? DBNull.Value);
            cmd.Parameters.AddWithValue("$evidence", (object?)payload.EvidenceText ?? DBNull.Value);
            cmd.Parameters.AddWithValue("$tags", JsonSerializer.Serialize(payload.Tags ?? new List<string>()));
            cmd.ExecuteNonQuery();
        }
        using var select = _db.CreateCommand();
        select.CommandText = "SELECT id FROM indexables WHERE user_id = $userId AND key = $key";
        select.Parameters.AddWithValue("$userId", userId);
        select.Parameters.AddWithValue("$key", key);
        var found = select.ExecuteScalar();
        return found?.ToString() ?? id;
    }

    public List<IndexableRecord> SelectIndexables(string userId, int limit = 100)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = "SELECT * FROM indexables WHERE user_id = $userId ORDER BY salience DESC, created_at DESC LIMIT $limit";
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$limit", limit);
        using var reader = cmd.ExecuteReader();
        var results = new List<IndexableRecord>();
        while (reader.Read()) results.Add(ReadIndexableRecord(reader));
        return results;
    }

    public MemoryRecord? GetMemory(string table, string id)
    {
        string sql = table switch
        {
            MemoryTables.Episodic => "SELECT *, 'episodic' as memory_table FROM episodic WHERE id = $id",
            MemoryTables.Semantic => "SELECT *, 'semantic' as memory_table FROM semantic WHERE id = $id",
            MemoryTables.Archival =>
                "SELECT id, user_id, content, NULL as strength, NULL as decay_rate, NULL as emotional_weight, NULL as confidence, " +
                "NULL as tags, NULL as source_episodes, NULL as source_kind, NULL as source_id, NULL as source_timestamp, " +
                "NULL as source_label, NULL as temporal_expression, NULL as resolved_time, NULL as resolved_time_confidence, " +
                "'archival' as memory_table, archived_at as created_at, NULL as last_accessed FROM archival WHERE id = $id",
            _ => throw new ArgumentException($"Unsupported table: {table}")
        };
        using var cmd = _db.CreateCommand();
        cmd.CommandText = sql;
        cmd.Parameters.AddWithValue("$id", id);
        using var reader = cmd.ExecuteReader();
        return reader.Read() ? ReadMemoryRecord(reader) : null;
    }

    public List<MemoryRecord> SelectMemories(string userId, IReadOnlyList<string>? tables = null, int limit = 100)
    {
        tables ??= new[] { MemoryTables.Semantic, MemoryTables.Episodic };
        var rows = new List<MemoryRecord>();
        foreach (var table in tables)
        {
            string? sql = table switch
            {
                MemoryTables.Episodic => "SELECT *, 'episodic' as memory_table FROM episodic WHERE user_id = $userId ORDER BY created_at DESC LIMIT $limit",
                MemoryTables.Semantic => "SELECT *, 'semantic' as memory_table FROM semantic WHERE user_id = $userId ORDER BY created_at DESC LIMIT $limit",
                MemoryTables.Archival =>
                    "SELECT id, user_id, content, NULL as strength, NULL as decay_rate, NULL as emotional_weight, NULL as confidence, " +
                    "NULL as tags, NULL as source_episodes, NULL as source_kind, NULL as source_id, NULL as source_timestamp, " +
                    "NULL as source_label, NULL as temporal_expression, NULL as resolved_time, NULL as resolved_time_confidence, " +
                    "'archival' as memory_table, archived_at as created_at, NULL as last_accessed FROM archival WHERE user_id = $userId ORDER BY archived_at DESC LIMIT $limit",
                _ => null
            };
            if (sql is null) continue;
            using var cmd = _db.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("$userId", userId);
            cmd.Parameters.AddWithValue("$limit", limit);
            using var reader = cmd.ExecuteReader();
            while (reader.Read()) rows.Add(ReadMemoryRecord(reader));
        }
        return rows;
    }

    public List<MemoryFactRecord> SelectMemoryFacts(string userId, int limit = 100)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = "SELECT * FROM memory_facts WHERE user_id = $userId ORDER BY created_at DESC LIMIT $limit";
        cmd.Parameters.AddWithValue("$userId", userId);
        cmd.Parameters.AddWithValue("$limit", limit);
        using var reader = cmd.ExecuteReader();
        var results = new List<MemoryFactRecord>();
        while (reader.Read()) results.Add(ReadMemoryFactRecord(reader));
        return results;
    }

    public void UpdateAccess(IEnumerable<RankedMemory> memories)
    {
        foreach (var memory in memories)
        {
            if (memory.Table is not (MemoryTables.Episodic or MemoryTables.Semantic)) continue;
            using var cmd = _db.CreateCommand();
            cmd.CommandText = $"UPDATE {memory.Table} SET last_accessed = CURRENT_TIMESTAMP WHERE id = $id";
            cmd.Parameters.AddWithValue("$id", memory.Id);
            cmd.ExecuteNonQuery();
        }
    }

    public bool HasDuplicateMemoryContent(string userId, string content, string sourceId)
    {
        var normalized = NormalizeContentKey(content);
        if (string.IsNullOrEmpty(normalized)) return false;
        var family = BaseSourceId(sourceId);
        var memories = SelectMemories(userId, new[] { MemoryTables.Semantic, MemoryTables.Episodic }, 1000);
        return memories.Any(memory =>
        {
            if (NormalizeContentKey(memory.Content) != normalized) return false;
            var memorySource = memory.SourceId ?? "";
            return memorySource == sourceId || memorySource == family || memorySource.StartsWith($"{family}:chunk-");
        });
    }

    public List<System.Collections.Generic.Dictionary<string, object?>> SelectConflicts(string status, int limit)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = "SELECT * FROM conflicts WHERE status = $status ORDER BY created_at DESC LIMIT $limit";
        cmd.Parameters.AddWithValue("$status", status);
        cmd.Parameters.AddWithValue("$limit", limit);
        using var reader = cmd.ExecuteReader();
        var result = new List<Dictionary<string, object?>>();
        while (reader.Read())
        {
            var row = new Dictionary<string, object?>();
            for (var i = 0; i < reader.FieldCount; i++) row[reader.GetName(i)] = reader.IsDBNull(i) ? null : reader.GetValue(i);
            result.Add(row);
        }
        return result;
    }

    public void Dispose() => _db.Dispose();

    private void Exec(string sql)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private void EnsureMemoryMetadataColumns()
    {
        (string Column, string Type)[] columns =
        {
            ("source_kind", "TEXT"), ("source_id", "TEXT"), ("source_timestamp", "TEXT"), ("source_label", "TEXT"),
            ("temporal_expression", "TEXT"), ("resolved_time", "TEXT"), ("resolved_time_confidence", "REAL")
        };
        foreach (var table in new[] { MemoryTables.Episodic, MemoryTables.Semantic })
        {
            var existing = new HashSet<string>();
            using (var cmd = _db.CreateCommand())
            {
                cmd.CommandText = $"PRAGMA table_info({table})";
                using var reader = cmd.ExecuteReader();
                while (reader.Read()) existing.Add(reader.GetString(reader.GetOrdinal("name")));
            }
            foreach (var (column, type) in columns)
            {
                if (existing.Contains(column)) continue;
                Exec($"ALTER TABLE {table} ADD COLUMN {column} {type}");
            }
        }
    }

    private void InsertDecisionFacts(string userId, StorageDecision decision, List<WrittenMemoryRef> refs)
    {
        var sourceMemory = refs.FirstOrDefault();
        if (sourceMemory is null || decision.Facts.Count == 0) return;
        foreach (var fact in decision.Facts) InsertMemoryFact(userId, fact, sourceMemory, decision.Memory);
    }

    private void InsertDecisionIndexables(string userId, StorageDecision decision, List<WrittenMemoryRef> refs, MemoryPayload memory)
    {
        var sourceMemory = refs.FirstOrDefault();
        if (sourceMemory is null || decision.Indexables.Count == 0) return;
        foreach (var row in decision.Indexables)
        {
            UpsertIndexable(userId, new IndexablePayload
            {
                Kind = row.Kind,
                Key = row.Key,
                TargetMemoryTable = row.TargetMemoryTable ?? sourceMemory.Table,
                TargetMemoryId = row.TargetMemoryId ?? sourceMemory.Id,
                Steps = row.Steps,
                Salience = row.Salience,
                ReconstructiveHint = row.ReconstructiveHint,
                EvidenceText = row.EvidenceText ?? memory.Content,
                Tags = row.Tags
            });
        }
    }

    private static MemoryPayload CloneWithSourceId(MemoryPayload memory, string source)
    {
        var clone = memory.Clone();
        clone.SourceId ??= source;
        return clone;
    }

    private static MemoryPayload WithExtraTags(MemoryPayload memory, IReadOnlyList<string> extraTags)
    {
        if (extraTags.Count == 0) return memory;
        memory.Tags = (memory.Tags ?? new List<string>()).Concat(extraTags).ToList();
        return memory;
    }

    private static string? NormalizePredicate(string? value)
    {
        if (string.IsNullOrEmpty(value)) return null;
        var cleaned = Regex.Replace(value.Trim().ToLowerInvariant(), "[^a-z0-9]+", "_").Trim('_');
        return string.IsNullOrEmpty(cleaned) ? null : cleaned;
    }

    private static string? ValueToText(object? value) => value switch
    {
        null => null,
        string s when !string.IsNullOrWhiteSpace(s) => s,
        double or int or long or float or bool => value.ToString(),
        _ => null
    };

    private static string NormalizeContentKey(string content) =>
        Regex.Replace(content.Trim().ToLowerInvariant(), @"\s+", " ");

    private static string BaseSourceId(string sourceId) =>
        Regex.Replace(sourceId, @":chunk-\d+$", "");

    private static MemoryRecord ReadMemoryRecord(SqliteDataReader reader)
    {
        string? GetStr(string name)
        {
            var ordinal = SafeOrdinal(reader, name);
            return ordinal < 0 || reader.IsDBNull(ordinal) ? null : reader.GetValue(ordinal).ToString();
        }
        double? GetNum(string name)
        {
            var ordinal = SafeOrdinal(reader, name);
            if (ordinal < 0 || reader.IsDBNull(ordinal)) return null;
            return Convert.ToDouble(reader.GetValue(ordinal));
        }
        return new MemoryRecord
        {
            Id = GetStr("id") ?? "",
            UserId = GetStr("user_id") ?? "",
            Content = GetStr("content") ?? "",
            Strength = GetNum("strength"),
            DecayRate = GetNum("decay_rate"),
            EmotionalWeight = GetNum("emotional_weight"),
            Confidence = GetNum("confidence"),
            Tags = GetStr("tags"),
            SourceEpisodes = GetStr("source_episodes"),
            SourceKind = GetStr("source_kind"),
            SourceId = GetStr("source_id"),
            SourceTimestamp = GetStr("source_timestamp"),
            SourceLabel = GetStr("source_label"),
            TemporalExpression = GetStr("temporal_expression"),
            ResolvedTime = GetStr("resolved_time"),
            ResolvedTimeConfidence = GetNum("resolved_time_confidence"),
            Table = GetStr("memory_table") ?? "",
            CreatedAt = GetStr("created_at"),
            LastAccessed = GetStr("last_accessed")
        };
    }

    private static MemoryFactRecord ReadMemoryFactRecord(SqliteDataReader reader)
    {
        string? GetStr(string name)
        {
            var ordinal = SafeOrdinal(reader, name);
            return ordinal < 0 || reader.IsDBNull(ordinal) ? null : reader.GetValue(ordinal).ToString();
        }
        double? GetNum(string name)
        {
            var ordinal = SafeOrdinal(reader, name);
            if (ordinal < 0 || reader.IsDBNull(ordinal)) return null;
            return Convert.ToDouble(reader.GetValue(ordinal));
        }
        return new MemoryFactRecord
        {
            Id = GetStr("id") ?? "",
            UserId = GetStr("user_id") ?? "",
            Subject = GetStr("subject") ?? "",
            Predicate = GetStr("predicate") ?? "",
            Object = GetStr("object"),
            ValueText = GetStr("value_text") ?? "",
            ValueJson = GetStr("value_json"),
            FactType = GetStr("fact_type"),
            Confidence = GetNum("confidence"),
            InferenceKind = GetStr("inference_kind"),
            EvidenceText = GetStr("evidence_text"),
            SourceMemoryTable = GetStr("source_memory_table"),
            SourceMemoryId = GetStr("source_memory_id"),
            SourceId = GetStr("source_id"),
            SourceTimestamp = GetStr("source_timestamp"),
            TemporalExpression = GetStr("temporal_expression"),
            ResolvedTime = GetStr("resolved_time"),
            ResolvedTimeConfidence = GetNum("resolved_time_confidence"),
            CreatedAt = GetStr("created_at"),
            UpdatedAt = GetStr("updated_at")
        };
    }

    private static IndexableRecord ReadIndexableRecord(SqliteDataReader reader)
    {
        string? GetStr(string name)
        {
            var ordinal = SafeOrdinal(reader, name);
            return ordinal < 0 || reader.IsDBNull(ordinal) ? null : reader.GetValue(ordinal).ToString();
        }
        List<string> ParseArray(string? json)
        {
            if (string.IsNullOrEmpty(json)) return new List<string>();
            try { return JsonSerializer.Deserialize<List<string>>(json) ?? new List<string>(); }
            catch { return new List<string>(); }
        }
        var salienceOrdinal = SafeOrdinal(reader, "salience");
        var salience = salienceOrdinal >= 0 && !reader.IsDBNull(salienceOrdinal) ? Convert.ToDouble(reader.GetValue(salienceOrdinal)) : 0.8;
        return new IndexableRecord
        {
            Id = GetStr("id") ?? "",
            UserId = GetStr("user_id") ?? "",
            Kind = GetStr("kind") ?? "",
            Key = GetStr("key") ?? "",
            TargetMemoryTable = GetStr("target_memory_table"),
            TargetMemoryId = GetStr("target_memory_id"),
            Steps = ParseArray(GetStr("steps_json")),
            Salience = salience,
            ReconstructiveHint = GetStr("reconstructive_hint"),
            EvidenceText = GetStr("evidence_text"),
            Tags = ParseArray(GetStr("tags")),
            CreatedAt = GetStr("created_at")
        };
    }

    private static int SafeOrdinal(SqliteDataReader reader, string name)
    {
        // Microsoft.Data.Sqlite's GetOrdinal throws ArgumentOutOfRangeException (not
        // IndexOutOfRangeException) for a column name that isn't present in the result set --
        // e.g. episodic rows have no source_episodes column, only semantic does.
        try { return reader.GetOrdinal(name); }
        catch (ArgumentOutOfRangeException) { return -1; }
        catch (IndexOutOfRangeException) { return -1; }
    }
}
