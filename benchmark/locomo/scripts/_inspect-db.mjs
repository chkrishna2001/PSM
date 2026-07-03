import { readFileSync, existsSync } from "node:fs";
import { openSqliteDatabase } from "../../../dist/src/psm-core/src/sqlite.js";

const dbPath = process.argv[2] ?? "benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db";
if (!existsSync(dbPath)) {
  console.error("DB not found:", dbPath);
  process.exit(1);
}

const db = openSqliteDatabase(dbPath);
const count = (table) => {
  try {
    return db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get().n;
  } catch {
    return null;
  }
};

const report = {
  db: dbPath,
  counts: {
    episodic: count("episodic"),
    semantic: count("semantic"),
    archival: count("archival"),
    memory_embeddings: count("memory_embeddings"),
    memory_facts: count("memory_facts"),
    indexables: count("indexables"),
    decisions: count("decisions"),
    conflicts: count("conflicts")
  },
  embeddings: {
    models: db.prepare("SELECT model, COUNT(*) AS n, MIN(dimensions) AS min_dim, MAX(dimensions) AS max_dim FROM memory_embeddings GROUP BY model").all()
  },
  facts: {
    by_predicate: db.prepare("SELECT predicate, COUNT(*) AS n FROM memory_facts GROUP BY predicate ORDER BY n DESC LIMIT 10").all(),
    with_temporal: db.prepare("SELECT COUNT(*) AS n FROM memory_facts WHERE temporal_expression IS NOT NULL AND temporal_expression != ''").get().n,
    with_resolved: db.prepare("SELECT COUNT(*) AS n FROM memory_facts WHERE resolved_time IS NOT NULL AND resolved_time != ''").get().n,
    samples: db.prepare("SELECT subject, predicate, value_text, temporal_expression, resolved_time, inference_kind, evidence_text FROM memory_facts LIMIT 5").all()
  },
  episodic_temporal: {
    with_temporal_expression: db.prepare("SELECT COUNT(*) AS n FROM episodic WHERE temporal_expression IS NOT NULL AND temporal_expression != ''").get().n,
    with_resolved_time: db.prepare("SELECT COUNT(*) AS n FROM episodic WHERE resolved_time IS NOT NULL AND resolved_time != ''").get().n,
    with_source_timestamp: db.prepare("SELECT COUNT(*) AS n FROM episodic WHERE source_timestamp IS NOT NULL AND source_timestamp != ''").get().n
  },
  decisions_by_action: db.prepare("SELECT action, COUNT(*) AS n FROM decisions GROUP BY action ORDER BY n DESC").all(),
  users: db.prepare("SELECT user_id, COUNT(*) AS n FROM episodic GROUP BY user_id ORDER BY n DESC").all(),
  sample_stored: db.prepare("SELECT content, source_id, temporal_expression, resolved_time, tags FROM episodic WHERE tags LIKE '%locomo_dia_id:D1:3%' LIMIT 1").get(),
  sample_ignored_check: null
};

const summaryPath = dbPath.replace(/\.db$/i, "") + ".ingest-summary.json";
const altSummary = "benchmark/locomo/results/pod-sync/ingest-psm-model-summary.json";
for (const p of [summaryPath, altSummary, dbPath.replace(".db", "-summary.json")]) {
  if (existsSync(p)) {
    try {
      report.ingest_summary = JSON.parse(readFileSync(p, "utf8"));
      report.ingest_summary_path = p;
      break;
    } catch {
      // continue
    }
  }
}

// Grounding spot-check: memories should not be raw JSON wrappers
const jsonLike = db.prepare("SELECT COUNT(*) AS n FROM episodic WHERE content LIKE '{%' OR content LIKE '%\"action\"%'").get().n;
const emptyContent = db.prepare("SELECT COUNT(*) AS n FROM episodic WHERE trim(content) = ''").get().n;
report.quality = {
  episodic_json_like_content: jsonLike,
  episodic_empty_content: emptyContent
};

console.log(JSON.stringify(report, null, 2));
db.close();
