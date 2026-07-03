import json
import os
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()


def count(table: str):
    try:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.Error:
        return None


report = {
    "db": DB,
    "counts": {
        t: count(t)
        for t in [
            "episodic",
            "semantic",
            "archival",
            "memory_embeddings",
            "memory_facts",
            "indexables",
            "decisions",
            "conflicts",
        ]
    },
}

c.execute(
    "SELECT model, COUNT(*) AS n, MIN(dimensions) AS min_dim, MAX(dimensions) AS max_dim "
    "FROM memory_embeddings GROUP BY model"
)
report["embedding_models"] = [dict(r) for r in c.fetchall()]

c.execute("SELECT predicate, COUNT(*) AS n FROM memory_facts GROUP BY predicate ORDER BY n DESC LIMIT 10")
report["fact_predicates"] = [dict(r) for r in c.fetchall()]

report["facts_with_temporal"] = c.execute(
    "SELECT COUNT(*) FROM memory_facts WHERE temporal_expression IS NOT NULL AND temporal_expression != ''"
).fetchone()[0]
report["facts_with_resolved"] = c.execute(
    "SELECT COUNT(*) FROM memory_facts WHERE resolved_time IS NOT NULL AND resolved_time != ''"
).fetchone()[0]

c.execute(
    "SELECT subject, predicate, value_text, temporal_expression, resolved_time, inference_kind, "
    "substr(evidence_text, 1, 80) AS evidence_text FROM memory_facts LIMIT 8"
)
report["fact_samples"] = [dict(r) for r in c.fetchall()]

report["episodic_temporal_expr"] = c.execute(
    "SELECT COUNT(*) FROM episodic WHERE temporal_expression IS NOT NULL AND temporal_expression != ''"
).fetchone()[0]
report["episodic_resolved"] = c.execute(
    "SELECT COUNT(*) FROM episodic WHERE resolved_time IS NOT NULL AND resolved_time != ''"
).fetchone()[0]
report["episodic_source_ts"] = c.execute(
    "SELECT COUNT(*) FROM episodic WHERE source_timestamp IS NOT NULL AND source_timestamp != ''"
).fetchone()[0]

c.execute("SELECT action, COUNT(*) AS n FROM decisions GROUP BY action ORDER BY n DESC")
report["decisions"] = [dict(r) for r in c.fetchall()]

c.execute("SELECT user_id, COUNT(*) AS n FROM episodic GROUP BY user_id ORDER BY n DESC")
report["users_episodic"] = [dict(r) for r in c.fetchall()]

c.execute(
    "SELECT content, source_id, temporal_expression, resolved_time, tags FROM episodic "
    "WHERE tags LIKE '%locomo_dia_id:D1:3%' LIMIT 1"
)
row = c.fetchone()
report["d1_3_memory"] = dict(row) if row else None
if report["d1_3_memory"]:
    report["d1_3_memory"]["content"] = (report["d1_3_memory"].get("content") or "")[:220]

report["json_like_episodic"] = c.execute("SELECT COUNT(*) FROM episodic WHERE content LIKE '{%'").fetchone()[0]
report["wrapper_like_episodic"] = c.execute(
    "SELECT COUNT(*) FROM episodic WHERE content LIKE '%Current utterance:%'"
).fetchone()[0]

for path in [
    "benchmark/locomo/results/pod-sync/ingest-psm-model-summary.json",
    DB.replace(".db", "-summary.json"),
]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            report["ingest_summary"] = json.load(f)
        report["ingest_summary_path"] = path
        break

print(json.dumps(report, indent=2))
conn.close()
