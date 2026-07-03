#!/usr/bin/env python3
"""Inspect indexables in holdout gate SQLite DBs."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
PROFILES = ["v5n-dpo", "v5n", "v5h"]


def inspect(db_path: Path, profile: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    total = conn.execute("select count(*) from indexables").fetchone()[0]
    if total == 0:
        print("  (empty)")
        conn.close()
        return

    by_kind = Counter(
        r[0]
        for r in conn.execute("select kind, count(*) from indexables group by kind")
    )

    def pct(n: int) -> str:
        return f"{n}/{total} ({100 * n / total:.1f}%)"

    with_target = conn.execute(
        "select count(*) from indexables where target_memory_id is not null and target_memory_id != ''"
    ).fetchone()[0]
    with_hint = conn.execute(
        "select count(*) from indexables where reconstructive_hint is not null and reconstructive_hint != ''"
    ).fetchone()[0]
    with_evidence = conn.execute(
        "select count(*) from indexables where evidence_text is not null and evidence_text != ''"
    ).fetchone()[0]
    empty_steps = conn.execute(
        "select count(*) from indexables where steps_json is null or steps_json = '[]' or steps_json = ''"
    ).fetchone()[0]
    with_steps = total - empty_steps
    avg_sal = conn.execute("select avg(salience) from indexables").fetchone()[0]

    keys = [r[0] for r in conn.execute("select key from indexables")]
    dup_keys = total - len(set(keys))

    # linked episodic temporal fill
    linked_temporal = conn.execute(
        """
        select count(*)
        from indexables i
        join episodic e on e.id = i.target_memory_id
        where i.target_memory_table = 'episodic'
          and (
            (e.temporal_expression is not null and e.temporal_expression != '')
            or (e.resolved_time is not null and e.resolved_time != '')
          )
        """
    ).fetchone()[0]
    linked_ep = conn.execute(
        "select count(*) from indexables where target_memory_table = 'episodic'"
    ).fetchone()[0]

    # model-emitted indexables in decisions (rough)
    model_indexable_turns = 0
    decisions_with_idx = 0
    for (raw,) in conn.execute("select raw_json from decisions where action != 'ignore'"):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        idx = d.get("indexables")
        if isinstance(idx, list) and len(idx) > 0:
            decisions_with_idx += 1
            model_indexable_turns += len(idx)

    print(f"  by_kind: {dict(by_kind)}")
    print(f"  target_memory linked: {pct(with_target)}")
    print(f"  reconstructive_hint: {pct(with_hint)}")
    print(f"  evidence_text: {pct(with_evidence)}")
    print(f"  with workflow steps: {pct(with_steps)} | empty steps: {pct(empty_steps)}")
    print(f"  avg salience: {avg_sal:.3f}")
    print(f"  unique keys: {len(set(keys))} (rows lost to upsert dedupe: {dup_keys})")
    if linked_ep:
        print(f"  linked episodic w/ temporal: {linked_temporal}/{linked_ep} ({100*linked_temporal/linked_ep:.1f}%)")
    print(f"  decisions w/ model indexables[]: {decisions_with_idx} (explicit rows: {model_indexable_turns})")

    key_counts = Counter(keys).most_common(6)
    print("  top keys (upsert collisions):")
    for k, n in key_counts:
        suffix = "…" if len(k) > 52 else ""
        print(f"    {k[:52]}{suffix} ({n})")

    print("  samples:")
    for row in conn.execute(
        "select kind, key, reconstructive_hint, evidence_text, salience, steps_json, "
        "target_memory_table, target_memory_id from indexables order by random() limit 4"
    ):
        hint = (row["reconstructive_hint"] or "")[:70]
        ev = (row["evidence_text"] or "")[:65]
        steps = row["steps_json"] or "[]"
        try:
            steps_n = len(json.loads(steps))
        except json.JSONDecodeError:
            steps_n = "?"
        print(
            f"    [{row['kind']}] {row['key'][:48]} | sal={row['salience']:.2f} | "
            f"steps={steps_n} | tgt={row['target_memory_table'] or '-'}"
        )
        if hint:
            print(f"      hint: {hint}")
        if ev:
            print(f"      evidence: {ev}")

    conn.close()


def main() -> None:
    for profile in PROFILES:
        db = ROOT / f"holdout-gate-{profile}-conv-30-conv-41.db"
        if not db.is_file():
            print(f"=== {profile}: MISSING ===")
            continue
        total = sqlite3.connect(db).execute("select count(*) from indexables").fetchone()[0]
        stored = json.loads(
            (ROOT / f"holdout-gate-{profile}-conv-30-conv-41-ingest-summary.json").read_text(encoding="utf-8")
        ).get("stored", "?")
        print(f"=== {profile} stored_turns={stored} indexable_rows={total} ratio={total/stored:.1f}x ===")
        inspect(db, profile)
        print()


if __name__ == "__main__":
    main()
