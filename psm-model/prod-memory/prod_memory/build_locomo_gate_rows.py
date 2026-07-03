"""LoCoMo → binary gate curriculum rows (QA evidence = store, else ignore)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from prod_memory.binary_gate_teacher import binary_ignore_row, binary_store_row

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO / "benchmark/locomo/data/locomo10.json"


def _quote_text(value: str | None) -> str:
    text = (value or "").strip()
    return f'"{text}"' if text else '""'


def _source_timestamp(sample: dict[str, Any], session: str | None) -> str | None:
    if not session:
        return None
    conv = sample.get("conversation") or {}
    date_time = conv.get(f"{session}_date_time")
    if isinstance(date_time, str) and date_time.strip():
        return date_time.strip()
    match = re.match(r"^session_(\d+)$", session)
    if not match:
        return None
    events = (sample.get("event_summary") or {}).get(f"events_session_{match.group(1)}") or {}
    date = events.get("date")
    return date.strip() if isinstance(date, str) and date.strip() else None


def flatten_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conv = sample.get("conversation") or {}
    turns: list[dict[str, Any]] = []
    for key in sorted(
        (k for k in conv if re.match(r"^session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    ):
        session_turns = conv.get(key)
        if not isinstance(session_turns, list):
            continue
        for turn in session_turns:
            if isinstance(turn, dict):
                turns.append({**turn, "session": key})
    return turns


def build_remember_text(sample: dict[str, Any], turns: list[dict[str, Any]], index: int) -> str:
    """Same shape as LoCoMo ingest / probe (window=2)."""
    turn = turns[index]
    sample_id = str(sample.get("sample_id") or "unknown")
    session = str(turn.get("session") or "")
    dia_id = str(turn.get("dia_id") or "")
    source_timestamp = _source_timestamp(sample, session) or "unknown"
    window_start = max(0, index - 2)
    nearby = turns[window_start:index]

    def prior_line(item: dict[str, Any]) -> str:
        bits = [f'{item.get("speaker") or "Unknown"} said: {_quote_text(str(item.get("text") or ""))}']
        if item.get("query"):
            bits.append(f'image query: {item["query"]}')
        if item.get("blip_caption"):
            bits.append(f'image caption: {item["blip_caption"]}')
        return f'- [prior {item.get("session") or "unknown"} {item.get("dia_id") or "unknown"}] {"; ".join(bits)}'

    image_lines: list[str] = []
    if turn.get("query"):
        image_lines.append(f'Image query: {turn["query"]}')
    if turn.get("blip_caption"):
        image_lines.append(f'Image caption: {turn["blip_caption"]}')
    if turn.get("img_url"):
        image_lines.append(f'Image URLs: {", ".join(map(str, turn["img_url"]))}')

    prior_lines = [prior_line(item) for item in nearby] if nearby else ["- none"]
    lines = [
        f"Source id: {sample_id}:{dia_id}",
        f"Sample id: {sample_id}",
        f"Session: {session or 'unknown'}",
        f"Session time: {source_timestamp}",
        f'Current speaker: {turn.get("speaker") or "unknown"}',
        f'Current utterance: {_quote_text(str(turn.get("text") or ""))}',
        *image_lines,
        "Previous context:",
        *prior_lines,
    ]
    return "\n".join(lines)


def evidence_dia_ids(sample: dict[str, Any]) -> set[str]:
    ev: set[str] = set()
    for qa in sample.get("qa") or []:
        for e in qa.get("evidence") or []:
            ev.add(str(e))
    return ev


def build_locomo_gate_binary_rows(
    data_path: Path = DEFAULT_DATA,
    *,
    neg_per_pos: int = 2,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """All QA-evidence turns → store; sample neg_per_pos non-evidence per conv → ignore."""
    rng = random.Random(seed)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "unknown")
        turns = flatten_turns(sample)
        ev = evidence_dia_ids(sample)
        pos_idx = [i for i, t in enumerate(turns) if str(t.get("dia_id")) in ev]
        neg_idx = [i for i, t in enumerate(turns) if str(t.get("dia_id")) not in ev]
        rng.shuffle(neg_idx)
        neg_take = min(len(neg_idx), max(1, len(pos_idx) * neg_per_pos))
        for idx in pos_idx:
            dia = str(turns[idx].get("dia_id"))
            text = build_remember_text(sample, turns, idx)
            rows.append(
                binary_store_row(
                    f"locomo-gate-{sample_id}-{dia.replace(':', '_')}",
                    text,
                    source="locomo_qa_evidence",
                )
            )
        for idx in neg_idx[:neg_take]:
            dia = str(turns[idx].get("dia_id"))
            text = build_remember_text(sample, turns, idx)
            rows.append(
                binary_ignore_row(
                    f"locomo-gate-{sample_id}-{dia.replace(':', '_')}-neg",
                    text,
                    source="locomo_non_evidence",
                )
            )
    return rows


def build_locomo_v5m_store_rows(
    data_path: Path = DEFAULT_DATA,
    *,
    max_rows: int = 120,
) -> list[dict[str, Any]]:
    """QA-evidence turns only → JSON store rows (LoCoMo slice for v5m, no teacher)."""
    from prod_memory.build_minimal_fixture_rows import grounded_store_content
    from prod_memory.curriculum_sources import _store_expected, remember_input

    if not data_path.is_file():
        return []
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "unknown")
        turns = flatten_turns(sample)
        ev = evidence_dia_ids(sample)
        for idx, turn in enumerate(turns):
            if str(turn.get("dia_id")) not in ev:
                continue
            dia = str(turn.get("dia_id") or "")
            text = build_remember_text(sample, turns, idx)
            utterance = str(turn.get("text") or "").strip()
            speaker = str(turn.get("speaker") or "")
            key_tokens = [t for t in (speaker, utterance[:48]) if t]
            content = grounded_store_content(text, key_tokens)
            row_id = f"locomo-v5m-{sample_id}-{dia.replace(':', '_')}"
            rows.append({
                "id": row_id,
                "input": remember_input(text, source_id=row_id, source_kind="locomo_dialogue"),
                "expected": _store_expected(
                    text,
                    content,
                    tags=[f"locomo:{sample_id}", f"locomo_dia:{dia}"],
                    reasoning=content[:160],
                    memory_type="episodic",
                ),
                "source": "exp_a_locomo_v5m",
            })
            if len(rows) >= max_rows:
                return rows
    return rows


def build_v5l_gate_rows(*, locomo_path: Path = DEFAULT_DATA) -> list[dict[str, Any]]:
    """Fixture anchors + LoCoMo evidence store/ignore (fixes v5k gate missing store labels)."""
    from prod_memory.build_binary_fixture_rows import (
        _dup_rows,
        build_binary_fixture_rows,
        build_noise_rows,
    )
    from prod_memory.curriculum_sources import load_fixture_cases

    ids = [str(case["id"]) for case in load_fixture_cases()]
    rows = build_binary_fixture_rows(ids)
    rows = _dup_rows(rows, prefix="v5lfx", copies=8)
    rows.extend(_dup_rows(build_noise_rows(), prefix="v5lnoi", copies=12))
    locomo = build_locomo_gate_binary_rows(locomo_path)
    store = [r for r in locomo if str(r["expected"].get("action") or "").lower() not in {"ignore", "ignore_noise"}]
    ignore = [r for r in locomo if str(r["expected"].get("action") or "").lower() in {"ignore", "ignore_noise"}]
    rows.extend(_dup_rows(store, prefix="v5lstore", copies=3))
    rows.extend(_dup_rows(ignore, prefix="v5lign", copies=2))
    return rows
