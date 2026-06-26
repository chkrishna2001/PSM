from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from prod_memory.build_minimal_fixture_rows import build_minimal_fixture_rows
from prod_memory.curriculum_sources import build_noise_rows, load_fixture_cases


def _dup_rows(rows: list[dict[str, Any]], *, prefix: str, copies: int) -> list[dict[str, Any]]:
    if copies <= 0:
        return []
    out: list[dict[str, Any]] = []
    for copy_idx in range(copies):
        for row in rows:
            cloned = dict(row)
            cloned["id"] = f"{prefix}-{copy_idx:03d}-{row['id']}"
            cloned["source"] = f"{prefix}:{row.get('source', 'row')}"
            out.append(cloned)
    return out


def storage_row_to_binary(row: dict[str, Any]) -> dict[str, Any]:
    action = str((row.get("expected") or {}).get("action") or "ignore").lower()
    cloned = dict(row)
    if action in {"ignore", "ignore_noise"}:
        cloned["expected"] = {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "indexables": [],
            "reasoning": "No durable memory.",
        }
    else:
        cloned["expected"] = {
            "action": "store_episodic",
            "memory": {"content": "classify-store", "type": "episodic"},
            "facts": [],
            "indexables": [],
            "reasoning": "Durable information present.",
        }
    return cloned


def build_binary_fixture_rows(fixture_ids: list[str], *, fixtures_path: Path | None = None) -> list[dict]:
    rows = build_minimal_fixture_rows(fixture_ids, fixtures_path=fixtures_path)
    for row in rows:
        action = str(row["expected"].get("action") or "ignore")
        if action != "ignore":
            row["expected"] = {
                "action": "store_episodic",
                "memory": {
                    "content": "classify-store",
                    "type": "episodic",
                    "strength": 0.86,
                    "decay_rate": 0.02,
                    "emotional_weight": 0.22,
                    "confidence": 0.92,
                    "tags": [],
                },
                "facts": [],
                "indexables": [],
                "reasoning": "Durable information present.",
            }
    return rows


def write_binary_fixture_jsonl(out_path: Path, fixture_ids: list[str], *, fixtures_path: Path | None = None) -> int:
    rows = build_binary_fixture_rows(fixture_ids, fixtures_path=fixtures_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_v5k_gate_rows(*, locomo_path: Path | None = None) -> list[dict[str, Any]]:
    """Binary ignore/store curriculum only — no extraction content (fresh gate LoRA)."""
    ids = [str(case["id"]) for case in load_fixture_cases()]
    rows = build_binary_fixture_rows(ids)
    rows = _dup_rows(rows, prefix="gkfix", copies=14)
    rows.extend(_dup_rows(build_noise_rows(), prefix="gknoi", copies=24))
    if locomo_path and locomo_path.is_file():
        for line in locomo_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            action = str((row.get("expected") or {}).get("action") or "").lower()
            if action not in {"ignore", "ignore_noise"}:
                continue
            rows.extend(_dup_rows([storage_row_to_binary(row)], prefix="gkloc", copies=12))
    return rows


def build_v5k_gate_fixture_only_rows() -> list[dict[str, Any]]:
    """Ten prod fixtures only — heavy ignore copies for the 2 noise eval cases."""
    ids = [str(case["id"]) for case in load_fixture_cases()]
    seed = build_binary_fixture_rows(ids)
    store = [row for row in seed if str(row["expected"].get("action") or "").lower() not in {"ignore", "ignore_noise"}]
    ignore = [row for row in seed if str(row["expected"].get("action") or "").lower() in {"ignore", "ignore_noise"}]
    rows: list[dict[str, Any]] = list(seed)
    rows.extend(_dup_rows(store, prefix="gfst", copies=19))
    rows.extend(_dup_rows(ignore, prefix="gfig", copies=59))
    return rows


def _load_distill_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.is_file():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def build_v5k_gate_distill_rows(*, cache_path: Path | None = None) -> list[dict[str, Any]]:
    """Fixture GT + Claude-generated ignore diversity (binary gate distill)."""
    from prod_memory.binary_gate_teacher import binary_ignore_row

    root = Path(__file__).resolve().parents[1]
    cache = _load_distill_cache(cache_path or root / "data" / "v5k-gate-distill-cache.json")

    ids = [str(case["id"]) for case in load_fixture_cases()]
    seed = build_binary_fixture_rows(ids)
    store = [row for row in seed if str(row["expected"].get("action") or "").lower() not in {"ignore", "ignore_noise"}]
    ignore = [row for row in seed if str(row["expected"].get("action") or "").lower() in {"ignore", "ignore_noise"}]

    synth_ignore: list[dict[str, Any]] = []
    for idx, text in enumerate(cache.get("noise_variants") or []):
        synth_ignore.append(
            binary_ignore_row(f"distill-noise-{idx:03d}", str(text).strip(), source="gate_distill_noise")
        )
    for idx, text in enumerate(cache.get("fallback_noise") or []):
        synth_ignore.append(
            binary_ignore_row(f"distill-fb-{idx:03d}", str(text).strip(), source="gate_distill_fallback")
        )

    rows: list[dict[str, Any]] = list(seed)
    rows.extend(_dup_rows(store, prefix="gdst", copies=11))
    rows.extend(_dup_rows(ignore, prefix="gdig", copies=49))
    rows.extend(_dup_rows(synth_ignore, prefix="gdsn", copies=14))
    rows.extend(_dup_rows([storage_row_to_binary(r) for r in build_noise_rows()], prefix="gdno", copies=8))
    return rows


def build_v5k_gate_dpo_rows(*, cache_path: Path | None = None) -> list[dict[str, Any]]:
    """Preference pairs: chosen/rejected binary gate labels (noise + fixtures)."""
    from prod_memory.hf_prompts import storage_inference_messages

    cache = _load_distill_cache(cache_path or Path(__file__).resolve().parents[1] / "data" / "v5k-gate-distill-cache.json")
    rows: list[dict[str, Any]] = []

    def add_pair(row_id: str, text: str, chosen: str, rejected: str, source: str) -> None:
        prompt = storage_inference_messages(text.strip(), output_format="binary")
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "source": source,
            }
        )

    for case in load_fixture_cases():
        text = str(case["llmResponse"])
        cid = str(case["id"])
        exp = str(case.get("expectAction") or "store")
        if exp == "ignore":
            add_pair(f"dpo-fix-{cid}", text, "ignore", "store", "fixture")
        else:
            add_pair(f"dpo-fix-{cid}", text, "store", "ignore", "fixture")

    noise_texts = list(cache.get("noise_variants") or []) + list(cache.get("fallback_noise") or [])
    for idx, text in enumerate(noise_texts):
        add_pair(f"dpo-noise-{idx:03d}", str(text), "ignore", "store", "gate_dpo_noise")

    seed = list(rows)
    store_pairs = [r for r in seed if r["chosen"][0]["content"] == "store"]
    ignore_pairs = [r for r in seed if r["chosen"][0]["content"] == "ignore"]
    rows.extend(_dup_rows(store_pairs, prefix="dpos", copies=4))
    rows.extend(_dup_rows(ignore_pairs, prefix="dpoi", copies=19))
    return rows
