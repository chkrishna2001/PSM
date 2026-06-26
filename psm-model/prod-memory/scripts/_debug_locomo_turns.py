#!/usr/bin/env python3
"""Debug two-pass path on specific LoCoMo turns."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.eval_classify import binary_predicts_store  # noqa: E402
from prod_memory.eval_hf_grounding import open_hf_session  # noqa: E402
from prod_memory.eval_hf_locomo import _flatten_turns, _product_text  # noqa: E402
from prod_memory.grounding import apply_storage_guards, stored_text_from_decision, would_model_store  # noqa: E402
from psm_model.remember_cli import apply_product_boundary  # noqa: E402

data = json.loads(
    urllib.request.urlopen(
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        timeout=60,
    ).read()
)
binary = open_hf_session(
    PACKAGE_ROOT / "checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter",
    device="cpu",
)
extract = open_hf_session(PACKAGE_ROOT / "checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter", device="cpu")
by_id = {str(t.get("dia_id")): t for t in _flatten_turns(data[0])}

for did in sys.argv[1:] or ["D1:3", "D1:5", "D1:7", "D1:12"]:
    text = _product_text(by_id[did])
    rb = binary.generate(text, output_format="binary", max_new_tokens=16)
    if not binary_predicts_store(rb):
        print(did, "GATE_IGNORE", rb.strip()[:60])
        continue
    raw = extract.generate(text, output_format="minimal_extract", max_new_tokens=384)
    report = apply_product_boundary(raw, output_format="minimal")
    dec = report.get("parsed") or {}
    content = stored_text_from_decision(dec)
    guarded = apply_storage_guards(text, dec)
    print(
        did,
        "action=",
        dec.get("action"),
        "would_store=",
        would_model_store(dec),
        "guard=",
        guarded["rejected"],
        "repair=",
        report.get("repair_status"),
        "content=",
        (content or "")[:80],
        "raw=",
        raw.strip()[:80],
    )
