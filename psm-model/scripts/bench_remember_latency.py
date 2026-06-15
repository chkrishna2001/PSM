#!/usr/bin/env python3
"""Benchmark remember() latency for a PSM checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from psm_model.generate import generate_storage_json, open_generation_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--output-format", default="tagged")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = {
        "conversation": "User: Caroline said she went to an LGBTQ support group yesterday.",
        "operation": "remember",
        "source_timestamp": "1:56 pm on 8 May, 2023",
    }
    session = open_generation_session(args.checkpoint, output_format=args.output_format, device=args.device)
    use_kv_cache = not args.no_kv_cache
    generate_storage_json(
        args.checkpoint,
        payload,
        output_format=args.output_format,
        device=args.device,
        session=session,
        use_kv_cache=use_kv_cache,
    )
    timings: list[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        generate_storage_json(
            args.checkpoint,
            payload,
            output_format=args.output_format,
            device=args.device,
            session=session,
            use_kv_cache=use_kv_cache,
        )
        timings.append(time.perf_counter() - start)

    report = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "use_kv_cache": use_kv_cache,
        "runs": args.runs,
        "seconds_avg": round(statistics.mean(timings), 3),
        "seconds_p50": round(statistics.median(timings), 3),
        "seconds_p95": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 3),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
