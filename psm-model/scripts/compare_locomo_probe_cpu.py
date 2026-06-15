from __future__ import annotations

import json
from pathlib import Path

from psm_model.generate import generate_storage_json, open_generation_session


def main() -> int:
    ckpt = Path("psm-model/checkpoints/real-v3-50m-full-v2-step-058000.pt")
    session = open_generation_session(ckpt, output_format="tagged", device="cpu")
    cases = {
        "locomo_d1_3": {
            "conversation": (
                'User: Earlier in the conversation: Caroline said "Hey Mel!"; Melanie said "Hey Caroline!". '
                'Session time: 1:56 pm on 8 May, 2023. Caroline said "I went to a LGBTQ support group yesterday and it was so powerful.".'
            ),
            "operation": "remember",
            "source_timestamp": "1:56 pm on 8 May, 2023",
        },
        "probe_simple": {
            "conversation": "User: Caroline said she went to an LGBTQ support group yesterday and it was powerful.",
            "operation": "remember",
            "source_timestamp": "1:56 pm on 8 May, 2023",
        },
        "probe_gate": {
            "conversation": "User: Today I met Dana at 3pm to review the PSM roadmap.",
            "operation": "remember",
            "source_timestamp": "2026-06-05T12:00:00Z",
        },
    }
    out = {}
    for name, payload in cases.items():
        raw = generate_storage_json(
            ckpt,
            payload,
            output_format="tagged",
            device="cpu",
            session=session,
        )
        out[name] = raw
        print(f"=== {name} ===")
        print(raw)
        print()
    Path("benchmark/locomo/results/cpu-probe-compare.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
