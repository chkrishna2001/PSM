from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Nano PSM to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(json.dumps({
        "status": "export_scaffold_ready",
        "checkpoint": args.checkpoint,
        "out": args.out
    }, indent=2))


if __name__ == "__main__":
    main()

