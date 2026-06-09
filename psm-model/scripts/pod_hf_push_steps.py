#!/usr/bin/env python3
"""Upload step checkpoint triples to HF from pod."""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

root = Path("/workspace/PSM")
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
steps = [int(x) for x in os.environ.get("STEPS", "42000,42400").split(",") if x.strip()]
stem = "real-v3-50m-full-v2"
api = HfApi()
uploaded = []
for step in steps:
    for suffix in (".pt", ".tokenizer.json", ".meta.json"):
        local = root / f"psm-model/checkpoints/{stem}-step-{step:06d}{suffix}"
        remote = f"psm-model/checkpoints/{stem}-step-{step:06d}{suffix}"
        if not local.is_file():
            print(json.dumps({"error": "missing_local", "path": str(local)}), file=sys.stderr)
            sys.exit(1)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type="model")
        uploaded.append(remote)
        print(json.dumps({"uploaded": remote, "bytes": local.stat().st_size}))
print(json.dumps({"ok": True, "count": len(uploaded)}))
