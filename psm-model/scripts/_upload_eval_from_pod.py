#!/usr/bin/env python3
"""Upload existing eval JSON from pod to HF."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

from _run_hf_lora_eval import EVAL_OUT, MODEL_REPO, _hf_token, _ns, _ssh  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--proxy-user", required=True)
    args = parser.parse_args()

    token = _hf_token()
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1

    body = f"""set -euo pipefail
cd /workspace/PSM
export HF_TOKEN={token!r}
export PSM_HF_MODEL_REPO={MODEL_REPO!r}
export HF_EVAL_OUT={EVAL_OUT!r}
python3 - <<'PY'
import os
from huggingface_hub import HfApi
from pathlib import Path
repo = os.environ["PSM_HF_MODEL_REPO"]
path = Path(os.environ["HF_EVAL_OUT"])
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(path),
    path_in_repo="eval/hf-prod-v1-qwen0.5b-prod-grounding.json",
    repo_id=repo,
    repo_type="model",
    commit_message="upload prod grounding eval report",
)
print("uploaded", path.stat().st_size, "bytes")
PY
"""
    alias, host, port, user = _ssh(args.pod_id, args.proxy_user)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
        tmp.write(body)
        path = Path(tmp.name)
    try:
        return int(
            rc._ssh_run_script(
                alias, path, host=host, port=port, user=user, timeout_sec=120, extra_env={"HF_TOKEN": token}
            )
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
