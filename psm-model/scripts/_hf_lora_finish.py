#!/usr/bin/env python3
"""Upload HF LoRA adapter + run prod fixture eval on pod."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ADAPTER = "psm-model/prod-memory/checkpoints/hf-prod-v1-qwen0.5b/adapter"
EVAL_OUT = "psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json"
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"


def _ns(pod_id: str, proxy_user: str) -> argparse.Namespace:
    return argparse.Namespace(
        pod_id=pod_id,
        proxy_user=proxy_user,
        deploy=False,
        host_alias="runpod-psm-proxy",
        name="",
        image="",
        template="",
        gpu="",
        volume_gb=0,
        container_disk_gb=0,
        autostart=False,
        wait_ssh=0,
        ssh_ready_timeout_sec=300,
        auto_gpu=False,
    )


def _ssh(pod_id: str, proxy_user: str) -> tuple[str, str, str, str]:
    _, host, port, user = rc._resolve_train_pod_ssh(_ns(pod_id, proxy_user), proxy_user=proxy_user)
    return "runpod-psm-proxy", host, port, user


def _run_remote_script(
    pod_id: str,
    proxy_user: str,
    body: str,
    *,
    timeout_sec: int,
    extra_env: dict[str, str] | None = None,
) -> int:
    alias, host, port, user = _ssh(pod_id, proxy_user)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
        tmp.write(body)
        path = Path(tmp.name)
    try:
        return int(
            rc._ssh_run_script(
                alias,
                path,
                host=host,
                port=port,
                user=user,
                timeout_sec=timeout_sec,
                extra_env=extra_env,
            )
        )
    finally:
        path.unlink(missing_ok=True)


def _hf_token() -> str:
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    subprocess.run(["o", "krishnachhftoken"], check=False, capture_output=True)
    return os.environ.get("HF_TOKEN", "").strip()


def _push_eval_files(pod_id: str, proxy_user: str) -> None:
    alias, host, port, user = _ssh(pod_id, proxy_user)
    rc._push_repo_files_via_tar(
        alias,
        REPO,
        [
            "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
            "psm-model/prod-memory/prod_memory/hf_prompts.py",
            "psm-model/src/psm_model/hf_lora_train.py",
            "psm-model/src/psm_model/lean_format.py",
            "psm-model/src/psm_model/prompts.py",
            "psm-model/src/psm_model/remember_cli.py",
            "psm-model/prod-memory/prod_memory/eval_grounding.py",
            "psm-model/prod-memory/prod_memory/grounding.py",
        ],
        "/workspace/PSM",
        host=host,
        port=port,
        user=user,
    )


def cmd_upload(pod_id: str, proxy_user: str) -> int:
    script = f"""set -euo pipefail
cd /workspace/PSM
export HF_TOKEN="${{HF_TOKEN:?HF_TOKEN missing}}"
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo = "{MODEL_REPO}"
out = Path("{ADAPTER}")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True)
for path in sorted(out.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(out).as_posix()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"hf-prod-v1-qwen0.5b/{{rel}}",
        repo_id=repo,
        repo_type="model",
        commit_message=f"upload {{rel}}",
    )
    print("uploaded", rel)
print("done", repo)
PY
touch /tmp/psm-hf-lora.done
"""
    token = _hf_token()
    if not token:
        print("HF_TOKEN missing — run: o krishnachhftoken", file=sys.stderr)
        return 1
    return _run_remote_script(
        pod_id, proxy_user, script, timeout_sec=600, extra_env={"HF_TOKEN": token}
    )


def cmd_eval(pod_id: str, proxy_user: str) -> int:
    _push_eval_files(pod_id, proxy_user)
    script = f"""set -euo pipefail
cd /workspace/PSM
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1
python -m prod_memory.eval_hf_grounding \\
  --adapter-dir {ADAPTER} \\
  --model qwen0.5b \\
  --device cuda \\
  --output-format tagged \\
  --checkpoint-label hf-prod-v1-qwen0.5b \\
  --out {EVAL_OUT}
"""
    return _run_remote_script(pod_id, proxy_user, script, timeout_sec=900)


def cmd_pull_eval(pod_id: str, proxy_user: str) -> int:
    alias, host, port, user = _ssh(pod_id, proxy_user)
    local = REPO / EVAL_OUT
    local.parent.mkdir(parents=True, exist_ok=True)
    remote_path = f"/workspace/PSM/{EVAL_OUT}"
    probe = f"""
if [[ -f '{remote_path}' ]]; then
  echo PSM_JSON_BEGIN
  cat '{remote_path}'
  echo
  echo PSM_JSON_END
else
  echo PSM_JSON_MISSING
fi
"""
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-tt",
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            *rc._ssh_endpoint(alias, host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        input=f"{probe}exit\n",
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    if "PSM_JSON_MISSING" in proc.stdout:
        print(f"missing on pod: {remote_path}", file=sys.stderr)
        return 1
    begin = proc.stdout.find("PSM_JSON_BEGIN")
    end = proc.stdout.find("PSM_JSON_END")
    if begin < 0 or end <= begin:
        print("could not parse eval json from ssh output", file=sys.stderr)
        return 1
    payload = proc.stdout[begin + len("PSM_JSON_BEGIN") : end].strip()
    local.write_text(payload + "\n", encoding="utf-8")
    report = json.loads(payload)
    agg = report.get("aggregate", {})
    print(json.dumps({"checkpoint": report.get("checkpoint"), "aggregate": agg}, indent=2))
    for case in report.get("cases", [])[:3]:
        print(
            f"  {case.get('case_id')}: parse={case.get('parse_valid')} "
            f"stored={case.get('effective_stored')} raw={str(case.get('raw_output', ''))[:120]!r}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="ymb1lfyvf5kgoz")
    parser.add_argument("--proxy-user", default="ymb1lfyvf5kgoz-64410f25")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--pull-only", action="store_true")
    args = parser.parse_args()

    if args.pull_only:
        return cmd_pull_eval(args.pod_id, args.proxy_user)

    rc = 0
    if not args.eval_only:
        print("uploading adapter to HF...", flush=True)
        rc = cmd_upload(args.pod_id, args.proxy_user)
        if rc != 0:
            print(f"upload failed exit={rc}", file=sys.stderr)
            return rc

    if not args.upload_only:
        print("running prod fixture eval...", flush=True)
        rc = cmd_eval(args.pod_id, args.proxy_user)
        if rc != 0:
            print(f"eval failed exit={rc}", file=sys.stderr)
            return rc
        return cmd_pull_eval(args.pod_id, args.proxy_user)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
