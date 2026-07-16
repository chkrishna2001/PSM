#!/usr/bin/env python3
"""Run the 100-case coding-agent gate ON THE POD's GPU, then optionally delete the pod.

The gate takes ~45 min on a laptop CPU vs ~2-3 min on the pod GPU we're already renting. Doing it
on-pod means we get the score sooner and can delete immediately after, instead of paying a pod to
idle (or deleting it and then waiting out a CPU run).

    python psm-model/scripts/_run_gate_on_pod.py --pod-id <id> --profile storage-v17 --step 900 \
        [--delete-after] [--no-upload]

Pulls the report back via the HF dataset repo (the pod already has HF_TOKEN), so no scp needed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"

PROFILE_OUT = {
    "storage-v16b-06b": "hf-prod-storage-v16b-qwen0.6b",
    "storage-v17": "hf-prod-storage-v17-qwen0.5b",
    "storage-v16b": "hf-prod-storage-v16b-qwen0.5b",
    "storage-v16": "hf-prod-storage-v16-qwen0.5b",
    "storage-v15": "hf-prod-storage-v15-qwen0.5b",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", required=True)
    ap.add_argument("--proxy-user", default="")
    ap.add_argument("--profile", required=True, choices=sorted(PROFILE_OUT))
    ap.add_argument("--step", type=int, required=True, help="checkpoint step, e.g. 900")
    ap.add_argument("--fixtures", default="psm-model/prod-memory/fixtures/holdout-coding-agent-cases.json")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--delete-after", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dataset-repo", default="krishnach7262/psm-prod-memory-data")
    args = ap.parse_args()

    prefix = PROFILE_OUT[args.profile]
    adapter = f"psm-model/prod-memory/checkpoints/{prefix}/checkpoint-{args.step}"
    out_name = f"{args.profile}-ckpt{args.step}-coding-agent-eval.json"
    out = f"psm-model/prod-memory/results/{out_name}"

    ns = argparse.Namespace(pod_id=args.pod_id, host_alias="runpod-psm-proxy",
                            autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False)
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=args.proxy_user or None)

    # `--sync-code` at deploy time only tar-pushes a fixed file list for TRAINING; the gate fixture
    # is not in it, so the pod's /workspace/PSM is missing it ("EVAL FAILED: missing ...fixtures/
    # holdout-coding-agent-cases.json" — found the hard way). Push whatever fixture file is being
    # used here explicitly before running the eval.
    print(f"pushing fixture file to pod: {args.fixtures}", flush=True)
    push_rc = rc._push_repo_files_via_tar(
        "runpod-psm-proxy", REPO, [args.fixtures], "/workspace/PSM",
        host=host, port=port, user=user,
    )
    if push_rc != 0:
        print(f"failed to push fixture file (rc={push_rc}) — aborting, pod left running", file=sys.stderr)
        return 1

    # HF_TOKEN already lives in the pod env (set at deploy), so the upload works without passing it.
    extra = {
        "GATE_ADAPTER_DIR": adapter,
        "GATE_FIXTURES": args.fixtures,
        "GATE_OUT": out,
        "GATE_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "GATE_UPLOAD": "0" if args.no_upload else "1",
        "PSM_HF_DATASET_REPO": args.dataset_repo,
        "PSM_REPO_ROOT": "/workspace/PSM",
    }
    print(f"running gate on pod {args.pod_id}: {adapter}", flush=True)
    # `_ssh_run_script` returns an int exit code on a failed REMOTE command — it does not raise —
    # so checking only for a python exception (the earlier version of this script) silently treated
    # every remote failure as success and deleted the pod anyway. Check the return value instead.
    eval_rc = 1
    try:
        eval_rc = rc._ssh_run_script(
            "runpod-psm-proxy",
            SCRIPTS / "runpod_gate_eval.sh",
            host=host, port=port, user=user,
            timeout_sec=1800,
            extra_env=extra,
            skip_ssh_wait=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"gate eval raised: {e}", file=sys.stderr)
    ok = (eval_rc == 0)

    # Only delete on success. Deleting after a failed eval throws away the one pod that could retry
    # it (that mistake cost a full re-provision once already) — leave it up so the run can be
    # re-attempted, and say so loudly rather than silently idle-billing.
    if args.delete_after and ok:
        print(f"deleting pod {args.pod_id}", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / "runpod_ctl.py"), "delete-pod",
                        args.pod_id, "--force-delete-pod"], cwd=REPO, check=False)
    elif args.delete_after and not ok:
        print(f"\n!! gate eval FAILED (rc={eval_rc}) — pod {args.pod_id} left RUNNING (and billing) "
              f"so you can retry.\n   retry: python psm-model/scripts/_run_gate_on_pod.py --pod-id "
              f"{args.pod_id} --profile {args.profile} --step {args.step} --delete-after\n"
              f"   or delete: python psm-model/scripts/runpod_ctl.py delete-pod {args.pod_id} "
              f"--force-delete-pod", file=sys.stderr)
        return 1
    if not ok:
        return 1
    print(f"\nreport uploaded as prod-memory/results/{out_name} (pull from {args.dataset_repo})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
