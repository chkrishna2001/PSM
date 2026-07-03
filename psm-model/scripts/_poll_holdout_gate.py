#!/usr/bin/env python3
"""Poll holdout-gate progress on RunPod via direct TCP SSH (parallel-safe)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "benchmark/locomo/results/holdout-gate-progress.jsonl"

REMOTE_PROBE = r"""
ROOT=/workspace/PSM
echo PSM_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F',' '{printf "PSM_GPU_UTIL=%s\nPSM_GPU_MEM_MIB=%s\n", $1, $2}'
pgrep -af 'holdout|ingest-psm|answer-evaluate|hf_single|evaluate\.js|runpod_holdout' 2>/dev/null | head -6 | while IFS= read -r line; do echo "PSM_PROC=$line"; done
if [[ -f /tmp/psm-holdout-gate-matrix.log ]]; then
  tail -10 /tmp/psm-holdout-gate-matrix.log | while IFS= read -r line; do echo "PSM_MATRIX_LOG=$line"; done
fi
TAGS="__TAGS__"
for tag in $TAGS; do
  log="$ROOT/benchmark/locomo/results/holdout-gate-${tag}.log"
  if [[ -f "$log" ]]; then
    tail -1 "$log" | sed "s/^/PSM_PROFILE_LOG[${tag}]=/"
  fi
  summary="$ROOT/benchmark/locomo/results/holdout-gate-${tag}-ingest-summary.json"
  if [[ -f "$summary" ]]; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('PSM_INGEST[{}]='.format(sys.argv[2])+json.dumps({k:d.get(k) for k in ('seen','stored','ignored','failed')}))" "$summary" "$tag"
  fi
  gate="$ROOT/benchmark/locomo/results/holdout-gate-${tag}.json"
  if [[ -f "$gate" ]]; then
    echo "PSM_GATE_DONE[${tag}]=1"
  fi
done
"""


def _tags(profiles: list[str], sample_ids: str) -> list[str]:
    suffix = sample_ids.replace(",", "-")
    return [f"{p.strip()}-{suffix}" for p in profiles if p.strip()]


def _direct_target(pod_id: str, proxy_user: str) -> dict[str, str] | None:
    pod = rc._fetch_pod(pod_id)
    for target in rc._pod_ssh_targets(pod_id, proxy_user=proxy_user or None):
        if target.get("mode") == "direct-tcp":
            return target
    return rc._direct_tcp_target(pod)


def _ssh_probe_direct(target: dict[str, str], script: str, timeout_sec: int = 45) -> tuple[int, str]:
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(target["port"]),
            f"{target['user']}@{target['host']}",
            "bash",
            "-s",
        ],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _parse_probe(text: str) -> dict:
    row: dict = {"matrix_log_tail": [], "procs": [], "profiles": {}}
    for line in text.splitlines():
        if not line.startswith("PSM_"):
            continue
        if line.startswith("PSM_TS="):
            row["ts"] = line.split("=", 1)[1]
        elif line.startswith("PSM_GPU_UTIL="):
            row["gpu_util_pct"] = line.split("=", 1)[1].strip()
        elif line.startswith("PSM_GPU_MEM_MIB="):
            row["gpu_mem_mib"] = line.split("=", 1)[1].strip()
        elif line.startswith("PSM_PROC="):
            row["procs"].append(line.split("=", 1)[1])
        elif line.startswith("PSM_MATRIX_LOG="):
            row["matrix_log_tail"].append(line.split("=", 1)[1])
        elif line.startswith("PSM_PROFILE_LOG["):
            key, _, val = line.partition("]=")
            tag = key[len("PSM_PROFILE_LOG[") :]
            row.setdefault("profile_log_tail", {})[tag] = val
        elif line.startswith("PSM_INGEST["):
            key, _, val = line.partition("]=")
            tag = key[len("PSM_INGEST[") :]
            prof = row.setdefault("profiles", {}).setdefault(tag, {})
            try:
                prof["ingest"] = json.loads(val)
            except json.JSONDecodeError:
                prof["ingest_raw"] = val
        elif line.startswith("PSM_GATE_DONE["):
            tag = line.split("[", 1)[1].rstrip("]=1")
            row.setdefault("profiles", {}).setdefault(tag, {})["gate_done"] = True
    return row


REMOTE_SNAPSHOT = "/workspace/PSM/benchmark/locomo/results/holdout-gate-progress.snapshot.json"


def _read_local_log(path: Path, n: int = 12) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def _fetch_snapshot(target: dict[str, str]) -> dict | None:
    rc_code, text = _ssh_probe_direct(
        target,
        f"cat {REMOTE_SNAPSHOT} 2>/dev/null || echo PSM_SNAPSHOT_MISSING\n",
        timeout_sec=30,
    )
    if rc_code != 0:
        return None
    body = text.strip()
    if not body or body == "PSM_SNAPSHOT_MISSING":
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"snapshot_raw": body[-500:]}


def poll_once(
    pod_id: str,
    proxy_user: str,
    profiles: list[str],
    sample_ids: str,
    *,
    local_log: Path | None = None,
) -> dict:
    target = _direct_target(pod_id, proxy_user)
    tags = _tags(profiles, sample_ids)
    base = {
        "pod_id": pod_id,
        "polled_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags,
        "ssh_mode": None,
        "ok": False,
    }
    if not target:
        base["error"] = "no_direct_tcp_target (public IP / port 22 mapping missing)"
        return base

    script = REMOTE_PROBE.replace("__TAGS__", " ".join(tags))
    rc_code, text = _ssh_probe_direct(target, script)
    parsed = _parse_probe(text)
    base.update(parsed)
    base["ssh_mode"] = "direct-tcp"
    base["ssh_rc"] = rc_code
    base["ok"] = rc_code == 0 and bool(parsed.get("ts") or parsed.get("procs") or parsed.get("matrix_log_tail"))
    if not base["ok"] and text.strip():
        base["ssh_tail"] = text.strip()[-500:]

    snapshot = _fetch_snapshot(target) if target else None
    if snapshot:
        base["snapshot"] = snapshot
        base["ok"] = True

    local_tail = _read_local_log(local_log) if local_log else []
    if local_tail:
        base["local_log_tail"] = local_tail
        base["ok"] = True
    return base


def _summarize(row: dict) -> str:
    gpu = row.get("gpu_util_pct") or (row.get("snapshot") or {}).get("gpu_util_pct", "?")
    mem = row.get("gpu_mem_mib") or (row.get("snapshot") or {}).get("gpu_mem_mib", "?")
    procs = len(row.get("procs") or (row.get("snapshot") or {}).get("procs") or [])
    profiles = row.get("profiles") or (row.get("snapshot") or {}).get("profiles") or {}
    done = sum(1 for p in profiles.values() if p.get("gate_done"))
    ingests = {t: p.get("ingest") for t, p in profiles.items() if p.get("ingest")}
    tail = (
        (row.get("matrix_log_tail") or (row.get("snapshot") or {}).get("matrix_log_tail") or row.get("local_log_tail") or [""])[-1]
    )[:120]
    return (
        f"[{row.get('polled_at', '')[:19]}Z] gpu={gpu}% mem={mem}MiB procs={procs} "
        f"gates_done={done}/{len(row.get('tags') or [])} ingest={ingests or '-'} | {tail}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll holdout gate progress on RunPod.")
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--profiles", default="v5n-dpo,v5n,v5h")
    parser.add_argument("--sample-ids", default="conv-30,conv-41")
    parser.add_argument("--interval", type=int, default=60, help="seconds between polls (0 = once)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--local-log", default=str(REPO / "benchmark/locomo/results/holdout-gate-run.log"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    out_path = Path(args.out)
    local_log = Path(args.local_log) if args.local_log else None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        row = poll_once(
            args.pod_id,
            args.proxy_user,
            profiles,
            args.sample_ids,
            local_log=local_log,
        )
        out_path.open("a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
        print(_summarize(row), flush=True)
        if args.once or args.interval <= 0:
            return 0 if row.get("ok") else 1
        time.sleep(max(15, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
