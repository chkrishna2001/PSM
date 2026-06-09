#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

PROXY = "juj6ykjygnjujl-64411dc8@ssh.runpod.io"
KEY = r"C:\Users\chkri\.ssh\id_ed25519"

stdin = """\
latest=$(ls -1 /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-step-*.pt 2>/dev/null | tail -1)
echo "LATEST=$latest"
test -f /tmp/psm-gate4.done && echo DONE
pgrep -af 'python3 -m psm_model.train' >/dev/null || echo NO_TRAIN
exit
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll RunPod until Gate 4 training reaches target step.")
    parser.add_argument(
        "--target-step",
        type=int,
        default=46000,
        help="Exit when latest checkpoint is real-v3-50m-full-v2-step-NNNNNN.pt at this step (default: 46000).",
    )
    parser.add_argument("--max-polls", type=int, default=90, help="Max poll iterations (default: 90).")
    parser.add_argument("--poll-sec", type=int, default=60, help="Seconds between polls (default: 60).")
    args = parser.parse_args()
    target_tag = f"step-{args.target_step:06d}"

    for _ in range(args.max_polls):
        proc = subprocess.run(
            [
                "ssh.exe",
                "-tt",
                "-i",
                KEY,
                "-o",
                "ConnectTimeout=20",
                PROXY,
                "bash",
                "-s",
            ],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        latest = ""
        done = False
        no_train = False
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("LATEST="):
                latest = line.split("=", 1)[1]
            if line == "DONE":
                done = True
            if line == "NO_TRAIN":
                no_train = True
        print(latest, "DONE" if done else "running", "NO_TRAIN" if no_train else "train_active", flush=True)
        if target_tag in latest and (done or no_train):
            print("READY_FOR_EVAL", flush=True)
            sys.exit(0)
        time.sleep(args.poll_sec)
    print("TIMEOUT", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
