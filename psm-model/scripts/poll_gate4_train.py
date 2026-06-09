#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

PROXY = "6c9efizq1aoocf-64411022@ssh.runpod.io"
KEY = r"C:\Users\chkri\.ssh\id_ed25519"

def _stdin(target_step: int) -> str:
    ckpt = f"/workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-step-{target_step:06d}.pt"
    return f"""\
test -f '{ckpt}' && echo TARGET_CKPT_OK || echo TARGET_CKPT_MISSING
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
            input=_stdin(args.target_step),
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        ckpt_ok = False
        done = False
        no_train = False
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line == "TARGET_CKPT_OK":
                ckpt_ok = True
            if line == "DONE":
                done = True
            if line == "NO_TRAIN":
                no_train = True
        print(target_tag if ckpt_ok else "missing", "DONE" if done else "running", "NO_TRAIN" if no_train else "train_active", flush=True)
        if ckpt_ok and (done or no_train):
            print("READY_FOR_EVAL", flush=True)
            sys.exit(0)
        time.sleep(args.poll_sec)
    print("TIMEOUT", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
