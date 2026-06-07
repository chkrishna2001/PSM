#!/bin/bash
# Gate 4: resume full StorageDecision @ step-22800 on expanded+anchor curriculum.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
DEVICE="${PSM_TRAIN_DEVICE:-cuda}"

RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-022800.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-full-v2-step-022800.tokenizer.json}"
TARGET_STEPS="${TARGET_STEPS:-28000}"
CURRICULUM="${GATE4_CURRICULUM:-psm-model/data/curriculum/psm-50m-full-storage-v4-gate4.jsonl}"

echo "=== PSM Gate 4 train $(date -u +%Y-%m-%dT%H:%M:%SZ) device=$DEVICE target=$TARGET_STEPS ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy

if [[ ! -d "$ROOT/psm-model/src" ]]; then
  if [[ -d "$ROOT/.git" ]]; then
    echo "Updating existing PSM repo..."
    git -C "$ROOT" pull --ff-only || true
  fi
  if [[ ! -d "$ROOT/psm-model/src" ]]; then
    echo "Cloning PSM repo..."
    if [[ -d "$ROOT" ]] && [[ -n "$(ls -A "$ROOT" 2>/dev/null || true)" ]]; then
      mv "$ROOT" "${ROOT}.stale.$(date +%s)" 2>/dev/null || rm -rf "$ROOT"
    fi
    mkdir -p "$(dirname "$ROOT")"
    git clone --depth 1 "$GIT_URL" "$ROOT"
  fi
fi
cd "$ROOT"
git pull --ff-only || true
export PYTHONPATH=psm-model/src

mkdir -p psm-model/checkpoints psm-model/data/curriculum psm-model/data/probes psm-model/data/direct-behavior-v1

download_ckpt() {
  local rel="$1"
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    hf download "$MODEL_REPO" "$rel" --local-dir .
  fi
}

for rel in \
  "$RESUME" \
  "${RESUME%.pt}.tokenizer.json" \
  "psm-model/checkpoints/real-v3-50m-full-v2.pt" \
  "psm-model/checkpoints/real-v3-50m-full-v2.tokenizer.json"; do
  download_ckpt "$rel"
done

if [[ ! -f psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl ]]; then
  echo "Downloading full-storage curriculum..."
  hf download "$DATASET_REPO" curriculum/psm-50m-full-storage-v1-filtered.jsonl \
    --repo-type dataset --local-dir psm-model/data || true
fi

for rel in \
  psm-model/data/probes/direct_probes.jsonl \
  psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
  psm-model/data/direct-behavior-v1/manual-probe.jsonl; do
  if [[ ! -f "$rel" ]]; then
    hf download "$DATASET_REPO" \
      data/probes/direct_probes.jsonl \
      data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
      data/direct-behavior-v1/manual-probe.jsonl \
      --repo-type dataset --local-dir . || true
  fi
done
for rel in \
  psm-model/data/probes/direct_probes.jsonl \
  psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
  psm-model/data/direct-behavior-v1/manual-probe.jsonl; do
  if [[ ! -f "$rel" ]]; then
    hf download "$DATASET_REPO" \
      probes/direct_probes.jsonl \
      probes/expanded-probe-v1-filtered.jsonl \
      probes/manual-probe.jsonl \
      --repo-type dataset --local-dir psm-model/data || true
    cp -f psm-model/data/probes/manual-probe.jsonl psm-model/data/direct-behavior-v1/ 2>/dev/null || true
    cp -f psm-model/data/probes/expanded-probe-v1-filtered.jsonl psm-model/data/direct-behavior-v1/ 2>/dev/null || true
  fi
done

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

build_curriculum() {
  if python3 -c "import psm_model.build_gate4_curriculum" 2>/dev/null; then
    python3 -m psm_model.build_gate4_curriculum "$CURRICULUM" \
      --base psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl \
      --direct-probes psm-model/data/probes/direct_probes.jsonl \
      --expanded-probes psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
      --direct-copies "${DIRECT_COPIES:-500}" \
      --expanded-copies "${EXPANDED_COPIES:-8}" \
      --ignore-extra-copies "${IGNORE_EXTRA_COPIES:-4}"
    return
  fi
  echo "psm_model.build_gate4_curriculum missing; using inline curriculum builder"
  export GATE4_CURRICULUM_OUT="$CURRICULUM"
  export DIRECT_COPIES="${DIRECT_COPIES:-500}"
  export EXPANDED_COPIES="${EXPANDED_COPIES:-8}"
  export IGNORE_EXTRA_COPIES="${IGNORE_EXTRA_COPIES:-4}"
  python3 - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

output = Path(os.environ["GATE4_CURRICULUM_OUT"])
base = Path("psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl")
direct = Path("psm-model/data/probes/direct_probes.jsonl")
expanded = Path("psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl")
direct_copies = int(os.environ["DIRECT_COPIES"])
expanded_copies = int(os.environ["EXPANDED_COPIES"])
ignore_extra_copies = int(os.environ["IGNORE_EXTRA_COPIES"])

def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def copy_rows(rows, *, prefix, copies, seen, out, action_filter=None):
    added = 0
    for row in rows:
        action = row["expected"]["action"]
        if action_filter is not None and action not in action_filter:
            continue
        row_id = str(row.get("id") or "row")
        for copy_index in range(copies):
            copied_id = f"{prefix}:{copy_index}:{row_id}"
            if copied_id in seen:
                continue
            seen.add(copied_id)
            out.append(
                {
                    "id": copied_id,
                    "input": row["input"],
                    "expected": row["expected"],
                    "source": f"gate4_curriculum:{prefix}",
                }
            )
            added += 1
    return added

rows = []
seen = set()
for row in load_rows(base):
    row_id = str(row.get("id") or f"base-{len(rows)}")
    if row_id in seen:
        continue
    seen.add(row_id)
    rows.append(row)
base_count = len(rows)
direct_added = copy_rows(load_rows(direct), prefix="direct-anchor", copies=direct_copies, seen=seen, out=rows)
expanded_added = copy_rows(load_rows(expanded), prefix="expanded-anchor", copies=expanded_copies, seen=seen, out=rows)
ignore_added = copy_rows(
    load_rows(expanded),
    prefix="expanded-ignore",
    copies=ignore_extra_copies,
    seen=seen,
    out=rows,
    action_filter={"ignore"},
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
print(
    json.dumps(
        {
            "output": str(output),
            "rows": len(rows),
            "base_rows": base_count,
            "direct_anchor_rows": direct_added,
            "expanded_anchor_rows": expanded_added,
            "ignore_extra_rows": ignore_added,
            "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
}
build_curriculum

echo "--- training (tmux session psm-gate4) ---"
TRAIN_LOG="/tmp/psm-gate4-train.log"
TRAIN_DONE="/tmp/psm-gate4.done"
rm -f "$TRAIN_DONE"
tmux kill-session -t psm-gate4 2>/dev/null || true
tmux new-session -d -s psm-gate4 bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PYTHONPATH=psm-model/src
  python3 -m psm_model.train \
    '$CURRICULUM' \
    --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
    --resume '$RESUME' \
    --tokenizer '$TOK' \
    --steps '$TARGET_STEPS' \
    --batch-size 1 \
    --preset 50m \
    --output-format tagged \
    --sampling action_balanced \
    --device '$DEVICE' \
    --save-every 200 \
    --metrics-out psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl \
    --action-span-weight ignore=8 \
    --action-span-weight promote_semantic=4 \
    --action-span-weight store_episodic=2 \
    --action-span-weight flag_conflict=3 \
    --eval-every 400 \
    --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --manual-probe psm-model/data/probes/direct_probes.jsonl \
    --abort-after-step 24000 \
    --collapse-threshold 0.90 \
    2>&1 | tee '$TRAIN_LOG'
  echo done > '$TRAIN_DONE'
"

sleep 2
tail -n 20 -f "$TRAIN_LOG" &
TAIL_PID=$!
while [[ ! -f "$TRAIN_DONE" ]]; do
  if ! tmux has-session -t psm-gate4 2>/dev/null; then
    echo "tmux session psm-gate4 ended unexpectedly" >&2
    break
  fi
  sleep 30
done
wait "$TAIL_PID" 2>/dev/null || true

echo "=== Gate 4 train done $(date -u +%H:%M:%SZ) ==="
