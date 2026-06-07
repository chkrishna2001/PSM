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
# Gate 3 pass @22800 is the clean resume base for eval-aligned v1 curriculum.
TARGET_STEPS="${TARGET_STEPS:-36000}"
CURRICULUM="${GATE4_CURRICULUM:-psm-model/data/curriculum/psm-50m-gate4-train-v1.jsonl}"
CURRICULUM_BUILDER="${GATE4_CURRICULUM_BUILDER:-v1}"

echo "=== PSM Gate 4 train $(date -u +%Y-%m-%dT%H:%M:%SZ) device=$DEVICE target=$TARGET_STEPS ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy

if [[ -d "$ROOT/psm-model/src" ]]; then
  echo "PSM repo present at $ROOT"
  cd "$ROOT"
  git pull --ff-only || true
else
  echo "PSM repo missing or incomplete; fresh clone into $ROOT"
  if [[ -d "$ROOT" ]]; then
    stale="${ROOT}.stale.$(date +%s)"
    if mv "$ROOT" "$stale" 2>/dev/null; then
      echo "Moved stale tree to $stale"
    else
      echo "Removing incomplete tree at $ROOT"
      rm -rf "$ROOT"
    fi
  fi
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$GIT_URL" "$ROOT"
  cd "$ROOT"
fi
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
  if [[ "$CURRICULUM_BUILDER" == "v1" ]]; then
    if python3 -c "import psm_model.build_gate4_train_v1" 2>/dev/null; then
      python3 -m psm_model.build_gate4_train_v1 "$CURRICULUM" \
        --direct-probes psm-model/data/probes/direct_probes.jsonl \
        --expanded-probes psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
        --stratified-source psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl \
        --direct-copies "${DIRECT_COPIES:-500}" \
        --expanded-copies "${EXPANDED_COPIES:-40}" \
        --drill-rows-per-action "${DRILL_ROWS_PER_ACTION:-120}" \
        --drill-copies "${DRILL_COPIES:-25}" \
        --stratified-max "${STRATIFIED_MAX:-2500}"
      return
    fi
    echo "psm_model.build_gate4_train_v1 missing; using inline gate4-train-v1 builder"
    export GATE4_CURRICULUM_OUT="$CURRICULUM"
    export DIRECT_COPIES="${DIRECT_COPIES:-500}"
    export EXPANDED_COPIES="${EXPANDED_COPIES:-40}"
    export DRILL_ROWS_PER_ACTION="${DRILL_ROWS_PER_ACTION:-120}"
    export DRILL_COPIES="${DRILL_COPIES:-25}"
    export STRATIFIED_MAX="${STRATIFIED_MAX:-2500}"
    python3 - <<'PY'
import json
import os
import random
from collections import Counter
from pathlib import Path

from psm_model.data import validate_training_row
from psm_model.generate_direct_behavior_curriculum import build_rows as build_direct_behavior_rows

output = Path(os.environ["GATE4_CURRICULUM_OUT"])
direct = Path("psm-model/data/probes/direct_probes.jsonl")
expanded = Path("psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl")
stratified_source = Path("psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl")
direct_copies = int(os.environ["DIRECT_COPIES"])
expanded_copies = int(os.environ["EXPANDED_COPIES"])
drill_rows_per_action = int(os.environ["DRILL_ROWS_PER_ACTION"])
drill_copies = int(os.environ["DRILL_COPIES"])
stratified_max = int(os.environ["STRATIFIED_MAX"])
parse_actions = {"promote_semantic", "store_episodic"}


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
                    "source": f"gate4_train_v1:{prefix}",
                }
            )
            added += 1
    return added


def sample_stratified(path: Path, *, max_rows: int, seed: int = 42):
    if not path.exists():
        return []
    by_action = {action: [] for action in sorted(parse_actions)}
    for row in load_rows(path):
        action = row["expected"]["action"]
        if action not in parse_actions:
            continue
        _, issues = validate_training_row(row)
        if issues:
            continue
        by_action[action].append(row)
    per_action_cap = max(1, max_rows // len(parse_actions))
    rng = random.Random(seed)
    sampled = []
    for action in sorted(parse_actions):
        pool = by_action[action]
        rng.shuffle(pool)
        sampled.extend(pool[:per_action_cap])
    rng.shuffle(sampled)
    return sampled[:max_rows]


rows = []
seen = set()
drill_rows = [
    row
    for row in build_direct_behavior_rows(drill_rows_per_action)
    if row["expected"]["action"] in parse_actions
]
stratified_rows = sample_stratified(stratified_source, max_rows=stratified_max)
direct_added = copy_rows(load_rows(direct), prefix="direct-anchor", copies=direct_copies, seen=seen, out=rows)
expanded_added = copy_rows(load_rows(expanded), prefix="expanded-full", copies=expanded_copies, seen=seen, out=rows)
drill_added = copy_rows(drill_rows, prefix="parse-drill", copies=drill_copies, seen=seen, out=rows)
stratified_added = copy_rows(stratified_rows, prefix="stratified-real", copies=1, seen=seen, out=rows)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
total = len(rows)
print(
    json.dumps(
        {
            "curriculum": "gate4-train-v1-inline",
            "output": str(output),
            "rows": total,
            "direct_anchor_rows": direct_added,
            "expanded_full_rows": expanded_added,
            "parse_drill_rows": drill_added,
            "stratified_real_rows": stratified_added,
            "mix_shares": {
                "expanded_full": round(expanded_added / total, 4) if total else 0.0,
                "parse_drill": round(drill_added / total, 4) if total else 0.0,
                "stratified_real": round(stratified_added / total, 4) if total else 0.0,
                "direct_anchor": round(direct_added / total, 4) if total else 0.0,
            },
            "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
    return
  fi
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
    --action-span-weight ignore=4 \
    --action-span-weight promote_semantic=8 \
    --action-span-weight store_episodic=8 \
    --action-span-weight flag_conflict=3 \
    --eval-every '${EVAL_EVERY:-200}' \
    --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --manual-probe psm-model/data/probes/direct_probes.jsonl \
    --abort-after-step '${ABORT_AFTER_STEP:-30000}' \
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
