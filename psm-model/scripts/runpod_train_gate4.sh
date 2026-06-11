#!/bin/bash
# Gate 4: resume full StorageDecision @ step-22800 on expanded+anchor curriculum.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
DEVICE="${PSM_TRAIN_DEVICE:-cuda}"

RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-036000.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-full-v2-step-036000.tokenizer.json}"
TARGET_STEPS="${TARGET_STEPS:-42000}"
CURRICULUM="${GATE4_CURRICULUM:-psm-model/data/curriculum/psm-50m-gate4-train-v4.jsonl}"
CURRICULUM_BUILDER="${GATE4_CURRICULUM_BUILDER:-v4}"
SAVE_EVERY="${SAVE_EVERY:-400}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-120}"
UPLOAD_ALL="${UPLOAD_ALL:-1}"
STRUCTURAL_LOSS_WEIGHT="${STRUCTURAL_LOSS_WEIGHT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-0}"
RESUME_STEP="$(basename "$RESUME" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"

echo "=== PSM Gate 4 train $(date -u +%Y-%m-%dT%H:%M:%SZ) device=$DEVICE target=$TARGET_STEPS resume_step=$RESUME_STEP ==="

export PSM_RUNPOD=1
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

for gate4_script in runpod_upload_gate4.sh runpod_locomo.sh; do
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${gate4_script}" \
    --repo-type dataset --local-dir /tmp/psm-gate4-scripts 2>/dev/null || true
  if [[ -f "/tmp/psm-gate4-scripts/psm-code/${gate4_script}" ]]; then
    mkdir -p psm-model/scripts
    cp "/tmp/psm-gate4-scripts/psm-code/${gate4_script}" "psm-model/scripts/${gate4_script}"
  fi
done
if [[ ! -f psm-model/scripts/runpod_upload_gate4.sh ]]; then
  echo "FATAL: runpod_upload_gate4.sh missing (HF psm-code sync failed)" >&2
  exit 1
fi
# Windows script-sync leaves CRLF; bash rejects `set -o pipefail` with \r suffix.
sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
chmod +x psm-model/scripts/runpod_upload_gate4.sh
# HF psm-code/ staging is source of truth for these modules (clone may be stale:
# local fixes are pushed there before launch, not necessarily committed to git).
for gate4_module in gate4_checkpoint_registry eval_generation; do
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${gate4_module}.py" \
    --repo-type dataset --local-dir /tmp/psm-gate4-code 2>/dev/null || true
  if [[ -f "/tmp/psm-gate4-code/psm-code/${gate4_module}.py" ]]; then
    mkdir -p psm-model/src/psm_model
    cp "/tmp/psm-gate4-code/psm-code/${gate4_module}.py" "psm-model/src/psm_model/${gate4_module}.py"
  fi
done

export PYTHONPATH=psm-model/src

mkdir -p psm-model/checkpoints psm-model/checkpoints/gate-eval psm-model/data/curriculum psm-model/data/probes psm-model/data/direct-behavior-v1

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

if [[ ! -f "$RESUME" ]]; then
  FALLBACK="psm-model/checkpoints/real-v3-50m-full-v2-step-036000.pt"
  if [[ "$RESUME" != "$FALLBACK" ]]; then
    echo "Resume checkpoint missing ($RESUME); falling back to $FALLBACK"
    RESUME="$FALLBACK"
    TOK="${FALLBACK%.pt}.tokenizer.json"
    download_ckpt "$RESUME"
    download_ckpt "$TOK"
  fi
fi
if [[ ! -f "$RESUME" ]]; then
  echo "Gate 4 resume checkpoint unavailable on HF: $RESUME" >&2
  exit 1
fi

if [[ ! -f psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl ]]; then
  echo "Downloading full-storage curriculum..."
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/psm-50m-full-storage-v1-filtered.jsonl \
    --repo-type dataset --local-dir psm-model/data || true
fi

for rel in \
  psm-model/data/probes/direct_probes.jsonl \
  psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
  psm-model/data/direct-behavior-v1/manual-probe.jsonl; do
  if [[ ! -f "$rel" ]]; then
    HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" \
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
    HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" \
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
  if [[ "$CURRICULUM_BUILDER" == "v4" ]]; then
    for module in build_gate4_train_v4 build_gate4_complete_tag_drills build_gate4_fact_format_drills mine_gate4_parse_failures; do
      if ! python3 -c "import psm_model.${module}" 2>/dev/null; then
        echo "Fetching psm_model.${module} from $DATASET_REPO..."
        HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${module}.py" \
          --repo-type dataset --local-dir /tmp/psm-gate4-code || true
        if [[ -f "/tmp/psm-gate4-code/psm-code/${module}.py" ]]; then
          cp "/tmp/psm-gate4-code/psm-code/${module}.py" "psm-model/src/psm_model/${module}.py"
        fi
      fi
    done
    if ! python3 -c "import psm_model.build_gate4_train_v4" 2>/dev/null; then
      echo "psm_model.build_gate4_train_v4 unavailable after HF fetch" >&2
      exit 1
    fi

    if [[ ! -f "$CURRICULUM" ]]; then
      CURRICULUM_BASE="$(basename "$CURRICULUM")"
      echo "Downloading prebuilt curriculum $CURRICULUM_BASE from $DATASET_REPO..."
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "curriculum/$CURRICULUM_BASE" \
        --repo-type dataset --local-dir psm-model/data || true
      if [[ -f "psm-model/data/curriculum/$CURRICULUM_BASE" ]]; then
        CURRICULUM="psm-model/data/curriculum/$CURRICULUM_BASE"
      fi
    fi
    if [[ -f "$CURRICULUM" && "${GATE4_FORCE_REBUILD:-0}" != "1" ]]; then
      echo "Using existing v4 curriculum: $CURRICULUM"
      return
    fi

    EXPANDED_BUDGET="${GATE4_EXPANDED_BUDGET:-psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl}"
    if [[ ! -f "$EXPANDED_BUDGET" && -f psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl ]]; then
      echo "Building expanded token-budget JSONL (eval-matched)..."
      python3 -m psm_model.filter_by_token_budget \
        psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
        "$EXPANDED_BUDGET" \
        --tokenizer "$TOK" \
        --max-tokens 1536 \
        --output-format tagged
    fi

    COMPLETE_TAG_DRILLS="${GATE4_COMPLETE_TAG_DRILLS:-psm-model/data/curriculum/gate4-complete-tag-drills.jsonl}"
    if [[ ! -f "$COMPLETE_TAG_DRILLS" ]]; then
      python3 -m psm_model.build_gate4_complete_tag_drills "$COMPLETE_TAG_DRILLS"
    fi

    PARSE_REPAIR="${GATE4_PARSE_REPAIR:-psm-model/data/curriculum/gate4-parse-repair-step-42000.jsonl}"
    EVAL_REPORT="${GATE4_EVAL_REPORT:-psm-model/checkpoints/gate-eval/gate4-full-expanded-step-42000.json}"
    REPAIR_SOURCE="${GATE4_REPAIR_SOURCE:-$EXPANDED_BUDGET}"
    if [[ ! -f "$EVAL_REPORT" ]]; then
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/gate4-full-expanded-step-42000.json \
        --repo-type dataset --local-dir psm-model/data/curriculum || true
      if [[ -f psm-model/data/curriculum/gate4-full-expanded-step-42000.json ]]; then
        EVAL_REPORT="psm-model/data/curriculum/gate4-full-expanded-step-42000.json"
      fi
    fi
    if [[ ! -f "$PARSE_REPAIR" ]]; then
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/gate4-parse-repair-step-42000.jsonl \
        --repo-type dataset --local-dir psm-model/data || true
      if [[ -f psm-model/data/curriculum/gate4-parse-repair-step-42000.jsonl ]]; then
        PARSE_REPAIR="psm-model/data/curriculum/gate4-parse-repair-step-42000.jsonl"
      fi
    fi

    BUILD_ARGS=(
      "$CURRICULUM"
      --direct-probes psm-model/data/probes/direct_probes.jsonl
      --expanded-probes "$EXPANDED_BUDGET"
      --complete-tag-drills "$COMPLETE_TAG_DRILLS"
      --stratified-source psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl
      --direct-copies "${DIRECT_COPIES:-300}"
      --expanded-copies "${EXPANDED_COPIES:-100}"
      --complete-tag-copies "${COMPLETE_TAG_COPIES:-50}"
      --stratified-max "${STRATIFIED_MAX:-1500}"
      --repair-copies "${REPAIR_COPIES:-1}"
    )
    if [[ -f "$PARSE_REPAIR" ]]; then
      BUILD_ARGS+=(--parse-repair "$PARSE_REPAIR")
    elif [[ -f "$EVAL_REPORT" && -f "$REPAIR_SOURCE" ]]; then
      BUILD_ARGS+=(--eval-report "$EVAL_REPORT" --repair-source "$REPAIR_SOURCE" --parse-repair "$PARSE_REPAIR")
    fi
    python3 -m psm_model.build_gate4_train_v4 "${BUILD_ARGS[@]}"
    return
  fi
  if [[ "$CURRICULUM_BUILDER" == "v2" ]]; then
    for module in build_gate4_train_v2 mine_gate4_parse_failures; do
      if ! python3 -c "import psm_model.${module}" 2>/dev/null; then
        echo "Fetching psm_model.${module} from $DATASET_REPO..."
        HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${module}.py" \
          --repo-type dataset --local-dir /tmp/psm-gate4-code || true
        if [[ -f "/tmp/psm-gate4-code/psm-code/${module}.py" ]]; then
          cp "/tmp/psm-gate4-code/psm-code/${module}.py" "psm-model/src/psm_model/${module}.py"
        fi
      fi
    done
    if ! python3 -c "import psm_model.build_gate4_train_v2" 2>/dev/null; then
      echo "psm_model.build_gate4_train_v2 unavailable after HF fetch" >&2
      exit 1
    fi

    PARSE_REPAIR="${GATE4_PARSE_REPAIR:-psm-model/data/curriculum/gate4-parse-repair-step-36000.jsonl}"
    EVAL_REPORT="${GATE4_EVAL_REPORT:-}"
    REPAIR_SOURCE="${GATE4_REPAIR_SOURCE:-psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl}"
    EXPANDED_BUDGET="$REPAIR_SOURCE"

    if [[ ! -f "$EXPANDED_BUDGET" && -f psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl ]]; then
      echo "Building expanded token-budget JSONL for parse-repair mining..."
      python3 -m psm_model.filter_by_token_budget \
        psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
        "$EXPANDED_BUDGET" \
        --tokenizer "$TOK" \
        --max-tokens 1536 \
        --output-format tagged
    fi

    if [[ ! -f "$PARSE_REPAIR" ]]; then
      echo "Downloading parse-repair pack from $DATASET_REPO..."
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/gate4-parse-repair-step-36000.jsonl \
        --repo-type dataset --local-dir psm-model/data || true
      if [[ -f psm-model/data/curriculum/gate4-parse-repair-step-36000.jsonl ]]; then
        PARSE_REPAIR="psm-model/data/curriculum/gate4-parse-repair-step-36000.jsonl"
      fi
    fi

    BUILD_ARGS=(
      "$CURRICULUM"
      --direct-probes psm-model/data/probes/direct_probes.jsonl
      --expanded-probes psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl
      --stratified-source psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl
      --direct-copies "${DIRECT_COPIES:-500}"
      --expanded-copies "${EXPANDED_COPIES:-25}"
      --drill-rows-per-action "${DRILL_ROWS_PER_ACTION:-120}"
      --drill-copies "${DRILL_COPIES:-50}"
      --stratified-max "${STRATIFIED_MAX:-1500}"
      --repair-copies "${REPAIR_COPIES:-3}"
    )

    if [[ -f "$PARSE_REPAIR" ]]; then
      BUILD_ARGS+=(--parse-repair "$PARSE_REPAIR")
    elif [[ -n "$EVAL_REPORT" && -f "$EVAL_REPORT" && -f "$REPAIR_SOURCE" ]]; then
      BUILD_ARGS+=(--eval-report "$EVAL_REPORT" --repair-source "$REPAIR_SOURCE" --parse-repair "$PARSE_REPAIR")
    else
      echo "gate4-train-v2 requires parse-repair pack or eval report + budget source" >&2
      exit 1
    fi

    python3 -m psm_model.build_gate4_train_v2 "${BUILD_ARGS[@]}"
    return
  fi
  if [[ "$CURRICULUM_BUILDER" == "micro" ]]; then
    EVAL_REPORT="${GATE4_EVAL_REPORT:-psm-model/checkpoints/gate-eval/gate4-full-expanded.json}"
    REPAIR_SOURCE="${GATE4_REPAIR_SOURCE:-psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl}"
    PARSE_REPAIR="${GATE4_PARSE_REPAIR:-psm-model/data/curriculum/gate4-parse-repair-step-42000.jsonl}"
    if [[ ! -f "$REPAIR_SOURCE" && -f psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl ]]; then
      python3 -m psm_model.filter_by_token_budget \
        psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
        "$REPAIR_SOURCE" \
        --tokenizer "$TOK" \
        --max-tokens 1536 \
        --output-format tagged
    fi
    if [[ ! -f "$EVAL_REPORT" ]]; then
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/gate4-full-expanded-step-40000.json \
        --repo-type dataset --local-dir psm-model/data/curriculum || true
      if [[ -f psm-model/data/curriculum/gate4-full-expanded-step-40000.json ]]; then
        EVAL_REPORT="psm-model/data/curriculum/gate4-full-expanded-step-40000.json"
      fi
    fi
    for module in build_gate4_parse_repair_micro mine_gate4_parse_failures; do
      if ! python3 -c "import psm_model.${module}" 2>/dev/null; then
        HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${module}.py" \
          --repo-type dataset --local-dir /tmp/psm-gate4-code || true
        cp "/tmp/psm-gate4-code/psm-code/${module}.py" "psm-model/src/psm_model/${module}.py" 2>/dev/null || true
      fi
    done
    python3 -m psm_model.build_gate4_parse_repair_micro "$CURRICULUM" \
      --direct-probes psm-model/data/probes/direct_probes.jsonl \
      --eval-report "$EVAL_REPORT" \
      --repair-source "$REPAIR_SOURCE" \
      --parse-repair "$PARSE_REPAIR" \
      --direct-copies "${DIRECT_COPIES:-20}" \
      --drill-rows-per-action "${DRILL_ROWS_PER_ACTION:-120}" \
      --drill-copies "${DRILL_COPIES:-5}" \
      --repair-copies "${REPAIR_COPIES:-12}"
    return
  fi
  if [[ "$CURRICULUM_BUILDER" == "v3" ]]; then
    for module in build_gate4_train_v3 build_gate4_fact_format_drills convert_chatgpt_exports mine_gate4_parse_failures; do
      if ! python3 -c "import psm_model.${module}" 2>/dev/null; then
        echo "Fetching psm_model.${module} from $DATASET_REPO..."
        HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${module}.py" \
          --repo-type dataset --local-dir /tmp/psm-gate4-code || true
        if [[ -f "/tmp/psm-gate4-code/psm-code/${module}.py" ]]; then
          cp "/tmp/psm-gate4-code/psm-code/${module}.py" "psm-model/src/psm_model/${module}.py"
        fi
      fi
    done
    if ! python3 -c "import psm_model.build_gate4_train_v3" 2>/dev/null; then
      echo "psm_model.build_gate4_train_v3 unavailable after HF fetch" >&2
      exit 1
    fi

    if [[ ! -f "$CURRICULUM" ]]; then
      echo "Downloading prebuilt v3 curriculum from $DATASET_REPO..."
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/psm-50m-gate4-train-v3.jsonl \
        --repo-type dataset --local-dir psm-model/data || true
      if [[ -f psm-model/data/curriculum/psm-50m-gate4-train-v3.jsonl ]]; then
        CURRICULUM="psm-model/data/curriculum/psm-50m-gate4-train-v3.jsonl"
      fi
    fi
    if [[ -f "$CURRICULUM" && "${GATE4_FORCE_REBUILD:-0}" != "1" ]]; then
      echo "Using existing v3 curriculum: $CURRICULUM"
      return
    fi

    EXPANDED_BUDGET="${GATE4_EXPANDED_BUDGET:-psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl}"
    if [[ ! -f "$EXPANDED_BUDGET" && -f psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl ]]; then
      echo "Building expanded token-budget JSONL (eval-matched)..."
      python3 -m psm_model.filter_by_token_budget \
        psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
        "$EXPANDED_BUDGET" \
        --tokenizer "$TOK" \
        --max-tokens 1536 \
        --output-format tagged
    fi

    FACT_DRILLS="${GATE4_FACT_DRILLS:-psm-model/data/curriculum/gate4-fact-format-drills.jsonl}"
    if [[ ! -f "$FACT_DRILLS" ]]; then
      python3 -m psm_model.build_gate4_fact_format_drills "$FACT_DRILLS"
    fi

    CHATGPT_ROWS="${GATE4_CHATGPT_ROWS:-psm-model/data/curriculum/chatgpt-export-rows.jsonl}"
    if [[ ! -f "$CHATGPT_ROWS" ]]; then
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/chatgpt-export-rows.jsonl \
        --repo-type dataset --local-dir psm-model/data || true
      if [[ -f psm-model/data/curriculum/chatgpt-export-rows.jsonl ]]; then
        CHATGPT_ROWS="psm-model/data/curriculum/chatgpt-export-rows.jsonl"
      fi
    fi

    PARSE_REPAIR="${GATE4_PARSE_REPAIR:-psm-model/data/curriculum/gate4-parse-repair-step-42000.jsonl}"
    EVAL_REPORT="${GATE4_EVAL_REPORT:-psm-model/checkpoints/gate-eval/gate4-full-expanded-step-42000.json}"
    REPAIR_SOURCE="${GATE4_REPAIR_SOURCE:-$EXPANDED_BUDGET}"
    if [[ ! -f "$EVAL_REPORT" ]]; then
      HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/gate4-full-expanded-step-42000.json \
        --repo-type dataset --local-dir psm-model/data/curriculum || true
      if [[ -f psm-model/data/curriculum/gate4-full-expanded-step-42000.json ]]; then
        EVAL_REPORT="psm-model/data/curriculum/gate4-full-expanded-step-42000.json"
      fi
    fi

    BUILD_ARGS=(
      "$CURRICULUM"
      --direct-probes psm-model/data/probes/direct_probes.jsonl
      --expanded-probes "$EXPANDED_BUDGET"
      --fact-drills "$FACT_DRILLS"
      --stratified-source psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl
      --direct-copies "${DIRECT_COPIES:-400}"
      --expanded-copies "${EXPANDED_COPIES:-60}"
      --fact-drill-copies "${FACT_DRILL_COPIES:-80}"
      --chatgpt-copies "${CHATGPT_COPIES:-4}"
      --stratified-max "${STRATIFIED_MAX:-2500}"
      --repair-copies "${REPAIR_COPIES:-2}"
    )
    if [[ -f "$CHATGPT_ROWS" ]]; then
      BUILD_ARGS+=(--chatgpt-rows "$CHATGPT_ROWS")
    fi
    if [[ -f "$PARSE_REPAIR" ]]; then
      BUILD_ARGS+=(--parse-repair "$PARSE_REPAIR")
    elif [[ -f "$EVAL_REPORT" && -f "$REPAIR_SOURCE" ]]; then
      BUILD_ARGS+=(--eval-report "$EVAL_REPORT" --repair-source "$REPAIR_SOURCE" --parse-repair "$PARSE_REPAIR")
    fi
    python3 -m psm_model.build_gate4_train_v3 "${BUILD_ARGS[@]}"
    return
  fi
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

if [[ "$CURRICULUM_BUILDER" == "micro" ]]; then
  STRUCTURAL_LOSS_WEIGHT="${STRUCTURAL_LOSS_WEIGHT:-8}"
  PROMOTE_SPAN_WEIGHT="${PROMOTE_SPAN_WEIGHT:-4}"
  echo "micro train: structural_loss=$STRUCTURAL_LOSS_WEIGHT promote_span=$PROMOTE_SPAN_WEIGHT"
fi

RUN_STEM="real-v3-50m-full-v2"
export GATE4_PINNED_STEPS="${RESUME_STEP}"
SYNC_CMD="cd '$ROOT' && export UPLOAD_ALL=${UPLOAD_ALL} KEEP_LOCAL=$KEEP_LOCAL GATE4_PINNED_STEPS='$RESUME_STEP' PSM_HF_MODEL_REPO='$MODEL_REPO' PSM_HF_DATASET_REPO='$DATASET_REPO' && while true; do sleep $SYNC_INTERVAL_SEC; bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate4-sync.log; done"
tmux kill-session -t psm-gate4-sync 2>/dev/null || true
tmux new-session -d -s psm-gate4-sync bash -lc "$(printf '%q' "$SYNC_CMD")"
echo "HF sync loop: tmux psm-gate4-sync every ${SYNC_INTERVAL_SEC}s keep-local=$KEEP_LOCAL"

echo "--- training (tmux session psm-gate4) ---"
TRAIN_LOG="/tmp/psm-gate4-train.log"
TRAIN_DONE="/tmp/psm-gate4.done"
rm -f "$TRAIN_DONE"
tmux kill-session -t psm-gate4 2>/dev/null || true
tmux new-session -d -s psm-gate4 bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.train \
    '$CURRICULUM' \
    --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
    --resume '$RESUME' \
    --tokenizer '$TOK' \
    --steps '$TARGET_STEPS' \
    --batch-size '$BATCH_SIZE' \
    --learning-rate '$LEARNING_RATE' \
    --min-learning-rate '$MIN_LEARNING_RATE' \
    --preset 50m \
    --output-format tagged \
    --sampling action_balanced \
    --device '$DEVICE' \
    --save-every '$SAVE_EVERY' \
    --metrics-out psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl \
    --action-span-weight ignore=4 \
    --action-span-weight promote_semantic='${PROMOTE_SPAN_WEIGHT:-8}' \
    --action-span-weight store_episodic=8 \
    --action-span-weight flag_conflict=3 \
    --structural-loss-weight '${STRUCTURAL_LOSS_WEIGHT}' \
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
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

echo "=== Gate 4 train done $(date -u +%H:%M:%SZ) ==="

tmux kill-session -t psm-gate4-sync 2>/dev/null || true
export UPLOAD_ALL=1
bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate4-sync.log

if [[ "${GATE4_EVAL_AFTER:-1}" == "1" ]]; then
  STEP_CKPT="psm-model/checkpoints/real-v3-50m-full-v2-step-$(printf '%06d' "$TARGET_STEPS").pt"
  if [[ ! -f "$STEP_CKPT" ]]; then
    STEP_CKPT="psm-model/checkpoints/real-v3-50m-full-v2.pt"
  fi
  echo "--- post-train expanded eval on $STEP_CKPT ---"
  EXPANDED_PROBE="psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl"
  if [[ ! -f "$EXPANDED_PROBE" && -f psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl ]]; then
    python3 -m psm_model.filter_by_token_budget \
      psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
      "$EXPANDED_PROBE" \
      --tokenizer "${STEP_CKPT%.pt}.tokenizer.json" \
      --max-tokens 1536 \
      --output-format tagged
  fi
  EVAL_OUT="psm-model/checkpoints/gate-eval/gate4-full-expanded-step-$(printf '%06d' "$TARGET_STEPS").json"
  set +e
  python3 -m psm_model.eval_checkpoint \
    "$STEP_CKPT" "$EXPANDED_PROBE" \
    --device "$DEVICE" --output-format tagged --gate-mode expanded \
    | tee "$EVAL_OUT"
  EVAL_RC=${PIPESTATUS[0]}
  set -e
  echo "post-train expanded eval exit=$EVAL_RC wrote $EVAL_OUT"
  if [[ -f "$EVAL_OUT" ]]; then
    python3 -m psm_model.gate4_checkpoint_registry update-eval "$EVAL_OUT" || true
  fi
fi

if [[ "${GATE4_LOCOMO_AFTER:-0}" == "1" ]]; then
  LOCOMO_CKPT="psm-model/checkpoints/real-v3-50m-full-v2-step-$(printf '%06d' "$TARGET_STEPS").pt"
  if [[ ! -f "$LOCOMO_CKPT" ]]; then
    LOCOMO_CKPT="psm-model/checkpoints/real-v3-50m-full-v2.pt"
  fi
  echo "--- post-train LoCoMo smoke (limit=${GATE4_LOCOMO_LIMIT:-25}) on $LOCOMO_CKPT ---"
  export LOCOMO_CHECKPOINT="$LOCOMO_CKPT"
  export LOCOMO_LIMIT="${GATE4_LOCOMO_LIMIT:-25}"
  export LOCOMO_DEVICE="$DEVICE"
  export LOCOMO_WAIT_FOR_EVAL=0
  bash psm-model/scripts/runpod_locomo.sh 2>&1 | tee "benchmark/locomo/results/locomo-post-train-step-$(printf '%06d' "$TARGET_STEPS").log" || true
fi

TARGET_STEP_NUM="$(printf '%06d' "$TARGET_STEPS" | sed 's/^0*//')"
export GATE4_FINAL_SYNC=1
export UPLOAD_ALL=1
export GATE4_KEEP_BEST_ONLY=1
export GATE4_PINNED_STEPS="${RESUME_STEP},${TARGET_STEP_NUM}"
echo "--- final HF sync: upload all, keep registry best only, prune HF ---"
bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate4-sync.log
python3 -m psm_model.gate4_checkpoint_registry verify-hf \
  --repo-id "$MODEL_REPO" \
  --run-stem "$RUN_STEM" \
  --checkpoint-dir psm-model/checkpoints
echo "GATE4_FINAL_SYNC_OK=1"
