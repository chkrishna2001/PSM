#!/bin/bash
# Clean HF LoRA train: teacher v3 llmResponse storage + recall probes.
# HF account: krishnach7262 — token via `o krishnachhftoken` (HF_TOKEN).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1

DATASET_REPO="${PSM_HF_DATASET_REPO:-krishnach7262/psm-prod-memory-data}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
MODEL_KEY="${HF_MODEL_KEY:-qwen0.5b}"
OUTPUT_FORMAT="${HF_OUTPUT_FORMAT:-tagged}"
RECALL_FRACTION="${HF_RECALL_FRACTION:-0.28}"
STEPS="${HF_TRAIN_STEPS:-1200}"
MAX_LENGTH="${HF_MAX_LENGTH:-2048}"
BATCH_SIZE="${HF_BATCH_SIZE:-2}"
GRAD_ACCUM="${HF_GRAD_ACCUM:-4}"
LEARNING_RATE="${HF_LEARNING_RATE:-2e-4}"
SAVE_STEPS="${HF_SAVE_STEPS:-200}"
OUT_DIR="${HF_OUTPUT_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v1-${MODEL_KEY}}"
CURRICULUM="${HF_CURRICULUM:-psm-model/prod-memory/data/hf-prod-v1.jsonl}"
SOURCE_V3="${HF_SOURCE_V3:-psm-model/prod-memory/data/prod-extraction-v3.jsonl}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== hf lora train $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL_KEY dataset=$DATASET_REPO model_repo=$MODEL_REPO ==="

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
if ! command -v tmux >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 || true
fi

pip install -q torch transformers peft datasets accelerate huggingface_hub bitsandbytes 2>/dev/null \
  || pip install -q torch transformers peft datasets accelerate huggingface_hub

mkdir -p psm-model/prod-memory/data psm-model/prod-memory/checkpoints psm-model/prod-memory/results

download_dataset() {
  local remote="$1"
  local dest="$2"
  hf download "$DATASET_REPO" "$remote" \
    --repo-type dataset --local-dir . --token "$HF_TOKEN" 2>/dev/null || return 1
  if [[ -f "$remote" && ! -f "$dest" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -f "$remote" "$dest"
  fi
}

if [[ ! -f "$SOURCE_V3" ]]; then
  echo "Downloading teacher v3 from $DATASET_REPO..."
  download_dataset "prod-memory/$(basename "$SOURCE_V3")" "$SOURCE_V3" || true
fi

if [[ ! -f "$CURRICULUM" ]]; then
  echo "Downloading HF curriculum from $DATASET_REPO..."
  download_dataset "prod-memory/$(basename "$CURRICULUM")" "$CURRICULUM" || true
fi

if [[ ! -f "$CURRICULUM" ]]; then
  python -m prod_memory.build_hf_curriculum \
    --source "$SOURCE_V3" \
    --output "$CURRICULUM" \
    --output-format "$OUTPUT_FORMAT" \
    --recall-fraction "$RECALL_FRACTION" \
    --dataset-repo "$DATASET_REPO" \
    --no-download
fi

TRAIN_LOG="/tmp/psm-hf-lora-train.log"
TRAIN_DONE="/tmp/psm-hf-lora.done"
UPLOAD_SCRIPT="/tmp/psm-hf-lora-upload.sh"
cat > "$UPLOAD_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
if [[ -z "\${HF_TOKEN:-}" ]]; then exit 0; fi
python3 - <<'PY'
import os
from huggingface_hub import HfApi
from pathlib import Path

repo = os.environ.get("PSM_HF_MODEL_REPO", "$MODEL_REPO")
out = Path("$OUT_DIR")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True)
for path in sorted(out.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(out).as_posix()
    api.upload_file(str(path), path_in_repo=f"hf-prod-v1-${MODEL_KEY}/{rel}", repo_id=repo, commit_message=f"upload {rel}")
print("uploaded adapter to", repo)
PY
EOF
chmod +x "$UPLOAD_SCRIPT"

rm -f "$TRAIN_DONE"
tmux kill-session -t psm-hf-lora 2>/dev/null || true
tmux new-session -d -s psm-hf-lora bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PYTHONPATH=psm-model/src:psm-model/prod-memory
  export PSM_RUNPOD=1
  export HF_TOKEN=\"\${HF_TOKEN:-}\"
  export PSM_HF_MODEL_REPO='$MODEL_REPO'
  export HF_HOME=\"\${HF_HOME:-/workspace/.cache/huggingface}\"
  python -m psm_model.hf_lora_train \
    --curriculum '$CURRICULUM' \
    --output-dir '$OUT_DIR' \
    --model '$MODEL_KEY' \
    --max-length '$MAX_LENGTH' \
    --steps '$STEPS' \
    --batch-size '$BATCH_SIZE' \
    --grad-accum '$GRAD_ACCUM' \
    --learning-rate '$LEARNING_RATE' \
    --save-steps '$SAVE_STEPS' \
    2>&1 | tee '$TRAIN_LOG'
  bash '$UPLOAD_SCRIPT' 2>&1 | tee -a '$TRAIN_LOG'
  touch '$TRAIN_DONE'
"
echo "HF LoRA train started in tmux: psm-hf-lora"
echo "log: $TRAIN_LOG"
echo "output: $OUT_DIR"
echo "model_repo: $MODEL_REPO"
