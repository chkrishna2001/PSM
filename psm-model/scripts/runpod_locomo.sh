#!/bin/bash
# LoCoMo ingest + retrieval eval on RunPod (GPU). Waits for Gate 4 eval if still running.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"

if [[ ! -d "$ROOT/psm-model/src" ]] || [[ ! -f "$ROOT/package.json" ]]; then
  echo "PSM repo missing; cloning into $ROOT"
  mkdir -p "$(dirname "$ROOT")"
  rm -rf "$ROOT"
  git clone --depth 1 "$GIT_URL" "$ROOT"
fi
cd "$ROOT"

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
CHECKPOINT="${LOCOMO_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-048000.pt}"
DEVICE="${LOCOMO_DEVICE:-cuda}"
LIMIT="${LOCOMO_LIMIT:-25}"
BATCH_SIZE="${LOCOMO_BATCH_SIZE:-5}"
TOP_K="${LOCOMO_TOP_K:-3}"
WINDOW_SIZE="${LOCOMO_WINDOW_SIZE:-2}"
WAIT_FOR_EVAL="${LOCOMO_WAIT_FOR_EVAL:-1}"
PYTHON="${LOCOMO_PYTHON:-$(command -v python3)}"
HF_BINARY="${LOCOMO_HF_BINARY_ADAPTER:-}"
HF_EXTRACT="${LOCOMO_HF_EXTRACT_ADAPTER:-}"
HF_BINARY_PREFIX="${LOCOMO_HF_BINARY_PREFIX:-hf-prod-v5k-gate-distill-qwen0.5b}"
HF_EXTRACT_PREFIX="${LOCOMO_HF_EXTRACT_PREFIX:-hf-prod-v5k-extract-qwen0.5b}"
HF_MODEL_KEY="${LOCOMO_HF_MODEL_KEY:-qwen0.5b}"
HF_LABEL="${LOCOMO_HF_LABEL:-hf-prod-v5k-two-pass}"

if [[ -n "$HF_BINARY" && -n "$HF_EXTRACT" ]]; then
  LIMIT_TAG="${LIMIT}"
  [[ "$LIMIT" == "0" ]] && LIMIT_TAG="full"
  DB="benchmark/locomo/results/locomo-${HF_LABEL}-n${LIMIT_TAG}.db"
  OUT="benchmark/locomo/results/locomo-${HF_LABEL}-n${LIMIT_TAG}-results.json"
  LOG="benchmark/locomo/results/locomo-${HF_LABEL}-n${LIMIT_TAG}.log"
else
  STEP="$(basename "$CHECKPOINT" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
  STEP="${STEP:-latest}"
  DB="benchmark/locomo/results/locomo-psm-model-step-${STEP}-n${LIMIT}.db"
  OUT="benchmark/locomo/results/locomo-psm-model-step-${STEP}-n${LIMIT}-results.json"
  LOG="benchmark/locomo/results/locomo-psm-model-step-${STEP}-n${LIMIT}.log"
fi

echo "=== PSM LoCoMo $(date -u +%Y-%m-%dT%H:%M:%SZ) checkpoint=$CHECKPOINT hf=$HF_BINARY limit=$LIMIT device=$DEVICE ==="

export PYTHONPATH="${ROOT}/psm-model/src:${ROOT}/psm-model/prod-memory"
export PSM_RUNPOD=1
export DEBIAN_FRONTEND=noninteractive

if ! command -v node >/dev/null 2>&1 || ! node --version | grep -qE 'v2[2-9]'; then
  echo "Installing Node.js 22 (node:sqlite requires >=22.13)..."
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -qq nodejs
fi

if [[ "$WAIT_FOR_EVAL" == "1" ]]; then
  while pgrep -f 'psm_model\.eval_checkpoint' >/dev/null 2>&1; do
    echo "Waiting for Gate 4 expanded eval to finish..."
    sleep 30
  done
  echo "Gate 4 eval finished (or not running)."
fi

pip install -q huggingface_hub 2>/dev/null || true
mkdir -p psm-model/checkpoints benchmark/locomo/data

_fetch_hf_adapter() {
  local prefix="$1"
  local dest="$2"
  if [[ -f "$dest/adapter_model.safetensors" ]]; then
    return 0
  fi
  echo "Downloading adapter $prefix from $MODEL_REPO..."
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "${prefix}/adapter/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "${HF_TOKEN:-}"
  mkdir -p "$dest"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/${prefix}/adapter"/*; do
    cp -a "$f" "$dest/"
  done
  shopt -u nullglob
}

if [[ -n "$HF_BINARY" && -n "$HF_EXTRACT" ]]; then
  pip install -q torch transformers peft accelerate sentencepiece 2>/dev/null \
    || pip install -q torch transformers peft accelerate sentencepiece
  _fetch_hf_adapter "$HF_BINARY_PREFIX" "$HF_BINARY"
  _fetch_hf_adapter "$HF_EXTRACT_PREFIX" "$HF_EXTRACT"
else
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Downloading $CHECKPOINT from $MODEL_REPO..."
  hf download "$MODEL_REPO" "$CHECKPOINT" --local-dir .
  hf download "$MODEL_REPO" "${CHECKPOINT%.pt}.tokenizer.json" --local-dir .
  hf download "$MODEL_REPO" "${CHECKPOINT%.pt}.meta.json" --local-dir . 2>/dev/null || true
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint missing after HF fetch: $CHECKPOINT" >&2
  exit 1
fi
fi

if [[ ! -f benchmark/locomo/data/locomo10.json ]]; then
  echo "Fetching LoCoMo data/locomo10.json..."
  curl -fsSL -o benchmark/locomo/data/locomo10.json \
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
fi
if [[ ! -f benchmark/locomo/data/locomo10.json ]]; then
  echo "LoCoMo data missing at benchmark/locomo/data/locomo10.json" >&2
  exit 1
fi

mkdir -p benchmark/locomo/results
exec > >(tee -a "$LOG") 2>&1

echo "--- npm install + build (skip postinstall; LoCoMo uses psm_model Python, not GGUF) ---"
rm -rf node_modules
if [[ -f package-lock.json ]]; then
  npm ci --no-audit --no-fund --ignore-scripts
else
  npm install --no-audit --no-fund --ignore-scripts
fi
npm run build

echo "--- LoCoMo ingest (limit=$LIMIT) ---"
INGEST_ARGS=(
  --limit "$LIMIT"
  --batch-size "$BATCH_SIZE"
  --db "$DB"
  --device "$DEVICE"
  --python "$PYTHON"
  --window-size "$WINDOW_SIZE"
  --repo-root "$ROOT"
)
if [[ -n "$HF_BINARY" && -n "$HF_EXTRACT" ]]; then
  INGEST_ARGS+=(--hf-binary-adapter "$HF_BINARY" --hf-extract-adapter "$HF_EXTRACT" --hf-model "$HF_MODEL_KEY")
else
  INGEST_ARGS+=(--checkpoint "$CHECKPOINT")
fi
node dist/benchmark/locomo/src/ingest-psm-model.js "${INGEST_ARGS[@]}"

INGEST_RC=$?
if [[ "$INGEST_RC" -ne 0 ]]; then
  echo "LoCoMo ingest failed (exit=$INGEST_RC)" >&2
  exit "$INGEST_RC"
fi

echo "--- LoCoMo retrieval eval (top-k=$TOP_K) ---"
node dist/benchmark/locomo/src/evaluate.js \
  --db "$DB" \
  --out "$OUT" \
  --top-k "$TOP_K"

EVAL_RC=$?
echo "=== LoCoMo done $(date -u +%H:%M:%SZ) ingest=$INGEST_RC eval=$EVAL_RC db=$DB out=$OUT ==="
exit "$EVAL_RC"
