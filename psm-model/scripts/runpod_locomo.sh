#!/bin/bash
# LoCoMo ingest + retrieval eval on RunPod (GPU). Waits for Gate 4 eval if still running.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"

CHECKPOINT="${LOCOMO_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2.pt}"
DEVICE="${LOCOMO_DEVICE:-cuda}"
LIMIT="${LOCOMO_LIMIT:-25}"
BATCH_SIZE="${LOCOMO_BATCH_SIZE:-5}"
TOP_K="${LOCOMO_TOP_K:-3}"
WINDOW_SIZE="${LOCOMO_WINDOW_SIZE:-2}"
WAIT_FOR_EVAL="${LOCOMO_WAIT_FOR_EVAL:-1}"
PYTHON="${LOCOMO_PYTHON:-python3}"

STEP="$(basename "$CHECKPOINT" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
STEP="${STEP:-latest}"
DB="benchmark/locomo/results/locomo-psm-model-step-${STEP}-n${LIMIT}.db"
OUT="benchmark/locomo/results/locomo-psm-model-step-${STEP}-n${LIMIT}-results.json"
LOG="benchmark/locomo/results/locomo-psm-model-step-${STEP}-n${LIMIT}.log"

echo "=== PSM LoCoMo $(date -u +%Y-%m-%dT%H:%M:%SZ) checkpoint=$CHECKPOINT limit=$LIMIT device=$DEVICE ==="

export PYTHONPATH="${ROOT}/psm-model/src"
export DEBIAN_FRONTEND=noninteractive

if ! command -v node >/dev/null 2>&1 || ! node --version | grep -qE 'v(1[89]|2[0-9])'; then
  echo "Installing Node.js 20..."
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
fi

if [[ "$WAIT_FOR_EVAL" == "1" ]]; then
  while pgrep -f 'psm_model\.eval_checkpoint' >/dev/null 2>&1; do
    echo "Waiting for Gate 4 expanded eval to finish..."
    sleep 30
  done
  echo "Gate 4 eval finished (or not running)."
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint missing: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f benchmark/locomo/data/locomo10.json ]]; then
  echo "LoCoMo data missing at benchmark/locomo/data/locomo10.json" >&2
  exit 1
fi

mkdir -p benchmark/locomo/results
exec > >(tee -a "$LOG") 2>&1

echo "--- npm install + build ---"
if [[ -f package-lock.json ]]; then
  npm ci --no-audit --no-fund
else
  npm install --no-audit --no-fund
fi
npm run build

echo "--- LoCoMo ingest (limit=$LIMIT) ---"
node dist/benchmark/locomo/src/ingest-psm-model.js \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --db "$DB" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --python "$PYTHON" \
  --window-size "$WINDOW_SIZE" \
  --repo-root "$ROOT"

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
