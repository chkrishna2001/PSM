#!/bin/bash
# Holdout gate: ingest holdout LoCoMo convs → SQLite → retrieval → Cloudflare answer eval.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"

MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
PROFILE="${GATE_PROFILE:-v5n-dpo}"
ADAPTER_DIR="${HF_ADAPTER_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v5n-dpo-qwen0.5b/adapter}"
ADAPTER_PREFIX="${HF_ADAPTER_PREFIX:-hf-prod-v5n-dpo-qwen0.5b}"
OUTPUT_FORMAT="${HF_OUTPUT_FORMAT:-json}"
MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-768}"
SAMPLE_IDS="${HOLDOUT_SAMPLE_IDS:-conv-30,conv-41}"
ANSWER_LIMIT="${GATE_ANSWER_LIMIT:-30}"
TOP_K="${GATE_TOP_K:-5}"
DEVICE="${GATE_DEVICE:-cuda}"
PYTHON="${GATE_PYTHON:-$(command -v python3)}"
HF_TOKEN="${HF_TOKEN:-}"

TAG="${PROFILE}-$(echo "$SAMPLE_IDS" | tr ',' '-')"
DB="benchmark/locomo/results/holdout-gate-${TAG}.db"
RETRIEVAL_OUT="benchmark/locomo/results/holdout-gate-${TAG}-retrieval.json"
ANSWER_OUT="benchmark/locomo/results/holdout-gate-${TAG}-answer.json"
GATE_OUT="benchmark/locomo/results/holdout-gate-${TAG}.json"
INGEST_SUMMARY_OUT="benchmark/locomo/results/holdout-gate-${TAG}-ingest-summary.json"
LOG="benchmark/locomo/results/holdout-gate-${TAG}.log"

echo "=== holdout gate $(date -u +%Y-%m-%dT%H:%M:%SZ) profile=$PROFILE samples=$SAMPLE_IDS adapter=$ADAPTER_DIR ==="

export PYTHONPATH="${ROOT}/psm-model/src:${ROOT}/psm-model/prod-memory"
export PSM_RUNPOD=1
export PSM_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export DEBIAN_FRONTEND=noninteractive

if [[ "${GATE_SKIP_SETUP:-0}" != "1" ]]; then
if ! command -v node >/dev/null 2>&1 || ! node --version | grep -qE 'v2[2-9]'; then
  echo "Installing Node.js 22..."
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -qq nodejs
fi

pip install -q torch transformers peft accelerate huggingface_hub sentencepiece 2>/dev/null \
  || pip install -q torch transformers peft accelerate huggingface_hub sentencepiece

mkdir -p benchmark/locomo/data benchmark/locomo/results "$(dirname "$ADAPTER_DIR")"

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Downloading adapter $ADAPTER_PREFIX from $MODEL_REPO..."
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "${ADAPTER_PREFIX}/adapter/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "$HF_TOKEN"
  mkdir -p "$ADAPTER_DIR"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/${ADAPTER_PREFIX}/adapter"/*; do
    cp -a "$f" "$ADAPTER_DIR/"
  done
  shopt -u nullglob
fi

if [[ ! -f benchmark/locomo/data/locomo10.json ]]; then
  curl -fsSL -o benchmark/locomo/data/locomo10.json \
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
fi
fi

exec > >(tee -a "$LOG") 2>&1

if [[ "${GATE_SKIP_BUILD:-0}" == "1" && -f dist/benchmark/locomo/src/ingest-psm-model.js ]]; then
  echo "Using existing dist (GATE_SKIP_BUILD=1)"
else
  echo "--- npm build ---"
  rm -rf node_modules
  if [[ -f package-lock.json ]]; then
    npm ci --no-audit --no-fund --ignore-scripts
  else
    npm install --no-audit --no-fund --ignore-scripts
  fi
  npm run build
fi

echo "--- ingest holdout convs → SQLite ---"
export PSM_FORCE_CPU=0
export INGEST_SUMMARY_OUT
node dist/benchmark/locomo/src/ingest-psm-model.js \
  --data benchmark/locomo/data/locomo10.json \
  --db "$DB" \
  --sample-ids "$SAMPLE_IDS" \
  --input-format psm \
  --output-format "$OUTPUT_FORMAT" \
  --hf-adapter "$ADAPTER_DIR" \
  --hf-model qwen0.5b \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --python "$PYTHON" \
  --repo-root "$ROOT"

echo "--- retrieval eval (real LoCoMo QA) ---"
node dist/benchmark/locomo/src/evaluate.js \
  --data benchmark/locomo/data/locomo10.json \
  --db "$DB" \
  --sample-ids "$SAMPLE_IDS" \
  --out "$RETRIEVAL_OUT" \
  --top-k "$TOP_K" \
  --answerable-only

echo "--- answer eval (Cloudflare Workers AI) ---"
export LOCOMO_LLM_PROVIDER=cloudflare
node dist/benchmark/locomo/src/answer-evaluate.js \
  --data benchmark/locomo/data/locomo10.json \
  --db "$DB" \
  --sample-ids "$SAMPLE_IDS" \
  --out "$ANSWER_OUT" \
  --top-k "$TOP_K" \
  --answerable-only \
  --limit "$ANSWER_LIMIT" \
  --hf-adapter "$ADAPTER_DIR" \
  --hf-model qwen0.5b \
  --device "$DEVICE" \
  --python "$PYTHON" \
  --repo-root "$ROOT" \
  --resume false

echo "--- merge gate report ---"
"$PYTHON" - <<PY
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path("$ROOT")
ingest_path = Path(os.environ.get("INGEST_SUMMARY_OUT", "benchmark/locomo/results/ingest-psm-model-summary.json"))
ingest = json.loads(ingest_path.read_text(encoding="utf-8"))
retrieval = json.loads((root / "$RETRIEVAL_OUT").read_text(encoding="utf-8"))
answer = json.loads((root / "$ANSWER_OUT").read_text(encoding="utf-8"))
report = {
    "profile": "$PROFILE",
    "adapter_dir": "$ADAPTER_DIR",
    "sample_ids": "$SAMPLE_IDS".split(","),
    "output_format": "$OUTPUT_FORMAT",
    "max_new_tokens": int("$MAX_NEW_TOKENS"),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "ingest": ingest,
    "retrieval": retrieval.get("summary", {}),
    "answer": answer.get("summary", {}),
    "retrieval_records": len(retrieval.get("records", [])),
    "answer_records": len(answer.get("records", [])),
}
out = root / "$GATE_OUT"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"gate_out": str(out), "retrieval_hit_at_1": report["retrieval"].get("hit_at_1"), "answer_accuracy": report["answer"].get("answer_accuracy")}, indent=2))
PY

test -f "$GATE_OUT"
echo "holdout gate done: $GATE_OUT"
exit 0
