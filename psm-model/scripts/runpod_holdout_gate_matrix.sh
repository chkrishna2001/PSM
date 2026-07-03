#!/bin/bash
# Single-pod holdout matrix: parallel first, sequential fallback for failures.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"

# ponytail: tar-push from Windows may leave CRLF; bash treats pipefail\r as invalid
for f in psm-model/scripts/runpod_holdout_gate.sh psm-model/scripts/runpod_holdout_gate_matrix.sh; do
  [[ -f "$f" ]] && sed -i 's/\r$//' "$f"
done

MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
SAMPLE_IDS="${HOLDOUT_SAMPLE_IDS:-conv-30,conv-41}"
ANSWER_LIMIT="${GATE_ANSWER_LIMIT:-30}"
PROFILES_CSV="${GATE_PROFILES:-v5n-dpo,v5n,v5h}"
HF_TOKEN="${HF_TOKEN:-}"
PYTHON="${GATE_PYTHON:-$(command -v python3)}"
MATRIX_OUT="benchmark/locomo/results/holdout-gate-matrix.json"
MATRIX_LOG="/tmp/psm-holdout-gate-matrix.log"

declare -A PREFIX=(
  [v5n-dpo]=hf-prod-v5n-dpo-qwen0.5b
  [v5n]=hf-prod-v5n-qwen0.5b
  [v5h]=hf-prod-v5h-qwen0.5b
)
declare -A ADAPTER=(
  [v5n-dpo]=psm-model/prod-memory/checkpoints/hf-prod-v5n-dpo-qwen0.5b/adapter
  [v5n]=psm-model/prod-memory/checkpoints/hf-prod-v5n-qwen0.5b/adapter
  [v5h]=psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter
)

exec > >(tee -a "$MATRIX_LOG") 2>&1
echo "=== holdout matrix $(date -u +%Y-%m-%dT%H:%M:%SZ) profiles=$PROFILES_CSV samples=$SAMPLE_IDS ==="

export PYTHONPATH="${ROOT}/psm-model/src:${ROOT}/psm-model/prod-memory"
export PSM_RUNPOD=1
export DEBIAN_FRONTEND=noninteractive
export LOCOMO_LLM_PROVIDER=cloudflare
export GATE_PROFILES="$PROFILES_CSV"
export HOLDOUT_SAMPLE_IDS="$SAMPLE_IDS"

_fetch_adapter() {
  local prefix="$1"
  local dest="$2"
  if [[ -f "$dest/adapter_model.safetensors" ]]; then
    return 0
  fi
  echo "Downloading adapter $prefix..."
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "${prefix}/adapter/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "$HF_TOKEN"
  mkdir -p "$dest"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/${prefix}/adapter"/*; do
    cp -a "$f" "$dest/"
  done
  shopt -u nullglob
}

if ! command -v node >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq curl ca-certificates
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -qq nodejs
fi
pip install -q torch transformers peft accelerate huggingface_hub sentencepiece 2>/dev/null \
  || pip install -q torch transformers peft accelerate huggingface_hub sentencepiece
mkdir -p benchmark/locomo/data benchmark/locomo/results
if [[ ! -f benchmark/locomo/data/locomo10.json ]]; then
  curl -fsSL -o benchmark/locomo/data/locomo10.json \
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
fi
IFS=',' read -r -a PROFILES <<< "$PROFILES_CSV"
for p in "${PROFILES[@]}"; do
  p="${p// /}"
  _fetch_adapter "${PREFIX[$p]}" "${ADAPTER[$p]}"
done
rm -rf node_modules
if [[ -f package-lock.json ]]; then npm ci --no-audit --no-fund --ignore-scripts; else npm install --no-audit --no-fund --ignore-scripts; fi
npm run build

_run_one() {
  local profile="$1"
  GATE_PROFILE="$profile" \
  HF_ADAPTER_DIR="${ADAPTER[$profile]}" \
  HF_ADAPTER_PREFIX="${PREFIX[$profile]}" \
  HF_OUTPUT_FORMAT=json \
  HF_MAX_NEW_TOKENS=768 \
  HOLDOUT_SAMPLE_IDS="$SAMPLE_IDS" \
  GATE_ANSWER_LIMIT="$ANSWER_LIMIT" \
  GATE_SKIP_SETUP=1 \
  GATE_SKIP_BUILD=1 \
  HF_TOKEN="$HF_TOKEN" \
  bash psm-model/scripts/runpod_holdout_gate.sh
}

_parallel_failed=0
pids=()
for p in "${PROFILES[@]}"; do
  p="${p// /}"
  echo "--- parallel start $p ---"
  _run_one "$p" &
  pids+=("$!")
done
for i in "${!PROFILES[@]}"; do
  p="${PROFILES[$i]// /}"
  if ! wait "${pids[$i]}"; then
    echo "parallel failed: $p"
    _parallel_failed=1
  fi
done

if [[ "$_parallel_failed" == "1" ]]; then
  echo "--- sequential fallback for missing/failed gates ---"
  for p in "${PROFILES[@]}"; do
    p="${p// /}"
    tag="${p}-$(echo "$SAMPLE_IDS" | tr ',' '-')"
    gate_out="benchmark/locomo/results/holdout-gate-${tag}.json"
    if [[ -f "$gate_out" ]]; then
      echo "skip $p (already have $gate_out)"
      continue
    fi
    echo "--- sequential retry $p ---"
    _run_one "$p" || echo "sequential failed: $p"
  done
fi

"$PYTHON" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("/workspace/PSM")
profiles = [p.strip() for p in __import__("os").environ.get("GATE_PROFILES", "v5n-dpo,v5n,v5h").split(",") if p.strip()]
sample_ids = __import__("os").environ.get("HOLDOUT_SAMPLE_IDS", "conv-30,conv-41")
tag_suffix = sample_ids.replace(",", "-")
rows = []
for profile in profiles:
    tag = f"{profile}-{tag_suffix}"
    gate_out = root / f"benchmark/locomo/results/holdout-gate-{tag}.json"
    row = {"profile": profile, "gate_out": str(gate_out), "exit_code": 0 if gate_out.is_file() else 1}
    if gate_out.is_file():
        data = json.loads(gate_out.read_text(encoding="utf-8"))
        row["retrieval_hit_at_1"] = (data.get("retrieval") or {}).get("hit_at_1")
        row["answer_accuracy"] = (data.get("answer") or {}).get("answer_accuracy")
    rows.append(row)
matrix = {
    "mode": "single_pod",
    "sample_ids": sample_ids.split(","),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "profiles": rows,
}
out = root / "benchmark/locomo/results/holdout-gate-matrix.json"
out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"matrix_out": str(out), "profiles": rows}, indent=2))
PY

test -f "$MATRIX_OUT"
echo "matrix done: $MATRIX_OUT"
