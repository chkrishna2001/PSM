#!/usr/bin/env bash
# HF v5k two-pass LoCoMo ingest on a Colab VM.
# Phases: setup (extract, adapters, npm) | ingest (node ingest + eval) | all
set -euo pipefail

REPO="${PSM_REPO_ROOT:-/content/PSM}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
HF_MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
PHASE="${COLAB_PHASE:-all}"
LIMIT="${LOCOMO_LIMIT:-0}"
OFFSET="${LOCOMO_OFFSET:-0}"
LIMIT="${LIMIT//$'\r'/}"
OFFSET="${OFFSET//$'\r'/}"
DB="${LOCOMO_DB:-/content/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db}"
RESULTS_DIR="/content/locomo/results"
LABEL="hf-prod-v5k-two-pass"
INGEST_DB="benchmark/locomo/results/$(basename "$DB")"
SUMMARY="benchmark/locomo/results/ingest-psm-model-summary.json"
SETUP_MARKER="/content/locomo/.setup-done"
LOG="${RESULTS_DIR}/ingest.log"
RUN_LOG="${RESULTS_DIR}/run.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN_LOG" "$LOG"; }

flush_sqlite() {
  local db_path="$1"
  [[ -f "$db_path" ]] || return 0
  python3 -c "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);c.execute('pragma wal_checkpoint(truncate)');c.close()" "$db_path"
}

hf_download() {
  if command -v hf >/dev/null 2>&1; then
    hf download "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$@"
  else
    python3 -m huggingface_hub.cli.hf download "$@"
  fi
}

fetch_adapter() {
  local prefix="$1" dest="$2"
  mkdir -p "$dest"
  if [[ ! -f "$dest/adapter_model.safetensors" ]]; then
    hf_download "$HF_MODEL_REPO" --repo-type model \
      --include "${prefix}/adapter/*" \
      --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" --token "$HF_TOKEN"
    cp -a "psm-model/prod-memory/checkpoints/_hf_dl/${prefix}/adapter/"* "$dest/"
  fi
}

do_setup() {
  log "setup: phase start limit=$LIMIT offset=$OFFSET"
  mkdir -p "$RESULTS_DIR" "$(dirname "$DB")"

  if [[ -f /content/psm-repo.tgz ]]; then
    rm -rf "$REPO"
    mkdir -p "$REPO"
    tar -xzf /content/psm-repo.tgz -C "$REPO"
    mkdir -p "$REPO/psm-model/prod-memory/checkpoints"
  elif [[ ! -f "$REPO/package.json" ]]; then
    git clone --depth 1 "$GIT_URL" "$REPO"
  fi
  cd "$REPO"

  if ! command -v node >/dev/null || ! node --version | grep -qE 'v2[2-9]'; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs
  fi

  pip install -q 'huggingface_hub[cli]' torch transformers peft accelerate sentencepiece 'torchao>=0.16.0'
  export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
  export PYTHONPATH="$REPO/psm-model/src:$REPO/psm-model/prod-memory"
  export PSM_RUNPOD=0
  export PSM_FORCE_CPU=0
  export PSM_ALLOW_LOCAL_GPU=1

  log "setup: python=$(python3 --version) node=$(node --version)"
  python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee -a "$RUN_LOG" "$LOG"

  fetch_adapter hf-prod-v5k-gate-distill-qwen0.5b psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapte
  fetch_adapter hf-prod-v5k-extract-qwen0.5b psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapte
  log "setup: adapters ready"

  if [[ ! -f benchmark/locomo/data/locomo10.json ]]; then
    mkdir -p benchmark/locomo/data
    curl -fsSL -o benchmark/locomo/data/locomo10.json \
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
  fi

  mkdir -p benchmark/locomo/results
  if [[ -f "$DB" ]]; then
    cp -f "$DB" "$INGEST_DB"
    log "setup: seeded ingest db from $DB"
  fi

  rm -rf node_modules
  if [[ -f dist/benchmark/locomo/src/ingest-psm-model.js && -f src/psm-core/dist/remember-server.js ]]; then
    log "setup: using prebuilt dist from tarball"
    npm ci --no-audit --no-fund --ignore-scripts
  else
    log "setup: building dist on Colab VM"
    npm ci --no-audit --no-fund --ignore-scripts
    npm run build
  fi

  grep -q bareName src/psm-core/dist/remember-server.js && log "setup: remember-server python3 fix present" || log "setup: WARN remember-server missing bareName fix"

  touch "$SETUP_MARKER"
  log "setup: done"
}

do_ingest() {
  [[ -f "$SETUP_MARKER" ]] || { log "ingest: setup marker missing — run setup first"; exit 1; }
  cd "$REPO"
  export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
  export PYTHONPATH="$REPO/psm-model/src:$REPO/psm-model/prod-memory"
  export PSM_RUNPOD=0
  export PSM_FORCE_CPU=0
  export PSM_ALLOW_LOCAL_GPU=1

  if [[ -f "$DB" && ! -f "$INGEST_DB" ]]; then
    cp -f "$DB" "$INGEST_DB"
  fi

  log "ingest: start limit=$LIMIT offset=$OFFSET db=$INGEST_DB"
  INGEST_ARGS=(
    --limit "$LIMIT"
    --offset "$OFFSET"
    --batch-size 5
    --input-format locomo
    --python python3
    --db "$INGEST_DB"
    --device cuda
    --repo-root "$REPO"
    --hf-binary-adapter psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapte
    --hf-extract-adapter psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapte
    --hf-model qwen0.5b
  )

  BASELINE_DECISIONS="$(python3 -c "import sqlite3,sys;print(sqlite3.connect(sys.argv[1]).execute('select count(*) from decisions').fetchone()[0])" "$INGEST_DB")"
  log "ingest: baseline decisions=$BASELINE_DECISIONS"

  set +e
  PSM_REPO_ROOT="$REPO" node /content/ingest-cli.mjs "${INGEST_ARGS[@]}" >>"$LOG" 2>&1 &
  INGEST_PID=$!
  (
    while kill -0 "$INGEST_PID" 2>/dev/null; do
      sleep 120
      flush_sqlite "$INGEST_DB" 2>/dev/null || true
      cp -f "$INGEST_DB" "$DB" 2>/dev/null || true
      flush_sqlite "$DB" 2>/dev/null || true
    done
  ) &
  CHK_PID=$!
  wait "$INGEST_PID"
  INGEST_RC=$?
  kill "$CHK_PID" 2>/dev/null || true
  wait "$CHK_PID" 2>/dev/null || true
  set -e
  cp -f "$LOG" "$DB.ingest.log" 2>/dev/null || true

  if [[ ! -f "$SUMMARY" ]]; then
    log "ingest: FAIL missing $SUMMARY"
    tail -n 40 "$LOG" | tee -a "$RUN_LOG" || true
    exit 1
  fi
  log "ingest: summary"
  cat "$SUMMARY" | tee -a "$RUN_LOG" "$LOG"

  SEEN="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['seen'])" "$SUMMARY")"
  flush_sqlite "$INGEST_DB"
  cp -f "$INGEST_DB" "$DB"
  flush_sqlite "$DB"

  DECISIONS="$(python3 -c "import sqlite3,sys;print(sqlite3.connect(sys.argv[1]).execute('select count(*) from decisions').fetchone()[0])" "$DB")"
  log "ingest: rc=$INGEST_RC seen=$SEEN decisions=$DECISIONS (baseline=$BASELINE_DECISIONS)"

  if [[ "$INGEST_RC" -ne 0 ]]; then
    log "ingest: FAILED rc=$INGEST_RC"
    exit "$INGEST_RC"
  fi
  if [[ "$LIMIT" -gt 0 && "$SEEN" -lt "$LIMIT" ]]; then
    log "ingest: FAIL seen=$SEEN expected limit=$LIMIT"
    exit 1
  fi
  if [[ "$LIMIT" -eq 0 && "$DECISIONS" -le "$BASELINE_DECISIONS" ]]; then
    log "ingest: FAIL no new decisions (still $DECISIONS)"
    exit 1
  fi

  if [[ "${LOCOMO_SKIP_EVAL:-0}" != "1" ]]; then
    log "eval: start"
    node dist/benchmark/locomo/src/evaluate.js \
      --db "$INGEST_DB" \
      --out "benchmark/locomo/results/locomo-${LABEL}-nfull-results.json" \
      --top-k 3 >>"$LOG" 2>&1
    cp -f "benchmark/locomo/results/locomo-${LABEL}-nfull-results.json" "$RESULTS_DIR/" 2>/dev/null || true
    log "eval: done"
  fi

  log "ingest: complete db=$DB decisions=$DECISIONS"
}

mkdir -p "$RESULTS_DIR"
case "$PHASE" in
  setup) do_setup ;;
  ingest) do_ingest ;;
  all) do_setup; do_ingest ;;
  *) log "unknown phase=$PHASE"; exit 1 ;;
esac
