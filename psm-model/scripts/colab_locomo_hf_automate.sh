#!/usr/bin/env bash
# WSL driver: provision Colab GPU, resume LoCoMo HF ingest, periodic download sync.
set -euo pipefail

REPO_WIN="/mnt/c/Users/chkri/source/repos/PSM"
SESSION="${COLAB_SESSION:-psm-locomo-hf}"
GPU="${COLAB_GPU:-T4}"
SYNC_SEC="${COLAB_SYNC_SEC:-300}"
TIMEOUT_SEC="${COLAB_TIMEOUT_SEC:-43200}"
SETUP_TIMEOUT_SEC="${COLAB_SETUP_TIMEOUT_SEC:-900}"
LOCAL_SYNC="$REPO_WIN/benchmark/locomo/results/pod-sync"
CHECKPOINT_DB="${1:-$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-n2960-checkpoint.db}"
CHECKPOINT_DB="${CHECKPOINT_DB//$'\r'/}"
if [[ ! -f "$CHECKPOINT_DB" ]]; then
  CHECKPOINT_DB="$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-n2960-checkpoint.db"
fi
REMOTE_DB="/content/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db"
REMOTE_INGEST_DB="/content/PSM/benchmark/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db"
REMOTE_RESULTS="/content/locomo/results"
REMOTE_SCRIPT="/content/colab_locomo_hf.sh"
REMOTE_DRIVER="/content/colab_locomo_hf_driver.py"
REMOTE_ENV="/content/colab_env.sh"
REMOTE_RUN_LOG="/content/locomo/results/run.log"
REMOTE_INGEST_LOG="/content/locomo/results/ingest.log"
REMOTE_SUMMARY="/content/PSM/benchmark/locomo/results/ingest-psm-model-summary.json"
SCRIPT_DIR="$REPO_WIN/psm-model/scripts"

export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
: "${HF_TOKEN:?Set HF_TOKEN in WSL (export HF_TOKEN=...)}"

if ! command -v colab >/dev/null; then
  echo "Run: bash $SCRIPT_DIR/colab_wsl_setup.sh" >&2
  exit 1
fi

colab_upload() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if colab upload "$@"; then
      return 0
    fi
    echo "upload retry $attempt failed; sleeping 15s" >&2
    sleep 15
  done
  return 1
}

LOCOMO_OFFSET="${LOCOMO_OFFSET:-}"
if [[ -z "$LOCOMO_OFFSET" && "$CHECKPOINT_DB" =~ n([0-9]+)-checkpoint\.db$ ]]; then
  LOCOMO_OFFSET="${BASH_REMATCH[1]}"
fi
LOCOMO_OFFSET="${LOCOMO_OFFSET:-0}"

write_env() {
  local phase="${1:-all}"
  cat <<EOF
export HF_TOKEN='$HF_TOKEN'
export LOCOMO_LIMIT='${LOCOMO_LIMIT:-0}'
export LOCOMO_OFFSET='$LOCOMO_OFFSET'
export LOCOMO_DB='$REMOTE_DB'
export COLAB_PHASE='$phase'
export LOCOMO_SKIP_EVAL='${LOCOMO_SKIP_EVAL:-0}'
EOF
}

sync_artifacts() {
  # Live ingest DB (updated during run); REMOTE_DB is only copied at ingest end.
  colab download -s "$SESSION" "$REMOTE_INGEST_DB" \
    "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull.db" 2>/dev/null || \
  colab download -s "$SESSION" "$REMOTE_DB" \
    "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull.db" 2>/dev/null || true
  colab download -s "$SESSION" "${REMOTE_INGEST_DB}-wal" \
    "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull.db-wal" 2>/dev/null || true
  colab download -s "$SESSION" "$REMOTE_DB.ingest.log" \
    "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull.ingest.log" 2>/dev/null || true
  colab download -s "$SESSION" "$REMOTE_INGEST_LOG" \
    "$LOCAL_SYNC/locomo-ingest.log" 2>/dev/null || true
  colab download -s "$SESSION" "$REMOTE_RUN_LOG" \
    "$LOCAL_SYNC/colab-run.log" 2>/dev/null || true
  colab download -s "$SESSION" "$REMOTE_SUMMARY" \
    "$LOCAL_SYNC/ingest-psm-model-summary.json" 2>/dev/null || true
  colab download -s "$SESSION" "$REMOTE_RESULTS/locomo-hf-prod-v5k-two-pass-nfull-results.json" \
    "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull-results.json" 2>/dev/null || true
  if [[ -f "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull.db" ]]; then
    python3 -c "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);print('decisions',c.execute('select count(*) from decisions').fetchone()[0])" \
      "$LOCAL_SYNC/locomo-hf-prod-v5k-two-pass-nfull.db" 2>/dev/null || true
  fi
}

cleanup() {
  sync_artifacts
  colab stop -s "$SESSION" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$LOCAL_SYNC"
colab new -s "$SESSION" --gpu "$GPU"
colab status -s "$SESSION"

colab exec -s "$SESSION" --timeout 120 -f /dev/stdin <<'PY'
import os
os.makedirs("/content/locomo/results", exist_ok=True)
PY

REPO_TAR="$(bash "$SCRIPT_DIR/colab_pack_repo.sh")"
colab_upload -s "$SESSION" "$REPO_TAR" /content/psm-repo.tgz
rm -f "$REPO_TAR"

if [[ -f "$CHECKPOINT_DB" ]]; then
  echo "upload checkpoint $CHECKPOINT_DB -> $REMOTE_DB"
  colab_upload -s "$SESSION" "$CHECKPOINT_DB" "$REMOTE_DB"
fi

colab_upload -s "$SESSION" "$SCRIPT_DIR/colab_locomo_hf.sh" "$REMOTE_SCRIPT"
colab_upload -s "$SESSION" "$SCRIPT_DIR/colab_locomo_hf_driver.py" "$REMOTE_DRIVER"
colab_upload -s "$SESSION" "$SCRIPT_DIR/ingest-cli.mjs" /content/ingest-cli.mjs

ENV_TMP="$(mktemp)"
write_env all >"$ENV_TMP"
colab_upload -s "$SESSION" "$ENV_TMP" "$REMOTE_ENV"
rm -f "$ENV_TMP"

echo "starting LoCoMo offset=$LOCOMO_OFFSET gpu=$GPU (timeout=${TIMEOUT_SEC}s)"
colab exec -s "$SESSION" --timeout "$TIMEOUT_SEC" -f "$SCRIPT_DIR/colab_locomo_hf_driver.py" &
worker=$!

while kill -0 "$worker" 2>/dev/null; do
  sleep "$SYNC_SEC"
  sync_artifacts
  echo "sync $(date -u +%H:%M:%S)"
done
wait "$worker" || true
echo "done"
