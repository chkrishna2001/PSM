#!/usr/bin/env bash
# Pack local PSM repo for Colab upload (run from WSL). Prints tarball path on stdout.
set -euo pipefail

REPO_WIN="${PSM_REPO_WIN:-/mnt/c/Users/chkri/source/repos/PSM}"
MAX_MB="${COLAB_PACK_MAX_MB:-250}"

cd "$REPO_WIN"
if [[ ! -f package.json ]]; then
  echo "pack: missing package.json under $REPO_WIN" >&2
  exit 1
fi

echo "pack: npm run build..." >&2
if command -v npm >/dev/null 2>&1; then
  npm run build >&2
elif [[ -f "/mnt/c/Program Files/nodejs/npm.cmd" ]]; then
  cmd.exe //c "cd /d C:\\Users\\chkri\\source\\repos\\PSM && npm run build" >&2
else
  echo "pack: npm not found in WSL or Windows — run 'npm run build' in repo first" >&2
  exit 1
fi

TAR="$(mktemp /tmp/psm-colab-XXXX.tgz)"
tar -czf "$TAR" -C "$REPO_WIN" \
  --exclude=node_modules --exclude=.git \
  --exclude='psm-model/prod-memory/checkpoints' \
  --exclude='psm-model/prod-memory/results' \
  --exclude='benchmark/locomo/results' \
  package.json package-lock.json tsconfig.json tsconfig.base.json types \
  dist src benchmark psm-model/src psm-model/prod-memory

SIZE_MB="$(du -m "$TAR" | cut -f1)"
echo "pack: tarball ${SIZE_MB}MB" >&2
if [[ "$SIZE_MB" -gt "$MAX_MB" ]]; then
  echo "pack: tarball too large (${SIZE_MB}MB > ${MAX_MB}MB); check excludes" >&2
  rm -f "$TAR"
  exit 1
fi

for must in \
  dist/benchmark/locomo/src/ingest-psm-model.js \
  src/psm-core/dist/remember-server.js \
  psm-model/src/psm_model/hf_remember_server.py; do
  if ! tar -tzf "$TAR" "$must" >/dev/null 2>&1; then
    echo "pack: missing required file in tarball: $must" >&2
    rm -f "$TAR"
    exit 1
  fi
done

echo "$TAR"
