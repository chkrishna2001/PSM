#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/PSM
if [[ ! -f "$ROOT/package.json" ]]; then
  rm -rf "$ROOT"
  git clone --depth 1 "https://github.com/chkrishna2001/PSM.git" "$ROOT"
fi
echo clone_ok
