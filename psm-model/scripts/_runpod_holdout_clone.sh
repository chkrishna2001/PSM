#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/PSM
if [[ ! -f "$ROOT/package.json" ]]; then
  rm -rf "$ROOT"
  git clone --depth 1 "https://github.com/chkrishna2001/PSM.git" "$ROOT"
else
  git -C "$ROOT" pull --ff-only
fi
echo clone_ok
