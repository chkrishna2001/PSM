#!/usr/bin/env bash
# Quick Colab smoke: 3 turns, T4, ~15 min budget.
export PATH="$HOME/.local/bin:$PATH"
export LOCOMO_LIMIT="${LOCOMO_LIMIT:-3}"
export LOCOMO_OFFSET="${LOCOMO_OFFSET:-2960}"
export COLAB_TIMEOUT_SEC="${COLAB_TIMEOUT_SEC:-1800}"
export COLAB_SESSION="${COLAB_SESSION:-psm-locomo-smoke}"
export LOCOMO_SKIP_EVAL="${LOCOMO_SKIP_EVAL:-1}"
exec bash /mnt/c/Users/chkri/source/repos/PSM/psm-model/scripts/colab_locomo_hf_automate.sh "$@"
