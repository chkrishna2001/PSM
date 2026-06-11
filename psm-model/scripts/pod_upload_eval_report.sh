#!/bin/bash
# Upload the gate4 expanded eval report to the HF dataset repo (chkrishna token).
set -euo pipefail
cd "${PSM_REPO_ROOT:-/workspace/PSM}"
EVAL_STEP="${EVAL_STEP:?set EVAL_STEP}"
REPORT="psm-model/checkpoints/gate-eval/gate4-full-expanded-step-${EVAL_STEP}.json"
[[ -f "$REPORT" ]] || { echo "missing report: $REPORT" >&2; exit 1; }
export HF_TOKEN="${DATASET_HF_TOKEN:?need DATASET_HF_TOKEN}"
python3 - "$REPORT" "$EVAL_STEP" <<'PY'
import sys
from huggingface_hub import upload_file

report, step = sys.argv[1], sys.argv[2]
upload_file(
    path_or_fileobj=report,
    path_in_repo=f"eval-reports/gate4-full-expanded-step-{step}.json",
    repo_id="chkrishna2001/psm-50m-action-mixed-v1",
    repo_type="dataset",
)
print(f"PSM_REPORT_UPLOADED eval-reports/gate4-full-expanded-step-{step}.json")
PY
