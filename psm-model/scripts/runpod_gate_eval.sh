#!/usr/bin/env bash
# Run the coding-agent gate eval ON THE POD (GPU) instead of locally on CPU.
#
# Why: the 100-case gate takes ~45 min on a laptop CPU but ~2-3 min on the pod's GPU — and the pod
# is already rented. Running it here means we get the score sooner AND can delete the pod straight
# after, instead of paying it to sit idle (or deleting it and then waiting 45 min for a CPU run).
#
# Env:
#   GATE_ADAPTER_DIR  adapter/checkpoint dir on the pod (required)
#   GATE_FIXTURES     fixtures json (default: the 100-case coding-agent gate)
#   GATE_OUT          local-on-pod output path for the full report
#   GATE_MAX_NEW_TOKENS (default 768)
#   GATE_UPLOAD=1     upload the report to $PSM_HF_DATASET_REPO so the driver can pull it
set -uo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT" || exit 1
# Match runpod_hf_lora_train.sh exactly: stay at ROOT and keep PYTHONPATH relative to it. Do NOT cd
# into psm-model/prod-memory — that breaks the relative PYTHONPATH and `prod_memory` stops resolving
# ("No module named prod_memory.eval_hf_grounding").
export PYTHONPATH=psm-model/src:psm-model/prod-memory

ADAPTER="${GATE_ADAPTER_DIR:?GATE_ADAPTER_DIR required}"
FIXTURES="${GATE_FIXTURES:-psm-model/prod-memory/fixtures/holdout-coding-agent-cases.json}"
OUT="${GATE_OUT:-psm-model/prod-memory/results/gate-eval.json}"
MAXTOK="${GATE_MAX_NEW_TOKENS:-768}"

echo "=== gate eval on pod: adapter=$ADAPTER fixtures=$FIXTURES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$(dirname "$OUT")"
for p in "$ADAPTER" "$FIXTURES"; do
  [[ -e "$p" ]] || { echo "EVAL FAILED: missing $p"; exit 1; }
done

python -m prod_memory.eval_hf_grounding \
  --adapter-dir "$ADAPTER" \
  --fixtures "$FIXTURES" \
  --output-format json \
  --device cuda \
  --max-new-tokens "$MAXTOK" \
  --out "$OUT"
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "EVAL FAILED rc=$rc"
  exit $rc
fi

# Print the headline numbers so they land in the driver's captured output.
python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
a = d["aggregate"]
S = {"store","store_episodic","promote_semantic","update_existing","flag_conflict","flag_and_store"}
tp=fp=tn=fn=0
for c in d["cases"]:
    es = c["expectAction"] in S; ps = c["action"] in S
    if es and ps: tp+=1
    elif es and not ps: fn+=1
    elif not es and ps: fp+=1
    else: tn+=1
print("GATE_RESULT " + json.dumps({
    "action_match_rate": a.get("action_match_rate"),
    "overall": f"{tp+tn}/100",
    "store_recall": f"{tp}/{tp+fn}",
    "ignore_recall": f"{tn}/{tn+fp}",
    "false_store": fp, "false_ignore": fn,
    "parse_valid_rate": a.get("parse_valid_rate"),
    "fail_safe_ignore_rate": a.get("fail_safe_ignore_rate"),
}))
PY

if [[ "${GATE_UPLOAD:-0}" == "1" && -n "${PSM_HF_DATASET_REPO:-}" ]]; then
  python - "$OUT" "$PSM_HF_DATASET_REPO" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
out, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
name = Path(out).name
api.upload_file(path_or_fileobj=out, path_in_repo=f"prod-memory/results/{name}",
                repo_id=repo, repo_type="dataset")
print(f"uploaded prod-memory/results/{name} -> {repo}")
PY
fi
echo "=== gate eval done ==="
