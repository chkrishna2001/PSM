"""Pull prod grounding eval JSONs from HF."""
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo = "subbu83/psm-50m-mixed-v1-run"
out_dir = Path(__file__).resolve().parents[1] / "prod-memory" / "results"
out_dir.mkdir(parents=True, exist_ok=True)

for step in ("058000", "059200", "059500"):
    remote = f"psm-model/prod-memory/results/prod-grounding-{step}.json"
    local = hf_hub_download(repo, remote, repo_type="model")
    dest = out_dir / f"prod-grounding-{step}.json"
    shutil.copy(local, dest)
    agg = json.loads(dest.read_text(encoding="utf-8"))["aggregate"]
    print(
        f"{step}: effective_stored={agg['effective_stored']}/{agg['cases']} "
        f"parse={agg['parse_valid_rate']} ignore={agg['fail_safe_ignore_rate']}"
    )
