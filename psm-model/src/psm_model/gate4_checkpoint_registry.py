from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REGISTRY_NAME = "gate4-checkpoint-registry.json"
STEP_RE = re.compile(r"-step-(\d+)\.pt$")


def registry_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / REGISTRY_NAME


def load_registry(checkpoint_dir: Path) -> dict[str, Any]:
    path = registry_path(checkpoint_dir)
    if not path.exists():
        return {"best": None, "evals": [], "hf_prune_steps": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(checkpoint_dir: Path, registry: dict[str, Any]) -> Path:
    path = registry_path(checkpoint_dir)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def step_from_checkpoint(checkpoint: str) -> int | None:
    match = STEP_RE.search(checkpoint)
    return int(match.group(1)) if match else None


def _score(eval_report: dict[str, Any]) -> tuple[float, float]:
    return (
        float(eval_report.get("parse_valid_rate") or 0.0),
        float(eval_report.get("action_accuracy") or 0.0),
    )


def update_from_eval_report(checkpoint_dir: Path, eval_report_path: Path) -> dict[str, Any]:
    report = json.loads(eval_report_path.read_text(encoding="utf-8"))
    checkpoint = str(report.get("checkpoint") or "")
    step = step_from_checkpoint(checkpoint)
    if step is None:
        raise ValueError(f"could not parse step from checkpoint: {checkpoint}")

    entry = {
        "step": step,
        "checkpoint": checkpoint,
        "parse_valid_rate": report.get("parse_valid_rate"),
        "action_accuracy": report.get("action_accuracy"),
        "gate_passed": bool((report.get("gate") or {}).get("passed")),
        "eval_report": str(eval_report_path),
    }
    registry = load_registry(checkpoint_dir)
    evals = [row for row in registry.get("evals", []) if row.get("step") != step]
    evals.append(entry)
    evals.sort(key=lambda row: int(row["step"]))
    registry["evals"] = evals

    best = registry.get("best")
    if best is None or _score(entry) > _score(best):
        registry["best"] = entry
    else:
        registry.setdefault("hf_prune_steps", [])
        if step not in registry["hf_prune_steps"]:
            registry["hf_prune_steps"].append(step)

    save_registry(checkpoint_dir, registry)
    return registry


def prune_hf_steps(registry: dict[str, Any], *, repo_id: str, run_stem: str, dry_run: bool = False) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    deleted: list[str] = []
    prune_steps = sorted({int(step) for step in registry.get("hf_prune_steps", [])})
    best = registry.get("best") or {}
    best_step = int(best["step"]) if best.get("step") is not None else None

    for step in prune_steps:
        if best_step is not None and step == best_step:
            continue
        for suffix in (".pt", ".tokenizer.json", ".meta.json"):
            remote = f"psm-model/checkpoints/{run_stem}-step-{step:06d}{suffix}"
            if dry_run:
                deleted.append(remote)
                continue
            try:
                api.delete_file(path_in_repo=remote, repo_id=repo_id, repo_type="model")
                deleted.append(remote)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "hf_prune_skip", "remote": remote, "error": str(exc)}))
    registry["hf_prune_steps"] = []
    return deleted


def pinned_steps(registry: dict[str, Any], *, resume_step: int | None, latest_step: int | None) -> set[int]:
    pinned: set[int] = set()
    best = registry.get("best")
    if isinstance(best, dict) and best.get("step") is not None:
        pinned.add(int(best["step"]))
    if resume_step is not None:
        pinned.add(resume_step)
    if latest_step is not None:
        pinned.add(latest_step)
    return pinned


def remote_paths_for_step(run_stem: str, step: int) -> list[str]:
    base = f"psm-model/checkpoints/{run_stem}-step-{step:06d}"
    return [f"{base}.pt", f"{base}.tokenizer.json", f"{base}.meta.json"]


def verify_hf_steps(*, repo_id: str, run_stem: str, steps: set[int]) -> list[str]:
    """Return remote paths missing from HF for the given steps (all related files)."""
    from huggingface_hub import file_exists

    missing: list[str] = []
    for step in sorted(steps):
        for remote in remote_paths_for_step(run_stem, step):
            try:
                if not file_exists(remote, repo_id=repo_id, repo_type="model"):
                    missing.append(remote)
            except Exception:  # noqa: BLE001
                missing.append(remote)
    return missing


def best_step(registry: dict[str, Any]) -> int | None:
    best = registry.get("best") or {}
    step = best.get("step")
    return int(step) if step is not None else None


def prune_hf_keep_best_only(
    registry: dict[str, Any],
    *,
    repo_id: str,
    run_stem: str,
    dry_run: bool = False,
) -> list[str]:
    """Delete all HF step checkpoints except registry best (+ rolling full-v2.pt kept)."""
    from huggingface_hub import HfApi, list_repo_files

    keep_step = best_step(registry)
    if keep_step is None:
        return []

    api = HfApi()
    deleted: list[str] = []
    keep_prefix = f"psm-model/checkpoints/{run_stem}-step-{keep_step:06d}"
    for remote in list_repo_files(repo_id, repo_type="model"):
        if f"{run_stem}-step-" not in remote:
            continue
        if remote.startswith(keep_prefix):
            continue
        if dry_run:
            deleted.append(remote)
            continue
        try:
            api.delete_file(path_in_repo=remote, repo_id=repo_id, repo_type="model")
            deleted.append(remote)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"event": "hf_keep_best_prune_skip", "remote": remote, "error": str(exc)}))
    return deleted


def prune_hf_sprawl(
    *,
    repo_id: str,
    run_stem: str,
    registry: dict[str, Any] | None = None,
    min_step: int = 10_000,
    max_step: int = 40_000,
    dry_run: bool = False,
) -> list[str]:
    """Delete intermediate HF step checkpoints in a range, keeping registry best + milestones."""
    from huggingface_hub import HfApi, list_repo_files

    api = HfApi()
    registry = registry or {}
    best = registry.get("best") or {}
    keep = {22800, 28000, 32000, 35600, 36000, 41200, 41600, 42000, 42400, 42800, 45600}
    if best.get("step") is not None:
        keep.add(int(best["step"]))

    by_step: dict[int, list[str]] = {}
    for remote in list_repo_files(repo_id):
        if f"{run_stem}-step-" not in remote:
            continue
        match = STEP_RE.search(remote)
        if not match:
            continue
        step = int(match.group(1))
        by_step.setdefault(step, []).append(remote)

    deleted: list[str] = []
    for step in sorted(by_step):
        if step in keep or step < min_step or step >= max_step:
            continue
        for remote in by_step[step]:
            if dry_run:
                deleted.append(remote)
                continue
            try:
                api.delete_file(path_in_repo=remote, repo_id=repo_id, repo_type="model")
                deleted.append(remote)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "hf_sprawl_prune_skip", "remote": remote, "error": str(exc)}))
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 4 checkpoint registry (best eval + HF prune list).")
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update-eval", help="Record expanded eval and update best / prune list.")
    update.add_argument("eval_report", type=Path)
    update.add_argument("--checkpoint-dir", type=Path, default=Path("psm-model/checkpoints"))

    prune = sub.add_parser("prune-hf", help="Delete regressed step checkpoints from HF model repo.")
    prune.add_argument("--repo-id", required=True)
    prune.add_argument("--run-stem", default="real-v3-50m-full-v2")
    prune.add_argument("--checkpoint-dir", type=Path, default=Path("psm-model/checkpoints"))
    prune.add_argument("--dry-run", action="store_true")

    sprawl = sub.add_parser("prune-hf-sprawl", help="Delete intermediate HF step checkpoints to free storage.")
    sprawl.add_argument("--repo-id", required=True)
    sprawl.add_argument("--run-stem", default="real-v3-50m-full-v2")
    sprawl.add_argument("--checkpoint-dir", type=Path, default=Path("psm-model/checkpoints"))
    sprawl.add_argument("--min-step", type=int, default=10_000)
    sprawl.add_argument("--max-step", type=int, default=40_000)
    sprawl.add_argument("--dry-run", action="store_true")

    keep_best = sub.add_parser("prune-hf-keep-best", help="Delete all HF step checkpoints except registry best.")
    keep_best.add_argument("--repo-id", required=True)
    keep_best.add_argument("--run-stem", default="real-v3-50m-full-v2")
    keep_best.add_argument("--checkpoint-dir", type=Path, default=Path("psm-model/checkpoints"))
    keep_best.add_argument("--dry-run", action="store_true")

    verify = sub.add_parser("verify-hf", help="Verify step checkpoints exist on HF (all related files).")
    verify.add_argument("--repo-id", required=True)
    verify.add_argument("--run-stem", default="real-v3-50m-full-v2")
    verify.add_argument("--checkpoint-dir", type=Path, default=Path("psm-model/checkpoints"))
    verify.add_argument("--step", type=int, action="append", dest="steps", default=[])

    show = sub.add_parser("show", help="Print registry JSON.")
    show.add_argument("--checkpoint-dir", type=Path, default=Path("psm-model/checkpoints"))

    args = parser.parse_args()
    if args.command == "update-eval":
        registry = update_from_eval_report(args.checkpoint_dir, args.eval_report)
        print(json.dumps(registry, indent=2, sort_keys=True))
        return 0
    if args.command == "prune-hf":
        registry = load_registry(args.checkpoint_dir)
        deleted = prune_hf_steps(registry, repo_id=args.repo_id, run_stem=args.run_stem, dry_run=args.dry_run)
        if not args.dry_run:
            save_registry(args.checkpoint_dir, registry)
        print(json.dumps({"deleted": deleted, "dry_run": args.dry_run}, indent=2, sort_keys=True))
        return 0
    if args.command == "prune-hf-sprawl":
        registry = load_registry(args.checkpoint_dir)
        deleted = prune_hf_sprawl(
            repo_id=args.repo_id,
            run_stem=args.run_stem,
            registry=registry,
            min_step=args.min_step,
            max_step=args.max_step,
            dry_run=args.dry_run,
        )
        print(json.dumps({"deleted_count": len(deleted), "dry_run": args.dry_run}, indent=2, sort_keys=True))
        return 0
    if args.command == "prune-hf-keep-best":
        registry = load_registry(args.checkpoint_dir)
        deleted = prune_hf_keep_best_only(
            registry,
            repo_id=args.repo_id,
            run_stem=args.run_stem,
            dry_run=args.dry_run,
        )
        print(json.dumps({"deleted_count": len(deleted), "keep_step": best_step(registry), "dry_run": args.dry_run}, indent=2))
        return 0
    if args.command == "verify-hf":
        registry = load_registry(args.checkpoint_dir)
        steps = set(args.steps)
        if not steps:
            keep = best_step(registry)
            if keep is not None:
                steps.add(keep)
        missing = verify_hf_steps(repo_id=args.repo_id, run_stem=args.run_stem, steps=steps)
        print(json.dumps({"missing": missing, "verified_steps": sorted(steps)}, indent=2))
        return 1 if missing else 0
    registry = load_registry(args.checkpoint_dir)
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
