from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEMANTIC_TERMS = (
    "always",
    "default",
    "future",
    "likes",
    "prefers",
    "preference",
    "remember that",
    "rule",
    "use ",
    "want future",
)

EPISODIC_TERMS = (
    "attended",
    "created",
    "finished",
    "met ",
    "ran ",
    "reviewed",
    "today",
    "validated",
    "visited",
    "yesterday",
)

NOISE_TERMS = (
    "haha",
    "okay thanks",
    "please continue",
    "sounds good",
    "terminal is still running",
    "weather outside",
)

CONFLICT_TERMS = (
    "different from",
    "not ",
    "wrong",
)

UPDATE_TERMS = (
    "actually",
    "correction",
    "instead",
    "outdated",
)


@dataclass(frozen=True)
class LabelIssue:
    row_id: str
    action: str
    memory_type: str
    severity: str
    reason: str
    conversation: str
    source_kind: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.row_id,
            "action": self.action,
            "memory_type": self.memory_type,
            "severity": self.severity,
            "reason": self.reason,
            "conversation": self.conversation[:240],
            "source_kind": self.source_kind,
        }


def audit_label_risks(path: Path, *, max_examples_per_reason: int = 20) -> dict[str, Any]:
    rows = load_jsonl(path)
    issues: list[LabelIssue] = []
    action_counts: Counter[str] = Counter()
    memory_counts: Counter[str] = Counter()

    for row in rows:
        expected = row.get("expected", {})
        action = str(expected.get("action", "missing"))
        memory = expected.get("memory") if isinstance(expected.get("memory"), dict) else None
        memory_type = str(memory.get("type", "none")) if memory else "none"
        action_counts[action] += 1
        memory_counts[memory_type] += 1
        issues.extend(row_label_issues(row, action=action, memory_type=memory_type))

    issue_counts: Counter[str] = Counter(issue.reason for issue in issues)
    severity_counts: Counter[str] = Counter(issue.severity for issue in issues)
    examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        bucket = examples_by_reason[issue.reason]
        if len(bucket) < max_examples_per_reason:
            bucket.append(issue.to_json())

    return {
        "path": str(path),
        "rows": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "memory_counts": dict(sorted(memory_counts.items())),
        "issue_count": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "examples_by_reason": dict(sorted(examples_by_reason.items())),
    }


def row_label_issues(row: dict[str, Any], *, action: str, memory_type: str) -> list[LabelIssue]:
    payload = row.get("input", {})
    conversation = normalize_space(str(payload.get("conversation", "")))
    source_kind = payload.get("source_kind")
    context = normalize_space(str(payload.get("context", "")))
    haystack = f"{conversation} {context}".lower()
    issues: list[LabelIssue] = []

    semantic_score = count_matches(haystack, SEMANTIC_TERMS)
    episodic_score = count_matches(haystack, EPISODIC_TERMS) + count_date_like(haystack)
    noise_score = count_matches(haystack, NOISE_TERMS)
    conflict_score = count_matches(haystack, CONFLICT_TERMS)
    update_score = count_matches(haystack, UPDATE_TERMS)
    has_prior_context = "memory_store" in haystack or "prior" in haystack or "existing" in haystack

    def add(severity: str, reason: str) -> None:
        issues.append(
            LabelIssue(
                row_id=str(row.get("id", "")),
                action=action,
                memory_type=memory_type,
                severity=severity,
                reason=reason,
                conversation=conversation,
                source_kind=str(source_kind) if source_kind is not None else None,
            )
        )

    if action == "store_episodic" and semantic_score > episodic_score:
        add("high", "semantic_wording_labeled_store_episodic")
    if action == "promote_semantic" and episodic_score > semantic_score and not is_rule_like(haystack):
        add("high", "episodic_wording_labeled_promote_semantic")
    if action != "ignore" and noise_score and semantic_score == 0 and episodic_score == 0:
        add("medium", "noise_wording_not_labeled_ignore")
    if action == "ignore" and (semantic_score or episodic_score) and noise_score == 0:
        add("medium", "durable_wording_labeled_ignore")
    prior_context_actions = {"update_existing", "flag_conflict", "flag_and_store"}
    if has_prior_context and action not in prior_context_actions:
        add("medium", "prior_context_without_update_or_conflict_action")
    if conflict_score and has_prior_context and action not in {"flag_conflict", "flag_and_store"}:
        add("medium", "conflict_wording_not_labeled_flag_conflict")
    if update_score and has_prior_context and action not in prior_context_actions:
        add("medium", "update_wording_without_update_or_conflict_action")
    if action in {"store_episodic", "promote_semantic", "update_existing"} and memory_type == "none":
        add("high", "memory_action_missing_memory_payload")
    if action == "ignore" and memory_type != "none":
        add("high", "ignore_action_has_memory_payload")
    if source_kind == "local_psm_db" and action == "store_episodic" and semantic_score > 0 and episodic_score == 0:
        add("high", "local_psm_semantic_signal_labeled_store_episodic")

    return issues


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_matches(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def count_date_like(text: str) -> int:
    return len(re.findall(r"\b(?:20\d{2}-\d{2}-\d{2}|today|yesterday|friday|monday|tuesday|wednesday|thursday)\b", text))


def is_rule_like(text: str) -> bool:
    return any(term in text for term in ("always", "future", "rule", "for future", "prefer", "prefers"))


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag likely ambiguous or risky labels before long PSM training runs.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--max-examples-per-reason", type=int, default=20)
    parser.add_argument("--fail-on-high-risk", action="store_true", help="Exit non-zero when any high-severity issue is found.")
    parser.add_argument("--fail-on-any-risk", action="store_true", help="Exit non-zero when any issue is found.")
    args = parser.parse_args()

    report = audit_label_risks(args.data, max_examples_per_reason=args.max_examples_per_reason)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_any_risk and report["issue_count"]:
        return 1
    if args.fail_on_high_risk and report["severity_counts"].get("high", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
