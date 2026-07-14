"""Storage v13 data expansion: PromptMix-style consistent relabeling (arXiv:2310.14192) of the
store/ignore boundary, targeting the two failure patterns v11 showed on the 100-case gate
(0.82): 9 under-stores (terse decisions/findings + standalone technical FACTS wrongly ignored)
and 7 over-stores (transient infra status wrongly stored).

The core PromptMix idea: near-boundary examples relabeled by ONE consistent rule, not
inconsistent per-example hand-labels. The single rule applied here:

  STORE if the sentence CONTAINS the durable finding/decision/fact/result content itself
  (even when wrapped in "the ledger now records X" or "the docs now say X" framing).
  IGNORE only if it is pure transient status / next-step narration with NO durable content
  (infra state like pod/billing/download, "recorded X, now doing Y" with X absent).

This fixes the specific inconsistency found in v7: `ledger-reflects-contradiction-final-check`
was labeled IGNORE but states an actual finding ("flat comparison ignores small decisive
evidence; veto-style overreacts to noisy evidence") -- it directly contradicted the STORE row
"the ledger now records H22 as rejected...". The v13 curriculum builder relabels that row to
STORE. All rows below are mined real turns, excluded from the 100-case gate and v11 training.
"""
from __future__ import annotations

from typing import Any

# STORE: finding/decision/fact content IS in the sentence (the terse-decision pattern the model
# under-stores). "the ledger/docs now records/says X: <content>" -> the X content is durable.
STORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("interaction-history-contradiction-signal",
     "The ledger now says the quiet part plainly: interaction history alone gives us a "
     "contradiction signal, not separation."),
    ("ledger-separates-detect-resolve",
     "The ledger now separates \"detect contradiction\" from \"resolve contradiction,\" which "
     "is the architectural fork you pointed at."),
    ("raw-grid-breaks-similarity",
     "The docs now say the narrow thing we learned: raw-grid input breaks the current "
     "similarity mechanism."),
    ("ledger-rejects-permanent-rejection",
     "The ledger now rejects permanent rejection as a complete mechanism: an unresolved trace "
     "can recover once later observations form a consistent future."),
    ("roadmap-arrangement-necessary",
     "The roadmap is now set up for the stricter next test: identical basis values with "
     "different futures is the point where arrangement becomes necessary, not just independent."),
    # Standalone technical FACTS (durable reference content the model under-stores because
    # training is mostly agent-work turns, not standalone technical statements)
    ("grpc-http2-multiplexing",
     "gRPC runs over HTTP/2, so it multiplexes many concurrent calls on one connection and "
     "supports bidirectional streaming, unlike HTTP/1.1 REST."),
    ("jwt-stateless-no-revoke",
     "JWTs are stateless: the server does not store them, so you cannot revoke a single token "
     "before expiry without an extra denylist or short TTL plus refresh tokens."),
    ("redis-single-threaded",
     "Redis executes commands on a single thread, so individual commands are atomic but a slow "
     "command (like KEYS on a large dataset) blocks all other clients."),
    ("docker-layer-cache-order",
     "Docker caches image layers in order, so putting COPY of source before dependency install "
     "busts the dependency cache on every code change - install deps first."),
    ("postgres-jsonb-gin-index",
     "In Postgres, a GIN index on a jsonb column makes containment (@>) and key-existence "
     "queries fast, but B-tree indexes do not help those operators."),
    ("async-await-no-parallelism",
     "async/await gives concurrency, not parallelism: awaiting tasks sequentially runs them one "
     "at a time - use Task.WhenAll (or asyncio.gather) to actually overlap them."),
)

# IGNORE: pure transient infra status / next-step narration, NO durable project content
IGNORE_SEED_TURNS: tuple[tuple[str, str], ...] = (
    ("pod-deleted-resuming-eval",
     "Pod `2kk9a1eymp5d72` deleted (status 204 - confirmed terminated, no more billing). Now "
     "resuming the local CPU eval against the retrained adapter."),
    ("stopping-run-pull-db-shutdown",
     "Stopping the run. Let me first pull the DB with the 104 stored decisions and then shut "
     "down the pod."),
    ("db-size-matches-shutdown",
     "864KB decoded - matches the DB size on the pod exactly. Now let's shut down the pod to "
     "stop billing."),
    ("pod-stopped-billing-stopped",
     "Pod `udk4kfi83tx5st` is now stopped (`EXITED` as of 16:01:53 UTC) - billing on it has "
     "stopped. Let me check what the earlier SSH session actually captured before responding."),
    ("verifypod-ssh-timeout-connect",
     "The verify-pod call hit an SSH endpoint timeout (proxy hiccup). Let me connect directly "
     "and check the tmux session on the pod."),
    ("training-started-race-condition",
     "Confirmed same pattern as before - training started fine, the \"failed\" status is the "
     "same verify-pod race condition. Let's check current live status directly."),
    ("adapter-pulled-running-eval",
     "Adapter downloaded locally. Now running the same prod fixture eval on CPU at 768 tokens."),
    ("gpu-util-confirmed-checking-restart",
     "GPU is pegged at 100% on the pod, so the ingest is genuinely running on CUDA, not silently "
     "on CPU. Let me keep watching for the store-rate to settle."),
)


def _store_expected(content: str) -> dict[str, Any]:
    return {
        "action": "store_episodic",
        "memory": {"content": content, "type": "episodic"},
        "facts": [],
        "indexables": [],
        "reasoning": "The durable finding, decision, or reusable technical fact is stated in "
        "the sentence itself, so it is worth remembering regardless of any "
        "'now recorded/now says' framing around it.",
    }


def _ignore_expected() -> dict[str, Any]:
    return {
        "action": "ignore",
        "memory": None,
        "facts": [],
        "indexables": [],
        "reasoning": "Pure transient infrastructure status or next-step narration (pod/billing/"
        "download state) with no durable project fact, decision, or result to remember.",
    }


def build_storage_v13_new_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, text in STORE_SEED_TURNS:
        rows.append({
            "id": f"v13-store-{row_id}",
            "task": "remember_llm_response",
            "input": {"conversation": text, "operation": "remember_llm_response", "source_kind": "coding_agent_v13"},
            "expected": _store_expected(text),
            "source": "storage_v13",
        })
    for row_id, text in IGNORE_SEED_TURNS:
        rows.append({
            "id": f"v13-ignore-{row_id}",
            "task": "remember_llm_response",
            "input": {"conversation": text, "operation": "remember_llm_response", "source_kind": "coding_agent_v13"},
            "expected": _ignore_expected(),
            "source": "storage_v13",
        })
    return rows
