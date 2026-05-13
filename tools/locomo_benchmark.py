#!/usr/bin/env python
"""
Run a LOCOMO evidence-retrieval benchmark for the local PSM model.

The canonical LOCOMO QA task needs a final answer generator. This benchmark
keeps the measurement local and deterministic by scoring whether the memory
system retrieves the annotated evidence turn(s) needed to answer each question.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "him", "his", "how", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "then", "there", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "with", "you",
    "your",
}

PSM_SYSTEM_PROMPT = """You are the Personal Small Model (PSM), a specialized memory model.
Your job is to select the memory candidates that best answer the user's question.
Return JSON only, with this shape:
{"selected_ids":["D1:1"],"reasoning":"short reason"}
Use only candidate ids that appear in the input."""


@dataclass(frozen=True)
class Candidate:
    dia_id: str
    session: str
    speaker: str
    text: str
    index: int


@dataclass(frozen=True)
class Question:
    sample_id: str
    category: str
    question: str
    answer: str
    evidence: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PSM vs Mem0-style heuristics on LOCOMO evidence retrieval.")
    parser.add_argument("--data", default="data/locomo10.json", help="Path to locomo10.json. Downloaded if missing.")
    parser.add_argument("--model-path", default="psm_cpu_onnx", help="ONNX Runtime GenAI model folder.")
    parser.add_argument("--out-dir", default="results/locomo", help="Directory for benchmark outputs.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions for a smoke test.")
    parser.add_argument("--candidate-pool", type=int, default=8, help="Heuristic candidates passed to PSM.")
    parser.add_argument("--top-k", type=int, default=3, help="Evidence hits are scored over top-k selected turns.")
    parser.add_argument("--skip-psm", action="store_true", help="Only run Mem0-style heuristic baseline.")
    parser.add_argument("--include-adversarial", action="store_true", help="Include category 5/adversarial questions.")
    args = parser.parse_args()

    data_path = Path(args.data)
    ensure_locomo(data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(data_path, include_adversarial=args.include_adversarial)
    if args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("No LOCOMO questions with evidence were found.")

    psm = None if args.skip_psm else PsmReranker(args.model_path)
    started = time.time()
    records: list[dict[str, Any]] = []

    for i, q in enumerate(questions, start=1):
        ranked = rank_mem0_style(q.question, q.candidates)
        mem0_ids = [c.dia_id for c, _ in ranked[: args.top_k]]
        psm_ids: list[str] = []
        psm_error: str | None = None

        if psm is not None:
            pool = [c for c, _ in ranked[: args.candidate_pool]]
            try:
                psm_ids = psm.select(q.question, pool, top_k=args.top_k)
            except Exception as exc:  # Keep the run going and report parse/runtime failures.
                psm_error = str(exc)

        if i % 10 == 0 or i == len(questions):
            print(f"processed {i}/{len(questions)} questions")

        records.append({
            "sample_id": q.sample_id,
            "category": q.category,
            "question": q.question,
            "answer": q.answer,
            "evidence": list(q.evidence),
            "mem0_ids": mem0_ids,
            "psm_ids": psm_ids,
            "psm_error": psm_error,
            "mem0_hit_at_1": hit_at(q.evidence, mem0_ids, 1),
            "mem0_hit_at_k": hit_at(q.evidence, mem0_ids, args.top_k),
            "psm_hit_at_1": hit_at(q.evidence, psm_ids, 1) if psm is not None else None,
            "psm_hit_at_k": hit_at(q.evidence, psm_ids, args.top_k) if psm is not None else None,
        })

    summary = summarize(records, top_k=args.top_k, elapsed_seconds=time.time() - started)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"locomo-results-{stamp}.json"
    md_path = out_dir / f"locomo-results-{stamp}.md"
    json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(render_markdown(summary))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def ensure_locomo(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="snap-research/locomo",
            repo_type="dataset",
            filename="data/locomo10.json",
        )
        path.write_bytes(Path(downloaded).read_bytes())
        return
    except Exception:
        pass

    try:
        from urllib.request import urlopen

        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    except Exception as exc:
        raise SystemExit(
            f"Could not download LOCOMO to {path}. Download data/locomo10.json from "
            "https://github.com/snap-research/locomo and rerun with --data."
        ) from exc


def load_questions(path: Path, include_adversarial: bool) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for sample in raw:
        sample_id = str(sample.get("sample_id", "unknown"))
        candidates = tuple(flatten_candidates(sample.get("conversation", {})))
        for qa in sample.get("qa", []):
            evidence = tuple(str(x) for x in qa.get("evidence", []) if str(x).strip())
            category = str(qa.get("category", "unknown"))
            if not evidence:
                continue
            if not include_adversarial and category in {"5", "adversarial"}:
                continue
            questions.append(Question(
                sample_id=sample_id,
                category=category,
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                evidence=evidence,
                candidates=candidates,
            ))
    return questions


def flatten_candidates(conversation: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    session_keys = sorted(
        (k for k, v in conversation.items() if re.fullmatch(r"session_\d+", k) and isinstance(v, list)),
        key=lambda x: int(x.split("_")[1]),
    )
    for session in session_keys:
        for turn in conversation.get(session, []):
            text = str(turn.get("text", "")).strip()
            dia_id = str(turn.get("dia_id", "")).strip()
            if not text or not dia_id:
                continue
            candidates.append(Candidate(
                dia_id=dia_id,
                session=session,
                speaker=str(turn.get("speaker", "")),
                text=text,
                index=len(candidates),
            ))
    return candidates


def rank_mem0_style(question: str, candidates: tuple[Candidate, ...]) -> list[tuple[Candidate, float]]:
    q_tokens = tokenize(question)
    q_counts = Counter(q_tokens)
    q_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question))
    scored: list[tuple[Candidate, float]] = []
    total = max(1, len(candidates) - 1)

    for c in candidates:
        text_tokens = tokenize(c.text)
        text_counts = Counter(text_tokens)
        overlap = sum(min(q_counts[t], text_counts[t]) for t in q_counts)
        denom = math.sqrt(sum(v * v for v in q_counts.values()) * sum(v * v for v in text_counts.values()) or 1)
        cosine = overlap / denom
        phrase_bonus = 0.15 if any(token in c.text.lower() for token in important_phrases(question)) else 0.0
        entity_bonus = 0.1 * sum(1 for entity in q_entities if entity in c.text)
        recency_bonus = 0.03 * (c.index / total)
        scored.append((c, cosine + phrase_bonus + entity_bonus + recency_bonus))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in STOPWORDS
    ]


def important_phrases(text: str) -> list[str]:
    tokens = tokenize(text)
    phrases = tokens[:]
    phrases.extend(" ".join(tokens[i:i + 2]) for i in range(max(0, len(tokens) - 1)))
    return phrases


def hit_at(evidence: tuple[str, ...], selected: list[str], k: int) -> bool:
    return bool(set(evidence).intersection(selected[:k]))


def summarize(records: list[dict[str, Any]], top_k: int, elapsed_seconds: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len(records),
        "top_k": top_k,
        "elapsed_seconds": elapsed_seconds,
        "overall": summarize_group(records),
        "by_category": {},
    }
    for category, group in sorted(group_by(records, "category").items()):
        summary["by_category"][category] = summarize_group(group)
    return summary


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_bool(key: str) -> float | None:
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            return None
        return statistics.mean(1.0 if v else 0.0 for v in values)

    mem0 = mean_bool("mem0_hit_at_k")
    psm = mean_bool("psm_hit_at_k")
    return {
        "n": len(records),
        "mem0_hit_at_1": mean_bool("mem0_hit_at_1"),
        "mem0_hit_at_k": mem0,
        "psm_hit_at_1": mean_bool("psm_hit_at_1"),
        "psm_hit_at_k": psm,
        "delta_hit_at_k": None if mem0 is None or psm is None else psm - mem0,
        "psm_failures": sum(1 for r in records if r.get("psm_error")),
    }


def group_by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key, "unknown"))].append(record)
    return groups


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [("Overall", summary["overall"])]
    rows.extend((f"Category {k}", v) for k, v in summary["by_category"].items())
    lines = [
        "# LOCOMO Evidence-Retrieval Results",
        "",
        f"Questions: {summary['question_count']}",
        f"Metric: Hit@{summary['top_k']} against LOCOMO annotated evidence turns",
        "",
        "| Split | n | Mem0 heuristic Hit@1 | Mem0 heuristic Hit@k | PSM Hit@1 | PSM Hit@k | Delta Hit@k | PSM failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['n']} | {pct(metrics['mem0_hit_at_1'])} | "
            f"{pct(metrics['mem0_hit_at_k'])} | {pct(metrics['psm_hit_at_1'])} | "
            f"{pct(metrics['psm_hit_at_k'])} | {signed_pct(metrics['delta_hit_at_k'])} | "
            f"{metrics['psm_failures']} |"
        )
    lines.extend([
        "",
        "Note: this is an evidence-retrieval benchmark, not final QA accuracy. It measures whether the memory layer surfaces the LOCOMO turns needed by a downstream answerer.",
    ])
    return "\n".join(lines)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f} pp"


class PsmReranker:
    def __init__(self, model_path: str) -> None:
        import onnxruntime_genai as og

        self.og = og
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def select(self, question: str, candidates: list[Candidate], top_k: int) -> list[str]:
        prompt = self._prompt(question, candidates, top_k)
        tokens = self.tokenizer.encode(prompt)
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(
            do_sample=False,
            top_k=20,
            top_p=1.0,
            temperature=0.0,
            max_length=min(32768, len(tokens) + 192),
        )
        generator = self.og.Generator(self.model, params)
        generator.append_tokens(tokens)
        stream = self.tokenizer.create_stream()
        output: list[str] = []
        while not generator.is_done():
            generator.generate_next_token()
            piece = stream.decode(generator.get_next_tokens()[0])
            output.append(piece)
            text = "".join(output)
            if "}" in text and text.count("{") <= text.count("}"):
                break
        payload = extract_json("".join(output))
        ids = payload.get("selected_ids")
        if not isinstance(ids, list):
            raise ValueError(f"PSM did not return selected_ids: {payload}")
        valid_ids = {c.dia_id for c in candidates}
        return [str(x) for x in ids if str(x) in valid_ids][:top_k]

    @staticmethod
    def _prompt(question: str, candidates: list[Candidate], top_k: int) -> str:
        candidate_lines = "\n".join(
            f"{c.dia_id} | {c.session} | {c.speaker}: {c.text}"
            for c in candidates
        )
        return (
            f"<|system|>\n{PSM_SYSTEM_PROMPT}\n"
            f"<|user|>\nSelect the {top_k} best memory candidates for this question.\n"
            f"Question: {question}\n\nCandidates:\n{candidate_lines}\n\n"
            f"Return JSON only.\n<|assistant|>\n"
        )


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in PSM output: {text[:200]}")
    return json.loads(text[start:end + 1])


if __name__ == "__main__":
    raise SystemExit(main())
