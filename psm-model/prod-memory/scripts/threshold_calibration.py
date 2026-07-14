#!/usr/bin/env python3
"""Decision-threshold calibration on v11 (research-directed, arXiv:2409.19751 "Balancing the
Scales": threshold calibration is the most-effective + cheapest fix for class imbalance / the
negative(ignore)-bias we observed across every training lever).

No retraining. For each 100-case gate example we compute the model's P(store) in binary mode
(first-token probability of "store" vs "ignore"), then find the decision threshold that maximizes
accuracy. Measured HONESTLY with 5-fold CV (tune threshold on train folds, score held-out fold),
so we never tune-and-test on the same cases. Compares to the argmax (threshold=0.5) baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import torch  # noqa: E402
from prod_memory.eval_hf_grounding import open_hf_session  # noqa: E402
from prod_memory.hf_prompts import apply_chat_prompt, storage_inference_messages  # noqa: E402

GATE = ROOT / "fixtures" / "holdout-coding-agent-cases.json"
ADAPTER = ROOT / "checkpoints" / "hf-prod-storage-v11-qwen0.5b" / "adapter"


def _leading_token_ids(tok, words: list[str]) -> set[int]:
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tok(variant, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(enc[0])
    return ids


def p_store(session, llm_response: str, store_ids: set[int], ignore_ids: set[int]) -> float:
    messages = storage_inference_messages(llm_response, output_format="binary")
    prompt = apply_chat_prompt(messages, session.tokenizer)
    inputs = session.tokenizer(prompt, return_tensors="pt")
    dev = session._input_device()
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        out = session.model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    ps = float(sum(probs[i].item() for i in store_ids))
    pi = float(sum(probs[i].item() for i in ignore_ids))
    return ps / (ps + pi) if (ps + pi) > 0 else 0.0


def main() -> int:
    cases = json.loads(GATE.read_text(encoding="utf-8"))["cases"]
    session = open_hf_session(ADAPTER, model_key="qwen0.5b", device="cpu")
    tok = session.tokenizer
    store_ids = _leading_token_ids(tok, ["store"])
    ignore_ids = _leading_token_ids(tok, ["ignore"])
    print(f"store token ids: {store_ids} | ignore token ids: {ignore_ids}", flush=True)

    scored = []  # (p_store, is_store_label)
    for i, c in enumerate(cases):
        ps = p_store(session, c["llmResponse"], store_ids, ignore_ids)
        scored.append((ps, c["expectAction"] == "store"))
        if i % 20 == 0:
            print(f"scored {i}/{len(cases)}", flush=True)

    # baseline: argmax == threshold 0.5
    def acc_at(thr, subset):
        ok = sum(1 for ps, lab in subset if (ps >= thr) == lab)
        return ok / len(subset)

    base = acc_at(0.5, scored)
    # sweep candidate thresholds
    cand = [i / 100 for i in range(5, 96)]
    best_thr = max(cand, key=lambda t: acc_at(t, scored))
    best_full = acc_at(best_thr, scored)

    # HONEST 5-fold CV: tune threshold on train folds, score held-out fold
    import random
    random.seed(0)
    idx = list(range(len(scored)))
    random.shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    cv_correct = 0
    for f in range(5):
        test = [scored[i] for i in folds[f]]
        train = [scored[i] for i in idx if i not in folds[f]]
        thr = max(cand, key=lambda t: acc_at(t, train))
        cv_correct += sum(1 for ps, lab in test if (ps >= thr) == lab)
    cv_acc = cv_correct / len(scored)

    print(json.dumps({
        "baseline_argmax_0.5": round(base, 4),
        "best_threshold_full_fit": round(best_thr, 3),
        "best_acc_full_fit_OPTIMISTIC": round(best_full, 4),
        "cv_5fold_calibrated_acc_HONEST": round(cv_acc, 4),
        "n": len(scored),
    }, indent=2))
    out = ROOT / "results" / "v11-threshold-calibration.json"
    out.write_text(json.dumps({
        "baseline": base, "cv_acc": cv_acc, "best_thr": best_thr,
        "scores": [{"p_store": ps, "is_store": lab} for ps, lab in scored],
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
