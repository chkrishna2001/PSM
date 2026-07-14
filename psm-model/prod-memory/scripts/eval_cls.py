#!/usr/bin/env python3
"""Eval + threshold-calibration for the two-stage decision classifier (storage-cls-v1).

The classifier outputs reasoning-first, decision-only JSON: {"reasoning":..., "action":"store"|
"ignore"}. We report (1) raw greedy decision accuracy vs the 100-case gate, and (2) honest 5-fold
threshold-calibrated accuracy on P(store) extracted at the action token. Compare both to v11=0.82.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import torch  # noqa: E402
from prod_memory.eval_hf_grounding import open_hf_session  # noqa: E402
from prod_memory.hf_prompts import apply_chat_prompt, storage_inference_messages  # noqa: E402

GATE = ROOT / "fixtures" / "holdout-coding-agent-cases.json"
ADAPTER = ROOT / "checkpoints" / "hf-prod-storage-cls-v1-qwen0.5b" / "adapter"


def _first_ids(tok, words):
    ids = set()
    for w in words:
        enc = tok(w, add_special_tokens=False)["input_ids"]
        if enc:
            ids.add(enc[0])
    return ids


def run(session, llm_response, store_ids, ignore_ids, max_new=80):
    tok = session.tokenizer
    prompt = apply_chat_prompt(storage_inference_messages(llm_response, output_format="json"), tok)
    inputs = tok(prompt, return_tensors="pt")
    dev = session._input_device()
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        out = session.model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True, output_scores=True,
        )
    gen = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    # raw decision from generated text
    m = re.search(r'"action"\s*:\s*"(store|ignore)"', text)
    raw = m.group(1) if m else ("ignore" if '"ignore"' in text else "store" if '"store"' in text else None)
    # P(store) at the action-value token position
    marker = '"action":"'
    pos = text.find(marker)
    p = None
    if pos >= 0:
        prefix = text[: pos + len(marker)]
        for step in range(len(gen)):
            if len(tok.decode(gen[: step + 1], skip_special_tokens=True)) >= len(prefix):
                logits = out.scores[step][0].float()
                probs = torch.softmax(logits, dim=-1)
                ps = float(sum(probs[i].item() for i in store_ids))
                pi = float(sum(probs[i].item() for i in ignore_ids))
                p = ps / (ps + pi) if (ps + pi) > 0 else 0.5
                break
    return raw, (p if p is not None else 0.5), text


def main() -> int:
    cases = json.loads(GATE.read_text(encoding="utf-8"))["cases"]
    session = open_hf_session(ADAPTER, model_key="qwen0.5b", device="cpu")
    tok = session.tokenizer
    store_ids = _first_ids(tok, ["store"])
    ignore_ids = _first_ids(tok, ["ignore"])

    raw_ok = 0
    scored = []
    for i, c in enumerate(cases):
        raw, p, _ = run(session, c["llmResponse"], store_ids, ignore_ids)
        is_store = c["expectAction"] == "store"
        if raw is not None and (raw == "store") == is_store:
            raw_ok += 1
        scored.append((p, is_store))
        if i % 20 == 0:
            print(f"scored {i}/{len(cases)}", flush=True)

    def acc_at(thr, subset):
        return sum(1 for ps, lab in subset if (ps >= thr) == lab) / len(subset)

    import random
    random.seed(0)
    idx = list(range(len(scored)))
    random.shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    cand = [i / 100 for i in range(2, 99)]
    cv = 0
    for f in range(5):
        test = [scored[i] for i in folds[f]]
        train = [scored[i] for i in idx if i not in folds[f]]
        thr = max(cand, key=lambda t: acc_at(t, train))
        cv += sum(1 for ps, lab in test if (ps >= thr) == lab)

    result = {
        "raw_greedy_decision_acc": round(raw_ok / len(cases), 4),
        "prob_argmax_0.5_acc": round(acc_at(0.5, scored), 4),
        "cv_5fold_calibrated_acc": round(cv / len(scored), 4),
        "v11_baseline": 0.82,
        "n": len(cases),
    }
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "storage-cls-v1-eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
