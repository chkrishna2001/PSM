#!/usr/bin/env python3
"""Threshold calibration on v11's REAL (JSON reasoning-first) decision, not the weak binary
proxy. For each case we greedily generate the reasoning-first JSON, locate the action-value
token (right after `"action":"`), and read P(store_episodic|promote_semantic) vs P(ignore) at
that position. Then honest 5-fold CV threshold calibration on that P(store).

If calibration adds points here (like the +7 it added in binary mode), 0.82 was an uncorrected
threshold/bias, not a capacity ceiling.
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


def _first_ids(tok, words):
    ids = set()
    for w in words:
        enc = tok(w, add_special_tokens=False)["input_ids"]
        if enc:
            ids.add(enc[0])
    return ids


def p_store_json(session, llm_response, store_ids, ignore_ids, max_new=80):
    tok = session.tokenizer
    messages = storage_inference_messages(llm_response, output_format="json")
    prompt = apply_chat_prompt(messages, tok)
    inputs = tok(prompt, return_tensors="pt")
    dev = session._input_device()
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        out = session.model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id, return_dict_in_generate=True, output_scores=True,
        )
    gen = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    # find the action-value token: the token generated right after the substring '"action":"'
    marker = '"action":"'
    pos = text.find(marker)
    if pos < 0:
        return None, text
    # count decoded chars to find which generated-token index starts the value
    prefix = text[: pos + len(marker)]
    # walk token by token accumulating decoded text until we pass the marker end
    acc = ""
    for step, tid in enumerate(gen.tolist()):
        acc = tok.decode(gen[: step + 1], skip_special_tokens=True)
        if len(acc) >= len(prefix):
            # scores[step] is the distribution that produced token `step`
            logits = out.scores[step][0].float()
            probs = torch.softmax(logits, dim=-1)
            ps = float(sum(probs[i].item() for i in store_ids))
            pi = float(sum(probs[i].item() for i in ignore_ids))
            return (ps / (ps + pi) if (ps + pi) > 0 else 0.0), text
    return None, text


def main() -> int:
    cases = json.loads(GATE.read_text(encoding="utf-8"))["cases"]
    session = open_hf_session(ADAPTER, model_key="qwen0.5b", device="cpu")
    tok = session.tokenizer
    # action values start with these tokens
    store_ids = _first_ids(tok, ["store", "store_episodic", "promote", "promote_semantic"])
    ignore_ids = _first_ids(tok, ["ignore"])
    print(f"store_ids={store_ids} ignore_ids={ignore_ids}", flush=True)

    scored = []
    skipped = 0
    for i, c in enumerate(cases):
        ps, _ = p_store_json(session, c["llmResponse"], store_ids, ignore_ids)
        if ps is None:
            skipped += 1
            ps = 0.5
        scored.append((ps, c["expectAction"] == "store"))
        if i % 20 == 0:
            print(f"scored {i}/{len(cases)} (skipped {skipped})", flush=True)

    def acc_at(thr, subset):
        return sum(1 for ps, lab in subset if (ps >= thr) == lab) / len(subset)

    base = acc_at(0.5, scored)
    cand = [i / 100 for i in range(2, 99)]
    best_thr = max(cand, key=lambda t: acc_at(t, scored))
    best_full = acc_at(best_thr, scored)

    import random
    random.seed(0)
    idx = list(range(len(scored)))
    random.shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    cv = 0
    for f in range(5):
        test = [scored[i] for i in folds[f]]
        train = [scored[i] for i in idx if i not in folds[f]]
        thr = max(cand, key=lambda t: acc_at(t, train))
        cv += sum(1 for ps, lab in test if (ps >= thr) == lab)
    cv_acc = cv / len(scored)

    print(json.dumps({
        "json_argmax_0.5": round(base, 4),
        "best_threshold_full_fit": round(best_thr, 3),
        "best_acc_full_fit_OPTIMISTIC": round(best_full, 4),
        "cv_5fold_calibrated_HONEST": round(cv_acc, 4),
        "skipped_no_action_token": skipped,
    }, indent=2))
    (ROOT / "results" / "v11-threshold-calibration-json.json").write_text(
        json.dumps({"baseline": base, "cv_acc": cv_acc, "best_thr": best_thr,
                    "scores": [{"p_store": ps, "is_store": lab} for ps, lab in scored]}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
