import json, re

data = json.load(open("benchmark/locomo/data/locomo10.json"))
tot_turns = tot_ev = 0
for s in data:
    conv = s["conversation"]
    turns = [
        t
        for k, v in conv.items()
        if re.match(r"session_\d+$", k) and isinstance(v, list)
        for t in v
    ]
    ev = set()
    for q in s.get("qa") or []:
        for e in q.get("evidence") or []:
            ev.add(str(e))
    dia_ids = {str(t.get("dia_id")) for t in turns}
    ev_present = ev & dia_ids
    tot_turns += len(turns)
    tot_ev += len(ev_present)
    pct = 100 * len(ev_present) / max(1, len(turns))
    print(f"{s['sample_id']}: turns={len(turns)} evidence={len(ev_present)} ({pct:.0f}%)")
print(f"TOTAL turns={tot_turns} evidence={tot_ev} ({100*tot_ev/tot_turns:.1f}%)")
