#!/usr/bin/env python3
import os
import sqlite3

paths = [
    "/content/PSM/benchmark/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db",
    "/content/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db",
]
for p in paths:
    if os.path.isfile(p):
        c = sqlite3.connect(p)
        n = c.execute("select count(*) from decisions").fetchone()[0]
        print(f"{p} size={os.path.getsize(p)} decisions={n}")
    else:
        print(f"{p} MISSING")

log = "/content/locomo/results/ingest.log"
if os.path.isfile(log):
    with open(log, encoding="utf-8") as fh:
        lines = fh.readlines()
    print(f"ingest.log lines={len(lines)}")
    for line in lines[-8:]:
        print(line.rstrip())
