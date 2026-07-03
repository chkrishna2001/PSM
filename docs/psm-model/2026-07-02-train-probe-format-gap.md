# Train vs probe format gap — v5h/v5m LoCoMo (2026-07-02)

Trained on Codex dict-conversation; probed on flat LoCoMo session-metadata. Parses JSON but stores headers not facts. v5m LoCoMo rows helped D1:3, regressed elsewhere.

## 1. Format mismatch

| Source | Total | Dict `conversation[]` | LoCoMo flat | Other |
|--------|------:|----------------------:|------------:|------:|
| `prod-extraction-v3.jsonl` | 2,654 | 1,494 | **0** | 1,160 |
| Storage-only | 1,504 | ~1,494 | **0** | ~10 |

| Curriculum | Rows | LoCoMo-shaped (incl. copies) |
|------------|-----:|-------------------------------:|
| `hf-prod-v5h.jsonl` | 1,664 | 283 |
| `hf-prod-v5m.jsonl` | 1,724 | 283 (+60 ignore / v5m store rows) |

**Train** (`remember_target_from_input`) — joined assistant dumps:
```
Current BMAD state in this repo:
- Completed artifacts found: `Create PRD` ...
```

**Probe** (`build_locomo_remember_text`) — never in v3:
```
Source id: conv-26:D1:3
Session time: 1:56 pm on 8 May, 2023
Current utterance: "I went to a LGBTQ support group yesterday..."
Previous context: ...
```

---

## 2. Probe results (10 turns, JSON)

| Metric | v5h | v5m |
|--------|----:|----:|
| parse_ok | 90% | 80% |
| store_match | 80% | 60% |
| has_facts | 100% | 70% |
| has_temporal | 0% | 0% |

| Case | v5h | v5m | Delta |
|------|-----|-----|-------|
| conv-26 D1:3 | store (session meta) | store (**LGBTQ**) | content ↑ |
| conv-26 D1:8 | store | ignore | REGRESSED |
| conv-26 D3:4 | store | ignore | REGRESSED |
| ignore ×3 | 1/3 correct | 1/3 correct | same |

v5h D1:3 memory: *"Session 1 started at 1:56 pm…"* — misses QA gold (7 May 2023).

---

## 3. In-domain sanity (v5h, training format)

5 v3 rows (`_indomain_v5h_sanity.py`): 0/5 parse/store — `failed_safe` at 256 tok; probe uses 384. Run `eval_hf_grounding` for clean baseline.

---

## 4. Root causes

1. **Zero LoCoMo format in v3** — probe headers are OOD; model fixates on `Session time` / `Source id`.
2. **Wrong extraction** — stores metadata, not utterance facts (LGBTQ, adoption, etc.).
3. **v5m weak LoCoMo labels** — heuristic one-liners, not teacher → action calibration drift.
4. **No temporal in teacher** — 0% temporal on all probes.
5. **Ignore failure** — greetings still stored (both checkpoints).
6. **0.5B limits** — JSON parse OK; content/action quality weak.

---

## 5. Recommendations (ranked)

1. **Teacher-label LoCoMo remember-text** (GPT-4o, full StorageDecision + temporal) → 15–25% curriculum mix.
2. **Align format** — probe as `conversation[]` *or* add headers to teacher pipeline; one canonical shape.
3. **Stop heuristic LoCoMo rows** (v5m `build_locomo_v5m_store_rows`).
4. **Re-probe aligned format** before GPU; bar: parse ≥95%, evidence content correct, temporal >0%.
5. **Fixture eval in-domain** (`eval_hf_grounding`, 384 tok).
6. **Fresh train** from teacher cache + LoCoMo teacher only if above fails.

**Do not:** LoCoMo ingest with v5h/v5m as-is; more unlabeled LoCoMo copies; another minimal_extract pass.
