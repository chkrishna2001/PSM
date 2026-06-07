# PSM 50M model training

Single entry point for training the generative PSM storage model (`psm-model` package).

## Read first

1. [2026-06-07-end-of-day-handoff.md](2026-06-07-end-of-day-handoff.md) — **resume here** (Gate 4 @ 36k FAIL-close, v2 parse curriculum tomorrow)
2. [2026-06-06-end-of-day-handoff.md](2026-06-06-end-of-day-handoff.md) — Gate 2+3 PASS era
3. [runpod-ssh-ops.md](runpod-ssh-ops.md) — **RunPod / SSH / shell commands** (proxy pitfalls, `runpod_ctl.py`, eval, pull reports)
4. [2026-06-04-end-of-day-handoff.md](2026-06-04-end-of-day-handoff.md) — historical mixed-v1 era + LoCoMo/REALTALK data map
5. [training-playbook.md](training-playbook.md) — gates, commands, checkpoint denylist, local/HF/Colab workflow
6. [session-log.md](session-log.md) — rolling log of runs and gate results

## Product context (repo root)

- [../product-aligned-psm-context-plan.md](../product-aligned-psm-context-plan.md)
- [../psm-memory-product-plan.md](../psm-memory-product-plan.md)

## Nano-psm

The `nano-psm/` classifier path is **not** the product model. It may still be used for dataset generation via `data-pipeline/`. Training for the 50M product model uses `psm-model` only.

## Lessons from failed runs

See [archive/failed-runs-lessons.md](archive/failed-runs-lessons.md).
