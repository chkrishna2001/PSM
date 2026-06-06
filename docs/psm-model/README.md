# PSM 50M model training

Single entry point for training the generative PSM storage model (`psm-model` package).

## Read first

1. [2026-06-04-end-of-day-handoff.md](2026-06-04-end-of-day-handoff.md) — **resume here** after a break (checkpoints, commands, LoCoMo/REALTALK data map)
2. [training-playbook.md](training-playbook.md) — gates, commands, checkpoint denylist, local/HF/Colab workflow
3. [session-log.md](session-log.md) — rolling log of runs and gate results

## Product context (repo root)

- [../product-aligned-psm-context-plan.md](../product-aligned-psm-context-plan.md)
- [../psm-memory-product-plan.md](../psm-memory-product-plan.md)

## Nano-psm

The `nano-psm/` classifier path is **not** the product model. It may still be used for dataset generation via `data-pipeline/`. Training for the 50M product model uses `psm-model` only.

## Lessons from failed runs

See [archive/failed-runs-lessons.md](archive/failed-runs-lessons.md).
