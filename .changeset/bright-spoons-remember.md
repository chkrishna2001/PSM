---
"@psm-memory/sdk": minor
"@psm-memory/cli": minor
"@psm-memory/pi-plugin": minor
---

Add local agent hook installation, model-backed memory hooks, review logging, JSON repair retries, and vector-backed memory retrieval.

- Add Codex and Claude hook installers through `psm-memory install-agent`.
- Add hook audit logs and `review-log` for local user review of PSM behavior.
- Add model-authored context rendering so PSM produces final memory context items.
- Add retry repair prompts for invalid remember JSON before no-op fallback.
- Add local text embeddings with Hugging Face Transformers and vector candidate retrieval.
