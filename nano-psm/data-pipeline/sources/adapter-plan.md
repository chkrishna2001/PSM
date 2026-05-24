# PSM External Source Adapter Plan

| Source | Role | Adapter | Target Operations |
|---|---|---|---|
| PersonaMem | preference_extraction_latent_identity | personamem | promote_semantic, ignore, recall_context |
| LoCoMo | long_term_episodic_continuity | locomo | store_episodic, promote_semantic, recall_context, ignore |
| LongMemEval | updates_contradiction_handling | longmemeval | update_existing, flag_conflict, flag_and_store, recall_context |
| REALTALK | noisy_real_world_conversation | realtalk | ignore, store_episodic, promote_semantic, recall_context |
| PerLTQA | typed_memory_organization | perltqa | promote_semantic, store_episodic, recall_context |
| User Preference 564K | preference_extraction_bootstrapping | user_preference_564k | promote_semantic, ignore |

Adapter contract:

1. Read raw source records from `expected_raw_dir`.
2. Normalize them into `{ instruction, input, output }` rows.
3. Preserve source ids and evidence ids in `input`.
4. Emit canonical PSM output keys only.
5. Add `indexables` for every stored memory.
6. Use `recall_context` rows for retrieval/reconstruction tasks.
7. Validate with `validate-examples.mjs`.

Do not train directly on source QA answers as memory ingestion labels. Convert them into recall-context selection labels so PSM learns what memory rows and indexable keys to activate.
