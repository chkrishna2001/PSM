# PSM Memory Autoresearch Program

Goal: improve a small structured PSM model for memory management, mnemonic/indexable creation, and recall-context selection.

The agent may edit training code, model architecture, loss weights, sampling balance, and evaluation scripts, but must not weaken schema validation.

Primary dataset:

```powershell
node nano-psm\data-pipeline\src\generate-dataset.mjs `
  --locomo benchmark\locomo\data\locomo10.json `
  --out nano-psm\data-pipeline\data\generated `
  --limit 500 `
  --recall-limit 250 `
  --synthetic-count 250
```

Source mix:

- PersonaMem: preference extraction and latent identity/profile memories.
- LoCoMo: long-term episodic continuity and temporal recall.
- LongMemEval: updates, contradictions, temporal reasoning, and abstention.
- REALTALK: noisy real-world multi-day conversation.
- PerLTQA: typed semantic/episodic/profile/event memory organization.
- User Preference 564K: large preference extraction bootstrapping.

Run source setup first:

```powershell
node nano-psm\data-pipeline\src\prepare-external-sources.mjs
```

Hard gates:

- `validate-examples.mjs` must pass.
- Output JSON must use only canonical keys.
- Facts must be evidence-backed.
- Memories must not use generic `User` when a named speaker is present.
- Recall tasks must select memory ids/indexable keys, not produce ungrounded answers.

Optimization metrics:

- schema_valid_rate
- action_accuracy
- speaker_grounding_accuracy
- current_turn_grounding_rate
- fact_precision
- temporal_resolution_accuracy
- indexable_key_quality
- recall_context_hit_at_k
- unsupported_fact_rate
- generic_user_leakage_rate

Research directions:

- Compare encoder-only classifier/span-extractor against small generative fine-tune.
- Add class-balanced sampling so ignore examples do not dominate.
- Train indexable generation as a separate head or auxiliary JSON field.
- Weight temporal and recall-context losses higher than easy ignore decisions.
- Try curriculum: schema first, memory extraction second, indexables third, recall-context last.
- Export the best small model to ONNX for TypeScript inference.
