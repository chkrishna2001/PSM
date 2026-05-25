# Nano PSM Claude Feedback Fix Plan

This plan captures the fixes accepted from `docs/claude-opinion.md` and the items intentionally deferred.

## Scope

Nano PSM currently trains on both write-time memory operations and `recall_context` rows. The data pipeline, schema validation, compatibility checks, and reports all include `recall_context`, so this plan keeps the existing single-model contract intact.

## Fix Now

1. Add action normalization.
   - Accept common generated aliases such as `ignore_noise`, `flag_contradiction`, and `store_episodic_with_emotional_weighting`.
   - Preserve `recall_context` as a supported action.

2. Make JSONL loading defensive.
   - Accept `instruction` or `task`.
   - Default missing `input` and `output` to empty dictionaries.
   - Skip rows with missing or unsupported actions instead of crashing late in batching.

3. Protect score targets from silent bad zeros.
   - Keep zero score targets for non-memory rows such as `ignore` and `recall_context`.
   - Use durable-memory priors when a stored-memory row omits score fields.

4. Measure score prediction quality.
   - Add validation `score_mae`.
   - Include score quality in checkpoint selection.

5. Rebalance training loss.
   - Increase score loss weight because MSE is naturally smaller than cross-entropy losses.
   - Keep action prediction as the highest-priority task.

6. Add action class weights.
   - Compute action weights from the loaded training examples.
   - Use them in the action cross-entropy loss to reduce majority-class bias.

7. Improve tokenizer persistence and numeric handling.
   - Add `HashTokenizer.load()`.
   - Bucket decimal score-like numbers while preserving integers such as dates, ids, and counts.

## Defer

1. Do not remove `recall_context`.
   - Removing it would require a deliberate split of dataset generation, schemas, gates, evaluation, and runtime integration.

2. Do not switch count heads from classification to regression.
   - Counts are bounded to 0-8 and exact-count accuracy is already evaluated.

3. Do not add a content extraction head.
   - Durable memory content often requires normalization, not only span extraction.

4. Do not switch model pooling to CLS yet.
   - It is a valid experiment, but not a clear correctness fix.

5. Do not rewrite serialization into a new structured format in this patch.
   - That changes the training distribution and should be paired with a deliberate retrain/evaluation run.

## Validation

Run the focused Python checks that do not require a full training job:

```powershell
python -m compileall nano-psm\src\nano_psm
```

If PyTorch is installed, run a smoke evaluation/training command against the current gated train/validation files.
