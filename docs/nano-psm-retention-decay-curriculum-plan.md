# Nano PSM Retention And Decay Curriculum Plan

## Goal

Train Nano PSM to make useful retention decisions, not just memory-action decisions.

Current training already predicts four memory scores:

- `strength`
- `decay_rate`
- `emotional_weight`
- `confidence`

But recent datasets mostly use fixed defaults, so the model learns only a shallow rule:

```text
semantic memory -> lower decay
episodic memory -> higher decay
```

The next curriculum should teach when to forget, decay quickly, retain strongly, archive, update, or flag uncertainty.

## Product Behavior We Want

Nano PSM should learn these distinctions:

```text
ignore
  no durable memory should be written

short_lived / high_decay
  useful briefly, but should fade quickly

normal_episodic
  dated event or session result worth keeping with normal decay

durable_semantic
  stable preference, project rule, identity, or workflow habit

critical_low_decay
  safety, access, environment, user preference, or architecture fact that should persist

update_existing
  new information replaces an older memory

flag_conflict
  uncertain contradiction should not overwrite the old memory

flag_and_store
  certain correction conflicts with old memory and should be stored

archive_candidate
  old but still potentially useful memory should leave active recall but remain searchable
```

The model does not delete rows directly. It predicts structured memory scores and actions. Runtime logic must use those outputs with timestamps and access counts to decay, archive, or suppress recall.

## Target Label Policy

Use consistent target ranges so the score head learns meaningful gradients.

### Ignore

Use for:

- greetings
- acknowledgements
- command output with no durable fact
- one-off formatting requests
- unsafe/toxic requests with no user preference
- generic web/article generation instructions

Target:

```json
{
  "action": "ignore",
  "memory": null,
  "facts": [],
  "indexables": []
}
```

No score target is trained for ignore rows.

### Short-Lived Memory

Use for:

- temporary task state
- "today only" instructions
- active debugging lane status
- current command/session observations
- transient checkpoint/logistics state

Target ranges:

```text
memory.type: episodic
strength: 0.45 - 0.65
decay_rate: 0.12 - 0.30
emotional_weight: 0.05 - 0.30
confidence: 0.75 - 0.95
```

Example:

```json
{
  "action": "store_episodic",
  "memory": {
    "content": "On 2026-05-27, training run 18 is waiting for validation output upload.",
    "type": "episodic",
    "strength": 0.55,
    "decay_rate": 0.18,
    "emotional_weight": 0.10,
    "confidence": 0.92
  }
}
```

### Normal Episodic Memory

Use for:

- benchmark result
- session outcome
- dated project event
- concrete user event with future relevance

Target ranges:

```text
memory.type: episodic
strength: 0.70 - 0.85
decay_rate: 0.04 - 0.08
emotional_weight: 0.15 - 0.45
confidence: 0.85 - 0.98
```

### Durable Semantic Memory

Use for:

- stable preference
- repeated workflow habit
- project architecture decision
- durable environment constraint
- durable user or team rule

Target ranges:

```text
memory.type: semantic
strength: 0.80 - 0.92
decay_rate: 0.01 - 0.03
emotional_weight: 0.10 - 0.45
confidence: 0.85 - 0.98
```

### Critical Low-Decay Memory

Use for:

- security/access constraint
- hardware/environment limitation
- user preference that affects many future responses
- product rule that should not be forgotten
- strong correction to a dangerous or expensive assumption

Target ranges:

```text
memory.type: semantic
strength: 0.90 - 0.98
decay_rate: 0.001 - 0.01
emotional_weight: 0.25 - 0.70
confidence: 0.90 - 0.99
```

Example:

```json
{
  "action": "promote_semantic",
  "memory": {
    "content": "Do not assume PSM users have GPU access; many office VMs are CPU-only.",
    "type": "semantic",
    "strength": 0.96,
    "decay_rate": 0.005,
    "emotional_weight": 0.45,
    "confidence": 0.96
  }
}
```

### Update Existing

Use when new information clearly replaces old memory.

Target ranges:

```text
memory.type: semantic or episodic depending on content
strength: 0.82 - 0.95
decay_rate: 0.01 - 0.05
confidence: 0.90 - 0.99
updates: required
```

Example:

```json
{
  "action": "update_existing",
  "memory": {
    "content": "Dataset run 42 now targets 10k gated rows before primary training.",
    "type": "semantic",
    "strength": 0.88,
    "decay_rate": 0.02,
    "emotional_weight": 0.25,
    "confidence": 0.96
  },
  "updates": [
    {
      "target_id": "dataset-run-42",
      "relationship": "replaces",
      "reason": "New target supersedes the prior 1k compatibility-row target."
    }
  ]
}
```

### Flag Conflict

Use when new information conflicts with old memory but is uncertain.

Target ranges:

```text
strength: 0.55 - 0.75
decay_rate: 0.04 - 0.12
confidence: 0.55 - 0.75
conflicts: required
```

The lower confidence and higher decay should teach the model not to overcommit.

### Flag And Store

Use when the user gives a clear correction and the corrected information is durable.

Target ranges:

```text
strength: 0.85 - 0.98
decay_rate: 0.005 - 0.03
confidence: 0.92 - 0.99
conflicts: required
```

### Archive Candidate

The current action schema does not include `archive_candidate`.

For now, represent this as:

```text
action: store_episodic or promote_semantic
strength: medium
decay_rate: medium-high
tags: include archive_candidate
```

Later add a first-class action only if runtime supports it.

## Dataset Shape

Create a dedicated dataset:

```text
nano-psm-retention-decay-5k
```

Recommended initial size:

```text
5,000 rows
train: 4,285
validation: 715
```

Action/retention mix:

```text
ignore: 20%
short_lived episodic: 15%
normal_episodic: 15%
durable_semantic: 18%
critical_low_decay: 8%
update_existing: 10%
flag_conflict: 7%
flag_and_store: 7%
```

Source mix:

```text
synthetic_retention: 45%
local_psm_project_state: 20%
user_preference_564k reviewed subset: 15%
realtalk recall/noise: 10%
prior reviewed incremental patterns: 10%
```

Avoid PersonaMem for this curriculum unless rows are manually curated. The recent review showed PersonaMem creates awkward or context-dependent labels.

## Generator Design

Create a new generator or add a mode:

```text
nano-psm/data-pipeline/src/generate-retention-decay-dataset.mjs
```

Recommended functions:

```text
generateIgnoreRetentionRows()
generateShortLivedRows()
generateNormalEpisodicRows()
generateDurableSemanticRows()
generateCriticalLowDecayRows()
generateUpdateExistingRows()
generateFlagConflictRows()
generateFlagAndStoreRows()
generateArchiveCandidateRows()
```

Each row should include:

- explicit evidence text
- compact memory content
- meaningful indexables for stored memory
- tags that expose the retention reason
- score values from the target ranges above
- no unsupported facts

## Example Scenario Families

### Temporary Debug State

Inputs:

```text
This run is using checkpoint-temp.pt just for tonight.
Use the throwaway API key only for this smoke test.
The current Colab runtime has 12GB RAM available.
```

Expected:

```text
store_episodic
high decay_rate
low/medium strength
tags: temporary, debug, session_state
```

### Durable Project Rule

Inputs:

```text
Always use the hf CLI directly; do not use deprecated huggingface-cli.
The office VM is CPU-only, so CUDA cannot be assumed.
Reviewed datasets must be gated and sampled before training.
```

Expected:

```text
promote_semantic
low decay_rate
high strength
tags: project_rule, durable
```

### User Preference Retention

Inputs:

```text
I prefer concise implementation answers with exact file paths.
For training reports, show action mix and failure buckets first.
```

Expected:

```text
promote_semantic
low decay_rate
high confidence
```

### One-Off Formatting

Inputs:

```text
Write this response in pirate style.
Make this paragraph sound more dramatic.
Give me three title ideas for this article.
```

Expected:

```text
ignore
```

Unless phrased as a durable preference:

```text
From now on, keep release notes concise.
```

### Conflict And Correction

Inputs:

```text
Actually, dataset run 42 targets 10k rows, not 1k.
I may have been wrong; deployment might still require VPN.
No, use checkpoint-best.pt, not checkpoint-last.pt.
```

Expected:

```text
update_existing for certain replacement
flag_conflict for uncertain contradiction
flag_and_store for clear correction that should persist
```

### Forget Or Archive

Inputs:

```text
That temporary benchmark path is obsolete now.
The migration branch was deleted after merge.
This old Colab runtime detail is no longer relevant.
```

Expected:

For current schema:

```text
update_existing or flag_conflict when tied to an old memory
ignore when no durable memory is needed
archive_candidate tag when useful historically
```

## Review Gates

Run the standard gate:

```powershell
node nano-psm\data-pipeline\src\gate-dataset.mjs `
  --train nano-psm\data-pipeline\data\retention-decay-5k\train.jsonl `
  --validation nano-psm\data-pipeline\data\retention-decay-5k\validation.jsonl `
  --out nano-psm\data-pipeline\reports\gate-retention-decay-5k
```

Add retention-specific DuckDB checks:

```sql
-- No stored rows missing memory.
select count(*)
from read_json_auto('nano-psm/data-pipeline/data/retention-decay-5k/all.jsonl')
where output.action not in ('ignore', 'recall_context')
  and output.memory.content is null;

-- Score ranges by tag/action.
select
  output.action,
  output.memory.type,
  min(output.memory.decay_rate),
  max(output.memory.decay_rate),
  avg(output.memory.decay_rate),
  count(*)
from read_json_auto('nano-psm/data-pipeline/data/retention-decay-5k/all.jsonl')
where output.memory.content is not null
group by 1, 2
order by 1, 2;

-- Critical rows should have low decay.
select count(*)
from read_json_auto('nano-psm/data-pipeline/data/retention-decay-5k/all.jsonl')
where list_contains(output.memory.tags, 'critical')
  and output.memory.decay_rate > 0.02;

-- Temporary rows should have high decay.
select count(*)
from read_json_auto('nano-psm/data-pipeline/data/retention-decay-5k/all.jsonl')
where list_contains(output.memory.tags, 'temporary')
  and output.memory.decay_rate < 0.10;
```

Manual review requirements:

- inspect all gate warnings
- sample at least 3 rows per retention class
- sample at least 3 rows per action
- verify old-memory update/conflict rows include `memory_store`
- verify `ignore` rows truly have no durable preference
- verify one-off style requests are not promoted
- verify "from now on" instructions are promoted

## Training Plan

Do not train from zero.

Use the best existing reviewed checkpoint as the base:

```text
chkrishna2001/nano-psm-primary-10m-reviewed-5k-checkpoints
```

Then continue on the retention dataset:

```text
dataset repo:
chkrishna2001/nano-psm-retention-decay-5k

checkpoint repo:
chkrishna2001/nano-psm-primary-10m-retention-decay-from-reviewed-checkpoints
```

Recommended Colab settings:

```text
MAX_STEPS: 3000
EVAL_EVERY: 250
SAVE_EVERY: 500
resume: auto
```

Do not overtrain this curriculum. It is a behavior fine-tune, not a full replacement dataset.

## Evaluation Plan

Current evaluation reports only aggregate `score_mae`, which hides which score is wrong.

Add per-score evaluation:

```text
strength_mae
decay_rate_mae
emotional_weight_mae
confidence_mae
```

Add retention behavior checks:

```text
temporary_decay_accuracy
critical_low_decay_accuracy
durable_semantic_decay_accuracy
conflict_confidence_band_accuracy
ignore_oneoff_accuracy
promote_from_now_on_accuracy
```

Minimum acceptance for first pass:

```text
action_accuracy >= 96%
decay_rate_mae <= 0.035
temporary_decay_accuracy >= 90%
critical_low_decay_accuracy >= 90%
ignore_oneoff_accuracy >= 95%
no regression larger than 2 points on reviewed-5k action accuracy
```

## Runtime Follow-Up

Training decay is useful only if runtime consumes it.

Runtime should eventually compute effective recall weight:

```text
effective_score =
  model_strength
  * confidence
  * recency_decay(timestamp, decay_rate)
  * access_boost(access_count, last_accessed)
  * conflict_penalty
```

Suggested initial runtime policy:

```text
decay_rate <= 0.01
  durable; rarely suppress by age

0.01 < decay_rate <= 0.05
  normal memory; decay slowly

0.05 < decay_rate <= 0.12
  medium retention; rank down after stale

decay_rate > 0.12
  short-lived; suppress from normal recall after stale
```

Archive policy should be separate from deletion:

- suppress stale high-decay memories from default recall
- keep them searchable in archival recall
- never delete without an explicit user or retention policy decision

## Tomorrow Execution Checklist

1. Create `generate-retention-decay-dataset.mjs`.
2. Generate local `retention-decay-5k`.
3. Run standard gate.
4. Run retention-specific DuckDB checks.
5. Manually review warning rows and retention/action samples.
6. Patch generator until:
   - failures = 0
   - warnings = 0 or explicitly accepted
   - score ranges match policy
7. Upload dataset to:

```text
chkrishna2001/nano-psm-retention-decay-5k
```

8. Create Colab notebook:

```text
nano-psm/notebooks/nano-psm-retention-decay-from-reviewed-colab.ipynb
```

9. Continue from:

```text
chkrishna2001/nano-psm-primary-10m-reviewed-5k-checkpoints
```

10. Upload retention checkpoint to:

```text
chkrishna2001/nano-psm-primary-10m-retention-decay-from-reviewed-checkpoints
```

11. Compare:

- reviewed-5k validation
- incremental-5k validation
- retention-decay validation
- score MAE by score component

12. Only keep the retention checkpoint if it improves decay behavior without materially hurting action/memory classification.
