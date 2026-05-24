# PSM Memory Fine-Tune Data

This folder contains the first reproducible dataset-preparation pipeline for PSM memory fine-tuning.

The goal is to train and evaluate the small local model against the exact product contract:

- canonical JSON output
- speaker-aware memory extraction
- current-turn grounding
- temporal extraction
- factual extraction with evidence
- mnemonic/indexable creation for compact recall cues
- recall-context selection from memory rows and indexable keys
- update/conflict/ignore behavior
- developer/project memory use cases

## Quick Start

Prepare external source directories and adapter notes:

```powershell
node nano-psm\data-pipeline\src\prepare-external-sources.mjs
```

Generate the first dataset from local LOCOMO plus curated developer examples:

```powershell
node nano-psm\data-pipeline\src\generate-dataset.mjs `
  --locomo benchmark\locomo\data\locomo10.json `
  --out nano-psm\data-pipeline\data\generated `
  --limit 500 `
  --recall-limit 250 `
  --synthetic-count 250
```

Validate generated JSONL:

```powershell
node nano-psm\data-pipeline\src\validate-examples.mjs `
  nano-psm\data-pipeline\data\generated\train.jsonl `
  nano-psm\data-pipeline\data\generated\validation.jsonl
```

Generate local PSM/indexables examples:

```powershell
node nano-psm\data-pipeline\src\inspect-local-psm-sources.mjs `
  "$env:LOCALAPPDATA\psm-memory\psm-memory.db"

node nano-psm\data-pipeline\src\generate-local-psm-dataset.mjs `
  --out nano-psm\data-pipeline\data\generated-local-psm `
  --db "$env:LOCALAPPDATA\psm-memory\psm-memory.db" `
  --docs docs\indexables-conv.txt `
  --max-db-rows 100 `
  --max-doc-examples 25
```

Validate runtime compatibility:

```powershell
node nano-psm\data-pipeline\src\validate-runtime-compat.mjs `
  nano-psm\data-pipeline\data\generated-local-psm\train.jsonl `
  nano-psm\data-pipeline\data\generated-local-psm\validation.jsonl
```

Run the complete quality gate:

```powershell
node nano-psm\data-pipeline\src\gate-dataset.mjs `
  --train nano-psm\data-pipeline\data\generated-local-psm\train.jsonl `
  --validation nano-psm\data-pipeline\data\generated-local-psm\validation.jsonl `
  --out nano-psm\data-pipeline\reports\gate-local-psm
```

The gate runs:

- canonical schema validation
- runtime compatibility validation
- dataset quality audit
- review sample generation

Do not upload or train on data that fails this gate.

Prepare a Hugging Face-ready folder after the gate passes:

```powershell
node nano-psm\data-pipeline\src\prepare-hf-dataset.mjs `
  --data-dir nano-psm\data-pipeline\data\generated `
  --report-dir nano-psm\data-pipeline\reports\gate-current `
  --out hf-upload\nano-psm-training-data `
  --name nano-psm-starter-gated
```

Upload with the HF CLI after login:

```powershell
hf upload chkrishna2001/nano-psm `
  hf-upload\nano-psm-merged-starter `
  . `
  --repo-type dataset
```

The generated files are intentionally ignored by git. Commit scripts, schemas, and notebooks, not local datasets.

## Output Files

```text
nano-psm/data-pipeline/data/generated/
  train.jsonl
  validation.jsonl
  all.jsonl
  metadata.json
```

Each JSONL row has:

```json
{
  "instruction": "Perform the PSM memory operation...",
  "input": {},
  "output": {}
}
```

The output schema now includes `indexables` on every row. Stored memories get compact mnemonic/fact anchors; recall rows use `action:"recall_context"` and train the model to select grounded `selected_memory_ids` and `selected_indexable_keys` instead of answering from general knowledge.

## Autoresearch

Karpathy-style autoresearch is a good fit after the dataset and validation metric are stable. Use `autoresearch-program.md` as the research prompt for an agent loop that experiments with model architecture, loss weighting, class balance, span extraction, and indexable quality while preserving the schema gate.

## Source Roles

The source manifest is `sources/psm-source-manifest.json`.

| Source | Training Role |
|---|---|
| Local PSM | real PSM decisions, stored memories, benchmark failures, and indexables concepts |
| PersonaMem | preference extraction and latent identity/profile memories |
| LoCoMo | long-term episodic continuity and temporal recall |
| LongMemEval | updates, contradictions, temporal reasoning, abstention |
| REALTALK | noisy real-world multi-day conversation |
| PerLTQA | typed semantic/episodic/profile/event memory organization |
| User Preference 564K | large preference extraction bootstrapping |

## Notebook

Use `psm-memory-fine-tune-colab.ipynb` to generate/upload data and run a first LoRA fine-tune in Colab.

The notebook is intentionally parameterized. Set the base model, HF output repo, and dataset repo before training.

Use `../../nano-psm/notebooks/nano-psm-data-and-train-colab.ipynb` for Nano PSM dataset upload, checkpoint resume, and debug/primary model training.
