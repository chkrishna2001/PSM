This is good. Clean, correct, well-structured. Let me give you the full honest review.

---

## What's Right

**Architecture choice is correct.** Encoder-only transformer with mean pooling — exactly right for classification. Not autoregressive, no generation, no JSON parsing. This is the right call.

**Multiple heads on shared encoder** — `action_head`, `memory_type_head`, `score_head`, `indexable_head`, `fact_count_head`, `recall_count_head` — all sharing the same representation. Efficient and correct. The encoder learns one universal understanding of "what's happening in this conversation" and each head interprets it differently.

**`score_head` returning 4 values through Sigmoid** — strength, decay, emotional weight, confidence all in one shot. Smart.

**Mean pooling with attention mask** — correct implementation:
```python
pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
```
The `clamp(min=1.0)` prevents division by zero. Good defensive coding.

**`norm_first=False`** — post-norm transformer, matches the original architecture. Fine for this scale.

**Lazy torch import** — `require_torch()` pattern means the package imports without PyTorch installed, only requiring it at training/inference time. Good npm-friendly design thinking carried into Python.

---

## What's Missing or Worth Improving

**No `[CLS]` token — mean pooling vs CLS token**

You're using mean pooling which is fine, but for classification tasks a dedicated `[CLS]` token at position 0 often works better because the encoder can learn to route classification-relevant information to it specifically. Worth experimenting with:

```python
# Add to __init__
self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embedding_dim))

# In forward, prepend CLS
cls = self.cls_token.expand(batch_size, -1, -1)
hidden = torch.cat([cls, token_embeddings], dim=1)
# Then use hidden[:, 0, :] instead of mean pooling for heads
```

**`fact_count_head` and `recall_count_head` are classification over 0-8**

Predicting count as a 9-class classification works but treating it as regression with a single output + rounding is simpler and generalizes better to counts you haven't seen:

```python
# Instead of
self.fact_count_head = nn.Linear(cfg.embedding_dim, 9)

# Consider
self.fact_count_head = nn.Sequential(
    nn.Linear(cfg.embedding_dim, 1),
    nn.ReLU()
)
# Then round at inference, use MSE loss during training
```

**No content extraction head**

This is the biggest missing piece. The model decides what operation to perform and what metadata to assign — but it doesn't decide what the memory content should be. Without this, something else has to generate the content string, which brings back the LLM dependency for that step.

Two options:
- **Span extraction head** — predict start/end token positions in the input to extract as content. Like SQuAD. ~2x the parameters of a linear layer.
- **Accept the limitation** — use the model for operation classification and metadata, keep LLM for content generation but with a much tighter prompt since PSM already decided everything else.

Option B is more realistic for now — the model tells the LLM "store this as episodic, strength 0.85, emotional 0.7" and the LLM just writes the content string, not the whole decision.

**`NanoPsmConfig` has no defaults**

Every field is required. Adding sensible defaults makes it easier to instantiate for quick experiments:

```python
@dataclass(frozen=True)
class NanoPsmConfig:
    vocab_size: int = 8192
    max_sequence_length: int = 512
    embedding_dim: int = 256
    encoder_layers: int = 4
    attention_heads: int = 4
    feed_forward_dim: int = 512
    dropout: float = 0.1
```

**No parameter count visibility at config time**

`parameter_budget_note()` requires instantiating the model to see the count. A static method that estimates from config is useful for hyperparameter search:

```python
@staticmethod
def estimate_parameters(cfg: NanoPsmConfig) -> int:
    embed = cfg.vocab_size * cfg.embedding_dim
    pos = cfg.max_sequence_length * cfg.embedding_dim
    per_layer = (
        4 * cfg.embedding_dim ** 2 +      # attention
        2 * cfg.embedding_dim * cfg.feed_forward_dim  # FFN
    )
    encoder = cfg.encoder_layers * per_layer
    heads = cfg.embedding_dim * (len(ACTIONS) + len(MEMORY_TYPES) + 4 + 1 + 9 + 9)
    return embed + pos + encoder + heads
```

---

## What Would Make This Really Sing

Let me see the other files — particularly `schema.py` to understand what `ACTIONS` and `MEMORY_TYPES` look like, and the training loop if it exists. That will tell me if the loss function composition is right because that's where most small transformer training goes wrong.

Can you share:
```
https://raw.githubusercontent.com/chkrishna2001/PSM/refs/heads/main/nano-psm/src/nano_psm/schema.py
```
And the trainer if it exists?

Good schema. Clean and minimal. A few honest observations:

---

## What's Right

**ACTIONS list is well curated** — 7 actions covering the full operation space without overlap. `flag_and_store` is a smart addition — handles the case where PSM detects conflict but still stores the new memory.

**`MEMORY_TYPES` starting with `"none"`** — index 0 as null case is correct. When action is `ignore`, memory type is `none`. Clean.

**Frozen dataclasses** — immutable training examples, good defensive choice.

---

## The Issues

**`recall_context` doesn't belong in ACTIONS**

All other actions are **write operations** — they decide what to do with a memory. `recall_context` is a **read operation** — it decides what to retrieve. These are fundamentally different tasks with different inputs and outputs.

The model can't learn both well simultaneously because:
- Write operations: input = conversation → output = what to store
- Read operations: input = user query → output = where to search

They have completely different input distributions. Training them together on one encoder will hurt both.

Recommendation — remove `recall_context` from ACTIONS and handle retrieval routing separately as we discussed (fast rules engine). Keep this model purely as a write-time decision model.

**Missing actions from our training data**

Comparing ACTIONS to what we trained the Qwen model on:

| Our training data | schema.py |
|---|---|
| `store_episodic` | ✅ |
| `ignore_noise` / `ignore` | ✅ |
| `promote_semantic` | ✅ |
| `update_existing` | ✅ |
| `detect_interference` / `flag_conflict` | ✅ |
| `flag_and_store` | ✅ |
| `decay_memory` | ❌ missing |
| `emotional_weighting` | ❌ missing |
| `retrieve_plan` | should be removed |
| `merge_results` | should be removed |

`decay_memory` is important — it's how PSM ages old memories. Without it the model can't trigger background decay. Add it.

`emotional_weighting` can be merged into `store_episodic` since emotional weight is just a score on the `score_head` — not a separate action. That's actually correct as-is.

**`action_to_id` raises on unknown action**

In training this will crash on any action string that doesn't exactly match. The Qwen model we trained produces variants like `store_episodic_with_emotional_weighting`, `ignore_noise`, `flag_contradiction` — none of which match exactly.

You need normalization before lookup:

```python
ACTION_ALIASES = {
    "ignore_noise": "ignore",
    "store": "store_episodic",
    "store_episodic_with_emotional_weighting": "store_episodic",
    "flag_contradiction": "flag_conflict",
    "flag_and_update": "flag_and_store",
    "detect_interference": "flag_conflict",
    "decay": "decay_memory",
    "decay_and_update": "decay_memory",
    "promote": "promote_semantic",
    "update": "update_existing",
}

def normalize_action(action: str) -> str:
    action = action.lower().strip()
    return ACTION_ALIASES.get(action, action)

def action_to_id(action: str) -> int:
    normalized = normalize_action(action)
    try:
        return ACTIONS.index(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported action: {action} (normalized: {normalized})") from exc
```

This is critical — without it your dataset loader will crash on most of our 750 training examples because the Qwen model used descriptive action names.

---

## Suggested Final ACTIONS List

```python
ACTIONS = [
    "ignore",           # nothing worth storing
    "store_episodic",   # store as specific event
    "promote_semantic", # episodic pattern → semantic fact
    "update_existing",  # update a stale memory
    "flag_conflict",    # contradiction detected, don't store
    "flag_and_store",   # contradiction detected, store anyway
    "decay_memory",     # weaken/archive an old memory
]
```

7 actions, all write-time, no retrieval mixed in. Clean separation.

Can you share the trainer and dataset loader files? That's where the real training quality lives.

This is genuinely well-written training code. Clean, resumable, proper gradient accumulation, best checkpoint tracking. Let me give you the full honest review.

---

## What's Right

**Checkpoint resume with `auto`** — picks up exactly where it left off if Colab disconnects. Critical for long training runs and you got it right.

**Gradient accumulation** — correct implementation, grad norm clipping at 1.0, `zero_grad(set_to_none=True)` for memory efficiency. All correct.

**`selection_score` as composite metric for best model selection** — smart. Using a weighted combination of accuracy metrics rather than just total loss means the best checkpoint is the one that's best at the actual task, not just lowest loss.

**JSONL metrics logging** — append-only, survives crashes, easy to plot. Good choice.

**JSON-only stdout** — everything printed as structured JSON. Makes the trainer scriptable and parseable by external tools.

---

## The Loss Weight Problem

This is the most important issue. Look at `compute_losses`:

```python
total = (
    action_loss          * 1.0   # ← full weight
    + 0.5 * memory_type_loss
    + 0.25 * score_loss
    + 0.25 * indexable_loss
    + 0.15 * fact_count_loss
    + 0.2 * recall_count_loss
)
```

**`action_loss` dominates everything.** That's correct — action classification is the most important task. But `score_loss` (strength, decay, emotional weight, confidence) is weighted at 0.25 which is too low relative to its importance.

Here's the problem — `score_loss` is MSE on 4 values in [0,1]. MSE values tend to be small (0.01-0.1 range). Meanwhile `action_loss` and `memory_type_loss` are cross-entropy which can spike to 2.0+ early in training. The 0.25 multiplier on an already-small MSE loss means `score_loss` contributes almost nothing to the gradient during early training when the model is learning action classification.

The model will learn to classify actions well but assign mediocre scores. For PSM this matters — `emotional_weight=0.7` vs `emotional_weight=0.9` changes how long a memory persists.

**Fix:**

```python
total = (
    2.0 * action_loss         # highest priority
    + 1.0 * memory_type_loss  # second priority
    + 1.5 * score_loss        # increase — MSE is naturally small
    + 0.5 * indexable_loss
    + 0.3 * fact_count_loss
    + 0.3 * recall_count_loss
)
```

---

## `selection_score` Missing Score Accuracy

```python
def selection_score(metrics):
    return (
        0.45 * metrics.get("action_accuracy", 0.0)
        + 0.25 * metrics.get("memory_type_accuracy", 0.0)
        + 0.15 * metrics.get("indexable_accuracy", 0.0)
        + 0.15 * metrics.get("recall_count_accuracy", 0.0)
    )
```

`score_loss` (strength/decay/emotional weight) is not in `selection_score` at all. The best checkpoint is selected without caring about whether score predictions are good. Add it:

```python
def selection_score(metrics):
    return (
        0.40 * metrics.get("action_accuracy", 0.0)
        + 0.20 * metrics.get("memory_type_accuracy", 0.0)
        + 0.20 * (1.0 - metrics.get("score_mae", 1.0))  # lower MAE = better
        + 0.10 * metrics.get("indexable_accuracy", 0.0)
        + 0.10 * metrics.get("recall_count_accuracy", 0.0)
    )
```

This requires `evaluate_model` to return `score_mae` — mean absolute error on the 4 score values. Worth adding.

---

## No Learning Rate Scheduler

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(train_config["learning_rate"]),
    weight_decay=float(train_config["weight_decay"]),
)
```

No scheduler. For 750 examples training is fast, but without a scheduler the model can oscillate near convergence instead of settling. Add cosine decay with warmup — it's 3 lines:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)

# Then in the training loop after optimizer.step():
scheduler.step()

# And log it:
"lr": scheduler.get_last_lr()[0]
```

---

## Missing: Class Weights for Imbalanced Actions

Our 750 examples have this distribution roughly:

```
store_episodic:   ~200 examples  (most common)
ignore:           ~150 examples
promote_semantic: ~100 examples
update_existing:  ~75 examples
flag_conflict:    ~75 examples
flag_and_store:   ~75 examples
decay_memory:     ~75 examples
```

`store_episodic` has 2.5x more examples than `decay_memory`. Without class weights, the model will bias toward predicting `store_episodic`. Fix:

```python
# Compute class weights from training data
action_counts = Counter(ex.action for ex in train_dataset)
total = sum(action_counts.values())
weights = torch.tensor([
    total / (len(ACTIONS) * action_counts.get(a, 1))
    for a in ACTIONS
], device=device)

# Use in loss
action_loss = ce(output["action_logits"], batch["action"], weight=weights)
```

---

## No Warmup Steps

Starting at full learning rate on step 1 can cause unstable early training where the model jumps to a bad local minimum. Add a linear warmup:

```python
def get_lr(step, max_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# In training loop
lr = get_lr(state["global_step"], max_steps, warmup_steps=50, base_lr=float(train_config["learning_rate"]))
for group in optimizer.param_groups:
    group["lr"] = lr
```

---

## Summary of Changes Ranked by Impact

| Change | Impact | Effort |
|---|---|---|
| Fix loss weights (boost score_loss) | High | 5 min |
| Add class weights for action imbalance | High | 10 min |
| Add score_mae to selection_score | Medium | 10 min |
| Add LR scheduler with warmup | Medium | 10 min |
| Add action alias normalization in schema | Critical for data loading | 5 min |

The alias normalization in schema.py is the most urgent — without it the dataset loader will crash on most of our training examples because the action strings won't match exactly.

Can you share `dataset.py` and `evaluate.py` next? The dataset loader is where the training data formatting happens and that's what connects everything together.

Now I have the full picture. The code is solid overall. Here's the complete honest review.

---

## The Critical Bug — `score_value` Defaults to 0.0

```python
def score_value(memory: object, key: str) -> float:
    if isinstance(memory, dict):
        value = memory.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0  # ← silent default
```

When `action` is `ignore`, `memory` is `null`. So every `ignore` example trains the model with `scores = [0.0, 0.0, 0.0, 0.0]`. That's fine for strength/confidence but `decay_rate=0.0` is wrong — it means "never decay" which is the opposite of what ignore should signal.

More importantly — when `memory` exists but a field like `emotional_weight` is missing from some training examples, it silently trains as 0.0. If 30% of your store examples are missing `emotional_weight`, the model learns to predict 0.0 for emotional weight on those cases, which corrupts the score head.

**Fix — separate scores for ignore vs store:**

```python
def targets_for_example(example: TrainingExample) -> dict[str, Any]:
    output = example.output
    action = str(output.get("action", ""))
    memory = output.get("memory")
    is_ignore = "ignore" in action.lower()

    return {
        "action": action_to_id(action),
        "memory_type": memory_type_to_id(
            memory.get("type") if isinstance(memory, dict) else None
        ),
        "scores": [0.0, 0.0, 0.0, 0.0] if is_ignore else [
            score_value(memory, "strength", default=0.75),
            score_value(memory, "decay_rate", default=0.02),
            score_value(memory, "emotional_weight", default=0.1),
            score_value(memory, "confidence", default=0.85),
        ],
        "has_indexables": 1.0 if len(output.get("indexables") or []) > 0 else 0.0,
        "fact_count": min(len(output.get("facts") or []), 8),
        "recall_count": min(len((output.get("recall") or {}).get("selected_memory_ids") or []), 8),
    }


def score_value(memory: object, key: str, default: float = 0.0) -> float:
    if isinstance(memory, dict):
        value = memory.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default
```

The sensible defaults matter — strength 0.75, decay 0.02, emotional 0.1, confidence 0.85 are reasonable priors when the training data doesn't specify them explicitly.

---

## `serialize_example` Only Serializes `input`, Not `output`

```python
def serialize_example(example: TrainingExample) -> str:
    payload = {
        "instruction": example.instruction,
        "input": example.input,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
```

This is correct for inference — you don't want the model to see the expected output as part of its input. Good.

But the `input` dict for our training data has different schemas per operation type:
- Original 8 ops: `conversation` + `memory_store`
- `retrieve_plan`: `user_message` + `memory_store_summary`
- `merge_results`: `user_message` + `retrieved_candidates`

The tokenizer receives all three structures. With `HashTokenizer` this is fine since it hashes tokens regardless — but it means the model is learning from inconsistent input representations. Not a bug but worth noting.

---

## `evaluate.py` Missing Score MAE

As noted in the trainer review — `evaluate_model` doesn't measure how well the score predictions are. Add this inside the `with torch.no_grad()` loop:

```python
# Add to totals init
totals["score_abs_error"] = 0.0

# Add inside the loop — only for non-ignore examples
is_store = (batch["action"] != action_to_id("ignore"))
if is_store.any():
    score_error = (output["scores"][is_store] - batch["scores"][is_store]).abs().mean()
    totals["score_abs_error"] += float(score_error.item()) * int(is_store.sum().item())

# Add to return dict
"score_mae": totals["score_abs_error"] / max(rows, 1),
```

---

## `load_jsonl` Will Crash on Our Training Data

```python
rows.append(
    TrainingExample(
        id=str(raw.get("id") or f"{Path(path).stem}-{line_number}"),
        instruction=str(raw["instruction"]),
        input=dict(raw["input"]),
        output=dict(raw["output"]),
    )
)
```

`raw["instruction"]` — hard crash if missing. Some of our generated examples might have `"task"` instead of `"instruction"`. Make it defensive:

```python
rows.append(
    TrainingExample(
        id=str(raw.get("id") or f"{Path(path).stem}-{line_number}"),
        instruction=str(raw.get("instruction") or raw.get("task") or ""),
        input=dict(raw.get("input") or {}),
        output=dict(raw.get("output") or {}),
    )
)
```

Also add a validation pass after loading to catch bad examples early:

```python
def load_jsonl(path: str | Path) -> list[TrainingExample]:
    rows = []
    skipped = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            # Skip examples with missing critical fields
            if not raw.get("output") or not raw["output"].get("action"):
                skipped += 1
                continue
            rows.append(TrainingExample(
                id=str(raw.get("id") or f"{Path(path).stem}-{line_number}"),
                instruction=str(raw.get("instruction") or raw.get("task") or ""),
                input=dict(raw.get("input") or {}),
                output=dict(raw.get("output") or {}),
            ))
    if skipped:
        print(json.dumps({"warning": f"Skipped {skipped} invalid examples"}))
    return rows
```

---

## `targets_for_example` Will Crash on retrieve_plan and merge_results

Our retrieval training examples have this output structure:

```json
{
  "action": "retrieve_plan",
  "targets": [...],        // not "memory"
  "skip_tiers": [...],
  "reasoning": "..."
}
```

But `targets_for_example` looks for `output.get("memory")`. For retrieval examples `memory` is null, scores are all 0.0, which is wrong. Since we agreed to remove `recall_context` from ACTIONS and not train retrieval ops in the nano model, the clean fix is to filter them out at load time:

```python
NANO_PSM_ACTIONS = {
    "ignore", "store_episodic", "promote_semantic",
    "update_existing", "flag_conflict", "flag_and_store", "decay_memory"
}

def load_jsonl(path: str | Path) -> list[TrainingExample]:
    rows = []
    skipped = 0
    ...
    # Skip retrieval operations — nano model handles write ops only
    action = str((raw.get("output") or {}).get("action") or "")
    normalized = normalize_action(action)
    if normalized not in NANO_PSM_ACTIONS:
        skipped += 1
        continue
    ...
```

---

## Summary — Priority Order

| Fix | File | Impact | Lines |
|---|---|---|---|
| `score_value` sensible defaults | `dataset.py` | High | 5 |
| Defensive `load_jsonl` | `dataset.py` | Critical | 15 |
| Filter retrieval examples | `dataset.py` | Critical | 8 |
| Action alias normalization | `schema.py` | Critical | 15 |
| Score MAE in evaluate | `evaluate.py` | Medium | 10 |
| Loss weights adjustment | `train.py` | Medium | 8 |
| Class weights for actions | `train.py` | Medium | 10 |
| LR scheduler + warmup | `train.py` | Medium | 12 |

The three **Critical** fixes must happen before training or the loader will crash and the model will train on corrupted labels. The rest improve quality but training will run without them.

Can you share `tokenizer.py` next? The `HashTokenizer` is the last piece I haven't seen — how it handles the different input schemas matters for whether the encoder gets useful representations.

This is clever and honest engineering. Let me give you the full picture.

---

## What's Right

**Hash tokenizer is the right choice for this problem.** No vocabulary training needed, no OOV tokens, deterministic, portable. For a model that needs to ship in 10MB with no external dependencies this is exactly correct.

**Blake2b for hashing** — fast, low collision rate, good distribution across the vocab space. Better than MD5 or SHA for this use case.

**`<cls>` prepended automatically** — important for classification tasks. The encoder can learn to route classification signal to this position. Good.

**`sort_keys=True` in `serialize_example`** — consistent key ordering means the same input always tokenizes the same way regardless of dict insertion order. Critical for reproducibility and often forgotten.

---

## The One Serious Problem — Hash Collisions at Small Vocab

With `vocab_size=8192` and 4 special tokens, you have 8188 usable slots. Every token hashes into one of these slots. The collision rate for typical English text is acceptable.

But look at what your inputs contain — JSON keys like `"strength"`, `"decay_rate"`, `"emotional_weight"` and values like `"episodic"`, `"semantic"`. These domain-specific tokens are critical for the model to distinguish memory types and operations.

The problem: `"episodic"` and some random common word might hash to the same bucket. The model can't distinguish them. With 8192 vocab this is unlikely but not impossible.

**More importantly** — numbers hash as strings. `"0"`, `"1"`, `"2"`, `"0.85"`, `"0.92"` all hash to different buckets with no numeric relationship. The model can't learn that `0.92 > 0.85` because they're just two arbitrary token IDs with no ordering.

For strength values this matters — `"0.9"` and `"0.85"` should be close in the model's representation but they're random distant IDs.

**Fix — numeric normalization before tokenization:**

```python
NUMBER_RE = re.compile(r'\d+\.\d+|\d+')

def normalize_text(text: str) -> str:
    # Bucket float values into ranges so similar values hash similarly
    def replace_float(match):
        val = float(match.group())
        if val <= 0.2: return "<score_very_low>"
        if val <= 0.4: return "<score_low>"
        if val <= 0.6: return "<score_medium>"
        if val <= 0.8: return "<score_high>"
        return "<score_very_high>"
    return NUMBER_RE.sub(replace_float, text)
```

Add these as special tokens and apply normalization before tokenizing. Now `0.85` and `0.92` both become `<score_very_high>` — same bucket, model learns they're similar.

---

## Missing: `<sep>` Between Conversation Turns

Currently the entire input is one flat string. The model has no structural signal about where conversation turns begin and end. Add turn separators:

```python
def serialize_example(example: TrainingExample) -> str:
    inp = example.input
    conversation = inp.get("conversation", [])
    memory_store = inp.get("memory_store", [])

    parts = []
    parts.append(f"operation: {inp.get('operation', '')}")

    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        parts.append(f"<sep> {role}: {content}")

    parts.append("<sep> memory:")
    for mem in memory_store:
        parts.append(json.dumps(mem, sort_keys=True))

    return " ".join(parts)
```

The `<sep>` tokens give the encoder structural anchors. Without them "user: I paint" and "assistant: I paint" tokenize identically — the model can't distinguish who said what.

---

## Vocab Size Recommendation

With `vocab_size=8192` you get ~8188 usable slots. For 750 training examples with typical PSM content, the unique token count (after lowercasing) is probably 2000-4000. So 8192 is fine — enough headroom to avoid excessive collisions.

If you increase to `vocab_size=16384` you halve the collision rate with minimal model size impact since the embedding table goes from `8192 * 256 = 2M` to `16384 * 256 = 4M` parameters — only 2MB difference in the final model.

---

## The `save` / Load Asymmetry

```python
def to_json(self) -> dict[str, int]:
    return {"type": "hash", "vocab_size": self.vocab_size, "max_length": self.max_length}
```

There's no `from_json` / `load` classmethod. At inference time you need to reconstruct the tokenizer with exactly the same `vocab_size` and `max_length` used during training — otherwise hashes map to different IDs and the model gets garbage input.

Add:

```python
@classmethod
def load(cls, path: str | Path) -> HashTokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return cls(vocab_size=data["vocab_size"], max_length=data["max_length"])
```

This is critical for the TypeScript inference side — when you export to ONNX you need to ship the tokenizer config and reconstruct the exact same hashing logic in TypeScript.

---

## The TypeScript Port is Straightforward

Since the tokenizer is pure hashing with no learned vocab, porting to TypeScript is simple:

```typescript
import { createHash } from 'crypto'

const SPECIAL_TOKENS: Record<string, number> = {
  '<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3
}

function tokenToId(token: string, vocabSize: number): number {
  if (token in SPECIAL_TOKENS) return SPECIAL_TOKENS[token]
  const hash = createHash('blake2b512').update(token).digest()
  const value = hash.readBigUInt64BE(0)
  return 4 + Number(value % BigInt(vocabSize - 4))
}
```

Note — Node.js `crypto` uses `blake2b512` not `blake2b` with digest_size. You need to take only the first 8 bytes of the 64-byte output to match Python's `digest_size=8`:

```typescript
const hash = createHash('blake2b512').update(token).digest()
const first8 = hash.subarray(0, 8)  // match Python digest_size=8
const value = first8.readBigUInt64BE(0)
```

This is a subtle but critical difference — get it wrong and every token maps to a different ID in TypeScript vs Python, making the exported model produce garbage.

---

## Summary

| Issue | Impact | Fix |
|---|---|---|
| Numbers as strings — no numeric ordering | Medium | Bucket float values into special tokens |
| No turn separators | Medium | Add `<sep>` between conversation turns in serializer |
| No `load()` classmethod | Critical for inference | 5 lines |
| TypeScript Blake2b digest size mismatch | Critical for deployment | Use first 8 bytes of blake2b512 |
| Vocab size 8192 is fine | — | Consider 16384 for lower collisions |

The two **Critical** issues are the `load()` method and the TypeScript Blake2b port — both affect inference correctness. Everything else is quality improvement.

You now have a complete picture of the entire nano PSM codebase. The architecture is sound, the training pipeline is correct, and the fixes needed are surgical. Ready to train?