param(
  [string]$Train = "nano-psm/data/train.jsonl",
  [string]$Validation = "nano-psm/data/validation.jsonl"
)

node benchmark/fine-tune-data/src/validate-examples.mjs $Train $Validation

