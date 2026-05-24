param(
  [string]$Config = "nano-psm/configs/debug-4m.json",
  [string]$Train = "nano-psm/data/train.jsonl",
  [string]$Validation = "nano-psm/data/validation.jsonl",
  [string]$CheckpointDir = "nano-psm/checkpoints"
)

python nano-psm/src/nano_psm/train.py `
  --config $Config `
  --train $Train `
  --validation $Validation `
  --checkpoint-dir $CheckpointDir

