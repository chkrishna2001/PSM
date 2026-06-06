# Build Gate-2 mixed curriculum v2: storage + direct-behavior (8x) + manual-probe anchor (300x).
# Run from repo root: .\psm-model\scripts\build_mixed_curriculum_v2.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$env:PYTHONPATH = "psm-model/src"
$Py = ".\.venv\Scripts\python.exe"
$Tok = "psm-model\checkpoints\real-v3-50m-action-mixed-v1-step-000400.tokenizer.json"

if (-not (Test-Path $Tok)) {
    $Tok = "psm-model\tokenizers\real-v1-pattern.json"
}

& $Py -m psm_model.make_action_first_curriculum `
    psm-model\data\curriculum\psm-50m-action-direct-v2.jsonl `
    psm-model\data\direct-behavior-v1\train.jsonl --copies 8

& $Py -m psm_model.make_action_first_curriculum `
    psm-model\data\curriculum\psm-50m-manual-anchor-v2.jsonl `
    psm-model\data\direct-behavior-v1\manual-probe.jsonl --copies 300

& $Py -m psm_model.combine_jsonl `
    psm-model\data\curriculum\psm-50m-action-mixed-v2.jsonl `
    psm-model\data\curriculum\psm-50m-action-first-v1-filtered-ctx2048.jsonl `
    psm-model\data\curriculum\psm-50m-action-direct-v2.jsonl `
    psm-model\data\curriculum\psm-50m-manual-anchor-v2.jsonl

& $Py -m psm_model.filter_by_token_budget `
    psm-model\data\curriculum\psm-50m-action-mixed-v2.jsonl `
    psm-model\data\curriculum\psm-50m-action-mixed-v2-ctx2048.jsonl `
    --tokenizer $Tok --max-tokens 2049 --output-format action

Write-Host "Built psm-model\data\curriculum\psm-50m-action-mixed-v2-ctx2048.jsonl"
