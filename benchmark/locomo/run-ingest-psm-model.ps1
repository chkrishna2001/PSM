param(
    [string]$Checkpoint = "psm-model\checkpoints\real-v3-50m-full-v2.pt",
    [string]$Db = "benchmark\locomo\results\locomo-psm-model-smoke.db",
    [int]$Limit = 25,
    [int]$BatchSize = 5,
    [string]$Device = "cpu",
    [int]$WindowSize = 2
)

# Local runs must stay on CPU; use RunPod for GPU eval/training.
if ($Device -ne "cpu") {
    Write-Error "Local GPU is disabled (requested: $Device). Use RunPod via runpod_ctl.py for cuda."
    exit 1
}

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

Push-Location $root
try {
    npm run build
    $args = @(
        "dist\benchmark\locomo\src\ingest-psm-model.js",
        "--batch-size", "$BatchSize",
        "--db", $Db,
        "--checkpoint", $Checkpoint,
        "--device", $Device,
        "--window-size", "$WindowSize",
        "--limit", "$Limit"
    )
    node @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    node dist\benchmark\locomo\src\evaluate.js --db $Db --out benchmark\locomo\results\locomo-psm-model-smoke-results.json --top-k 3
} finally {
    Pop-Location
}
