param(
    [string]$Checkpoint = "psm-model\checkpoints\real-v3-50m-full-v2-step-048000.pt",
    [string]$Db = "benchmark\locomo\results\locomo-psm-model-step-048000-n25.db",
    [int]$Limit = 25,
    [int]$BatchSize = 5,
    [string]$Device = "cpu",
    [int]$WindowSize = 2
)

# Local LoCoMo stays on CPU — loading 50M on laptop GPU can OOM/crash the machine.
if ($Device -ne "cpu") {
    Write-Error "Local runs must use -Device cpu (requested: $Device). Use runpod_locomo.sh for GPU."
    exit 1
}

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) {
    ".venv\Scripts\python.exe"
} else {
    "python"
}

Push-Location $root
try {
    $env:PSM_FORCE_CPU = "1"
    npm run build
    $args = @(
        "dist\benchmark\locomo\src\ingest-psm-model.js",
        "--batch-size", "$BatchSize",
        "--db", $Db,
        "--checkpoint", $Checkpoint,
        "--device", $Device,
        "--python", $python,
        "--window-size", "$WindowSize",
        "--limit", "$Limit"
    )
    node @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $out = $Db -replace '\.db$', '-results.json'
    node dist\benchmark\locomo\src\evaluate.js --db $Db --out $out --top-k 3
} finally {
    Pop-Location
}
