param(
    [string]$Checkpoint = "psm-model\checkpoints\real-v3-50m-full-v2-step-058000.pt",
    [int]$Limit = 25,
    [string]$Device = "auto",
    [string]$Step = "058000",
    [switch]$SkipAnswer
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$stem = "locomo-psm-model-step-$Step-n$Limit"
$db = "benchmark\locomo\results\$stem.db"
$quality = "benchmark\locomo\results\$stem-quality.json"
$retrieval = "benchmark\locomo\results\$stem-retrieval.json"
$answers = "benchmark\locomo\results\$stem-answer-results.json"
$debug = "benchmark\locomo\results\$stem-ingest-debug.json"

Push-Location $root
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $env:PSM_ALLOW_LOCAL_GPU = "1"
    node dist\benchmark\locomo\src\ingest-psm-model.js `
        --input-format psm `
        --device $Device `
        --checkpoint $Checkpoint `
        --db $db `
        --limit $Limit `
        --debug-raw `
        --debug-out $debug
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    node dist\benchmark\locomo\src\ingest-quality-check.js `
        --db $db `
        --ingest-limit $Limit `
        --out $quality
    $qualityExit = $LASTEXITCODE

    node dist\benchmark\locomo\src\evaluate.js `
        --db $db `
        --out $retrieval `
        --top-k 3
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($qualityExit -ne 0) { exit $qualityExit }

    if (-not $SkipAnswer) {
        if (-not $env:OPENROUTER_API_KEY -and -not $env:OPENAI_API_KEY) {
            Write-Error "OPENROUTER_API_KEY is required for answer evaluation."
            exit 1
        }
        node dist\benchmark\locomo\src\answer-evaluate.js `
            --db $db `
            --checkpoint $Checkpoint `
            --device $Device `
            --out $answers `
            --limit $Limit `
            --answerable-only `
            --checkpoint-every 1
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} finally {
    Pop-Location
}
