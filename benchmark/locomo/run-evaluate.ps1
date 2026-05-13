param(
    [string]$Db = "benchmark\locomo\results\locomo-psm-memory.db",
    [string]$Out = "benchmark\locomo\results\locomo-results.json",
    [int]$TopK = 3
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

Push-Location $root
try {
    node dist\benchmark\locomo\src\evaluate.js --db $Db --out $Out --top-k $TopK
} finally {
    Pop-Location
}
