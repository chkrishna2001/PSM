param(
    [string]$Model = "models\psm-q4_k_m.gguf",
    [string]$Db = "benchmark\locomo\results\locomo-psm-memory-node.db",
    [int]$Limit = 0,
    [int]$BatchSize = 25,
    [string]$Gpu = "false",
    [string]$GpuLayers = "0",
    [int]$ContextSize = 4096,
    [int]$WindowSize = 2
)

# All GPU work belongs on RunPod. Local LoCoMo ingest stays CPU-only.
if ($Gpu -notin @("false", "cpu", "0") -or ($GpuLayers -notin @("0", "false", "cpu"))) {
    Write-Error "Local GPU is disabled. Use RunPod for GPU LoCoMo/training/eval (Gpu=false, GpuLayers=0 only)."
    exit 1
}

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

Push-Location $root
try {
    $args = @(
        "dist\benchmark\locomo\src\ingest-node.js",
        "--batch-size", "$BatchSize",
        "--db", $Db,
        "--model", $Model,
        "--gpu", $Gpu,
        "--gpu-layers", $GpuLayers,
        "--context-size", "$ContextSize",
        "--window-size", "$WindowSize"
    )
    if ($Limit -gt 0) {
        $args += @("--limit", "$Limit")
    }
    node @args
} finally {
    Pop-Location
}
