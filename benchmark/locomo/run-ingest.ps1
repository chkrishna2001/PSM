param(
    [string]$Model = "models\psm-q4_k_m.gguf",
    [string]$Db = "benchmark\locomo\results\locomo-psm-memory.db",
    [int]$Limit = 0,
    [int]$BatchSize = 4,
    [int]$Port = 8080,
    [int]$WindowSize = 2,
    [switch]$UseGpu
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$serverExe = "C:\Users\chkri\source\repos\llama.cpp\build\bin\Release\llama-server.exe"
$log = Join-Path $root "benchmark\locomo\results\llama-server.log"
$err = Join-Path $root "benchmark\locomo\results\llama-server.err.log"

$args = @(
    "-m", $Model,
    "-c", "4096",
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--cache-ram", "0",
    "--no-warmup"
)

if ($UseGpu) {
    $args += @("-ngl", "999")
}

$server = Start-Process `
    -FilePath $serverExe `
    -ArgumentList $args `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $log `
    -RedirectStandardError $err `
    -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 1
        if ($server.HasExited) {
            throw "llama-server exited before becoming ready. See $err"
        }
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch {
        }
    }

    if (-not $ready) {
        throw "llama-server did not become ready. See $err"
    }

    $ingestArgs = @(
        "dist\benchmark\locomo\src\ingest.js",
        "--batch-size", "$BatchSize",
        "--db", $Db,
        "--server", "http://127.0.0.1:$Port",
        "--window-size", "$WindowSize"
    )
    if ($Limit -gt 0) {
        $ingestArgs += @("--limit", "$Limit")
    }

    node @ingestArgs
} finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
