# Quick status for prod-extraction-v3 teacher labeling build.
$root = "C:\Users\chkri\source\repos\PSM\psm-model\prod-memory\data"
$cache = Join-Path $root "prod-teacher-cache.jsonl"
$log = Join-Path $root "prod-teacher-v3-build.log"
$out = Join-Path $root "prod-extraction-v3.jsonl"
$manifest = Join-Path $root "prod-extraction-v3.manifest.json"

$cacheLines = 0
if (Test-Path $cache) { $cacheLines = (Get-Content $cache | Measure-Object -Line).Lines }

$lastProgressLine = ""
if (Test-Path $log) {
    $lines = Get-Content $log | Where-Object { $_ -match 'teacher_progress' }
    if ($lines) { $lastProgressLine = $lines[-1] }
}

$buildRunning = $false
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.CommandLine -like '*build_prod_extraction_v2*') { $script:buildRunning = $true }
}

$status = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    build_running = $buildRunning
    cache_rows = $cacheLines
    total_sessions = 1475
    pct = [math]::Round(100 * $cacheLines / 1475, 1)
    last_progress = $lastProgressLine
    v3_jsonl_exists = (Test-Path $out)
    v3_manifest_exists = (Test-Path $manifest)
}
$status | ConvertTo-Json -Compress
