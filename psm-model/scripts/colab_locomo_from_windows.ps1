# Launch HF LoCoMo on Colab via Ubuntu WSL (one-time OAuth inside WSL first).
param(
  [string]$CheckpointDb = "",
  [string]$Gpu = "T4",
  [string]$Session = "psm-locomo-hf"
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\chkri\source\repos\PSM"
if (-not $CheckpointDb) {
  $CheckpointDb = Join-Path $repo "benchmark\locomo\results\pod-sync\locomo-hf-prod-v5k-two-pass-n2960-checkpoint.db"
}

$hfToken = $env:HF_TOKEN
if (-not $hfToken) {
  try {
    $hfToken = (o krishnachhftoken -r 2>$null)
  } catch {}
}
if (-not $hfToken) {
  try {
    $hfToken = (& op read "op://Personal/krishnachhftoken/password" 2>$null)
  } catch {}
}
if (-not $hfToken) {
  Write-Error "Set HF_TOKEN or configure 1Password item krishnachhftoken"
}

$env:HF_TOKEN = $hfToken
$env:WSLENV = "HF_TOKEN/u:LOCOMO_OFFSET/u:LOCOMO_LIMIT/u"
$env:COLAB_GPU = $Gpu
$env:COLAB_SESSION = $Session

if ($env:COLAB_SMOKE -eq "1") {
  if (-not $env:LOCOMO_OFFSET) { $env:LOCOMO_OFFSET = "2960" }
  Write-Host "Colab SMOKE (3 turns)"
  $wslCheckpoint = (wsl -d Ubuntu-24.04 -e wslpath -a $CheckpointDb).Trim()
  wsl -d Ubuntu-24.04 -u chkri env LOCOMO_OFFSET="${env:LOCOMO_OFFSET:-2960}" LOCOMO_LIMIT="${env:LOCOMO_LIMIT:-3}" bash /mnt/c/Users/chkri/source/repos/PSM/psm-model/scripts/colab_smoke.sh $wslCheckpoint
  exit $LASTEXITCODE
}

Remove-Item Env:COLAB_SMOKE -ErrorAction SilentlyContinue
if (-not $env:LOCOMO_OFFSET) { $env:LOCOMO_OFFSET = "2963" }
Write-Host "Colab via WSL (session=$Session gpu=$Gpu offset=${env:LOCOMO_OFFSET})"
$wslCheckpoint = (wsl -d Ubuntu-24.04 -e wslpath -a $CheckpointDb).Trim()
wsl -d Ubuntu-24.04 -u chkri env LOCOMO_OFFSET="${env:LOCOMO_OFFSET:-2963}" LOCOMO_LIMIT="${env:LOCOMO_LIMIT:-0}" bash /mnt/c/Users/chkri/source/repos/PSM/psm-model/scripts/colab_locomo_launch.sh $wslCheckpoint
