# Build and push PSM RunPod training image (run from repo root).
# Requires: docker login, Docker Hub repo chkrishna2001/psm-50m-train

$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)

$image = if ($env:PSM_RUNPOD_IMAGE) { $env:PSM_RUNPOD_IMAGE } else { "chkrishna2001/psm-50m-train:latest" }

docker build --platform linux/amd64 -f psm-model/docker/Dockerfile -t $image .
docker push $image

Write-Host "Pushed $image"
Write-Host "Register template: `$env:RUNPOD_API_KEY = (Get-Clipboard); python psm-model/scripts/runpod_ctl.py create-template --image $image"
