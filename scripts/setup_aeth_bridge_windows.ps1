[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu",
    [string]$RepoPath = "",
    [switch]$InstallInference,
    [switch]$InstallMaterials
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is required. Install it with: wsl --install -d Ubuntu"
}

if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$wslPath = (& wsl.exe -d $Distribution -- wslpath -a $RepoPath).Trim()
if (-not $wslPath) {
    throw "Could not translate repository path into WSL."
}

$install = @"
set -euo pipefail
cd '$wslPath'
python3 -m venv .venv-aeth-bridge
. .venv-aeth-bridge/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-aeth-bridge.txt pytest
"@

if ($InstallInference) {
    $install += @"
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
. ./setup.sh --basic --flash-attn --cumesh --o-voxel --flexgemm
"@
}

if ($InstallMaterials) {
    if (-not $InstallInference) {
        throw "-InstallMaterials requires -InstallInference."
    }
    $install += @"
. ./setup.sh --nvdiffrast --nvdiffrec
"@
}

& wsl.exe -d $Distribution -- bash -lc $install
if ($LASTEXITCODE -ne 0) {
    throw "Aeth bridge setup failed inside WSL."
}

Write-Host "Aeth bridge installed. Launch with:"
Write-Host "wsl.exe -d $Distribution -- bash -lc `"cd '$wslPath' && . .venv-aeth-bridge/bin/activate && python -m aeth_bridge`""
