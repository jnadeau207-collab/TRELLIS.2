[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu",
    [string]$RepoPath = "",
    [string]$ArtifactRoot = "~/.cache/aeth-bridge"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$wslPath = (& wsl.exe -d $Distribution -- wslpath -a $RepoPath).Trim()
$command = "cd '$wslPath' && . .venv-aeth-bridge/bin/activate && exec python -m aeth_bridge --artifact-root '$ArtifactRoot'"
& wsl.exe -d $Distribution -- bash -lc $command
exit $LASTEXITCODE
