# Interactive chat with Qwen3.5-4B using the q35fast decode variant (Q5→Q4 hot path).
#
# Usage:
#   .\scripts\chat-qwen35-4b-fast.ps1
#   .\scripts\chat-qwen35-4b-fast.ps1 -N 256 -Ctx 1024
#   .\scripts\chat-qwen35-4b-fast.ps1 -Baseline   # default quality path (no Q35_FAST)
#
# In chat:
#   type a message, Enter
#   /reset   clear history
#   /quit    exit

[CmdletBinding()]
param(
    [int]$N = 512,
    [int]$Ctx = 1024,
    [string]$Model = "",
    [string]$Tarafer = "",
    [string]$Decode = "flash",
    # Use default Q5→Q8 path instead of q35fast
    [switch]$Baseline,
    [switch]$NoCudaGraph,
    [switch]$Think
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root "Cargo.toml"))) {
    $root = (Get-Location).Path
}

if (-not $Model) {
    $Model = Join-Path $root "models\Qwen3.5-4B-Q4_K_M.gguf"
}
if (-not $Tarafer) {
    $candidates = @(
        (Join-Path $root "target\release\tarafer.exe"),
        (Join-Path $root "target\debug\tarafer.exe")
    )
    $Tarafer = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $Tarafer -or -not (Test-Path -LiteralPath $Tarafer)) {
    throw "tarafer.exe not found. Build with: cargo build --release -p taraference"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model`nDownload with: .\target\release\tarafer.exe --download 4b"
}

$argsList = @(
    $Model,
    "-n", "$N",
    "--ctx", "$Ctx",
    "--decode", $Decode
)
if ($NoCudaGraph) { $argsList += "--no-cuda-graph" }
if ($Think) { $argsList += "--think" }

if ($Baseline) {
    if (Test-Path Env:TARAFER_Q35_FAST) { Remove-Item Env:TARAFER_Q35_FAST -ErrorAction SilentlyContinue }
    if (Test-Path Env:TARAFER_VARIANT) { Remove-Item Env:TARAFER_VARIANT -ErrorAction SilentlyContinue }
    $label = "baseline (Q5→Q8)"
} else {
    $env:TARAFER_Q35_FAST = "1"
    $label = "q35fast (Q5→Q4)"
}

Write-Host "Qwen3.5-4B chat — $label" -ForegroundColor Cyan
Write-Host "  binary  $Tarafer"
Write-Host "  model   $Model"
Write-Host "  decode  $Decode  max_new=$N  ctx=$Ctx"
Write-Host "Commands: /reset clears context | /quit exits" -ForegroundColor DarkGray
Write-Host ""

Push-Location $root
try {
    & $Tarafer @argsList
    $code = $LASTEXITCODE
} finally {
    Pop-Location
    if (Test-Path Env:TARAFER_Q35_FAST) {
        Remove-Item Env:TARAFER_Q35_FAST -ErrorAction SilentlyContinue
    }
}

exit $code
