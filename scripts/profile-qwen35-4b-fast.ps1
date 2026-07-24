# Profile Qwen3.5-4B with the q35fast decode variant (Q5→Q4 hot path).
# Default path is unchanged — this only sets TARAFER_Q35_FAST for this process.
#
# Usage (from repo root or anywhere):
#   .\scripts\profile-qwen35-4b-fast.ps1
#   .\scripts\profile-qwen35-4b-fast.ps1 -Short          # quick -n 64 smoke
#   .\scripts\profile-qwen35-4b-fast.ps1 -WithBaseline   # baseline then fast
#   .\scripts\profile-qwen35-4b-fast.ps1 -Prompt "hi" -N 128

[CmdletBinding()]
param(
    # Full multi-turn --profile (default). -Short uses a fixed prompt instead.
    [switch]$Short,
    # Also run default (no variant) first for side-by-side numbers.
    [switch]$WithBaseline,
    [int]$N = 64,
    [string]$Prompt = "Write a short hello.",
    [string]$Model = "",
    [string]$Tarafer = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
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

function Invoke-TaraferRun {
    param(
        [string]$Label,
        [switch]$Fast
    )
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  binary  $Tarafer"
    Write-Host "  model   $Model"
    if ($Fast) {
        Write-Host "  variant q35fast (TARAFER_Q35_FAST=1)" -ForegroundColor Yellow
        $env:TARAFER_Q35_FAST = "1"
    } else {
        Write-Host "  variant default (Q5→Q8)"
        if (Test-Path Env:TARAFER_Q35_FAST) { Remove-Item Env:TARAFER_Q35_FAST }
        if (Test-Path Env:TARAFER_VARIANT) { Remove-Item Env:TARAFER_VARIANT }
    }

    Push-Location $root
    try {
        if ($Short) {
            & $Tarafer $Model -n $N --prompt $Prompt
        } else {
            & $Tarafer $Model --profile
        }
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        if (Test-Path Env:TARAFER_Q35_FAST) { Remove-Item Env:TARAFER_Q35_FAST -ErrorAction SilentlyContinue }
        if (Test-Path Env:TARAFER_VARIANT) { Remove-Item Env:TARAFER_VARIANT -ErrorAction SilentlyContinue }
    }
    if ($code -ne 0) {
        throw "tarafer exited with code $code ($Label)"
    }
}

if ($WithBaseline) {
    Invoke-TaraferRun -Label "BASELINE (default)" 
}

Invoke-TaraferRun -Label "Q35FAST" -Fast

Write-Host ""
Write-Host "Done. Profile logs (if --profile): profile-logs\" -ForegroundColor DarkGray
Write-Host "  Look for overall_decode_tps= / decode XX.X tok/s" -ForegroundColor DarkGray
