# Fair single-stream multi-turn: Ollama then taraference q35fast.
# Order matters: Ollama first, fully unload, confirm VRAM free, then tarafer.
#
# Usage:
#   .\scripts\bench-qwen35-4b-vs-ollama.ps1

[CmdletBinding()]
param(
    [string]$Gguf = "",
    [string]$OllamaName = "tara-qwen35-4b-q4km",
    [string]$Tarafer = "",
    [int]$NumCtx = 1024
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root "Cargo.toml"))) { $root = (Get-Location).Path }

if (-not $Gguf) { $Gguf = Join-Path $root "models\Qwen3.5-4B-Q4_K_M.gguf" }
if (-not $Tarafer) {
    $Tarafer = @(
        (Join-Path $root "target\release\tarafer.exe"),
        (Join-Path $root "target\debug\tarafer.exe")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not (Test-Path $Gguf)) { throw "Missing GGUF: $Gguf" }
if (-not $Tarafer) { throw "Missing tarafer.exe — cargo build --release -p taraference" }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "ollama not on PATH" }

$modelfile = Join-Path $root "benchmarks\ollama-qwen35-4b.Modelfile"
$benchPy = Join-Path $root "benchmarks\ollama_chat_bench.py"

function Get-GpuMemUsedMiB {
    $line = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if (-not $line) { return -1 }
    return [int]([double]$line.Trim())
}

function Stop-OllamaModels {
    Write-Host "Unloading Ollama models..." -ForegroundColor Yellow
    # keep_alive 0 unloads; also stop by name
    try {
        $ps = ollama ps 2>$null | Out-String
        if ($ps -match $OllamaName) {
            ollama stop $OllamaName 2>$null | Out-Null
        }
    } catch {}
    # Force unload via generate keep_alive 0 if still listed
    try {
        $body = @{ model = $OllamaName; keep_alive = 0; prompt = "" } | ConvertTo-Json
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30 | Out-Null
    } catch {}
    Start-Sleep -Seconds 2
    ollama stop $OllamaName 2>$null | Out-Null
    Start-Sleep -Seconds 1
}

function Assert-GpuMostlyFree([string]$phase) {
    $used = Get-GpuMemUsedMiB
    Write-Host "GPU VRAM used after $phase : $used MiB" -ForegroundColor DarkGray
    ollama ps
    if ($used -gt 400) {
        Write-Host "WARN: VRAM still >400 MiB — another process may be holding the GPU" -ForegroundColor Red
    }
}

Push-Location $root
try {
    Write-Host "════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Fair bench: Ollama then taraference q35fast" -ForegroundColor Cyan
    Write-Host "  Same GGUF: $Gguf" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════" -ForegroundColor Cyan

    # Ensure clean start
    Stop-OllamaModels
    Assert-GpuMostlyFree "initial cleanup"

    # Create/update Ollama model from exact GGUF
    Write-Host "`n[1/4] ollama create $OllamaName from local GGUF..." -ForegroundColor Cyan
    ollama create $OllamaName -f $modelfile
    if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }

    Write-Host "`n[2/4] Ollama multi-turn bench (num_ctx=$NumCtx, num_predict=128, temp=0)..." -ForegroundColor Cyan
    $ollamaOut = & python $benchPy $OllamaName --label "Qwen3.5-4B-Q4_K_M.gguf" --num-ctx $NumCtx --num-predict 128 2>&1 | Out-String
    Write-Host $ollamaOut
    $ollamaSummary = ($ollamaOut -split "`n" | Where-Object { $_ -match "^SUMMARY " } | Select-Object -Last 1)
    if ($ollamaSummary) {
        Write-Host "OLLAMA $ollamaSummary" -ForegroundColor Green
    }

    Write-Host "`n[3/4] Unload Ollama completely (required for fair tarafer run)..." -ForegroundColor Cyan
    Stop-OllamaModels
    Assert-GpuMostlyFree "ollama unload"
    $used = Get-GpuMemUsedMiB
    if ($used -gt 500) {
        throw "GPU still using $used MiB after ollama unload — fix residency before tarafer (unfair otherwise)"
    }

    Write-Host "`n[4/4] taraference q35fast --profile..." -ForegroundColor Cyan
    $env:TARAFER_Q35_FAST = "1"
    if (Test-Path Env:TARAFER_Q35_SPEC) { Remove-Item Env:TARAFER_Q35_SPEC -ErrorAction SilentlyContinue }
    if (Test-Path Env:TARAFER_PLD) { Remove-Item Env:TARAFER_PLD -ErrorAction SilentlyContinue }
    $taraOut = & $Tarafer $Gguf --profile --ctx $NumCtx 2>&1 | Out-String
    # print key lines
    $taraOut -split "`n" | Select-String -Pattern "variant|overall_decode|overall_prefill|decode_tps_first|decode_tps_last|PROFILE REPORT|SUMMARY_KV|saved_at" | ForEach-Object { $_.Line }
    if (Test-Path Env:TARAFER_Q35_FAST) { Remove-Item Env:TARAFER_Q35_FAST -ErrorAction SilentlyContinue }

    Write-Host "`n════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  DONE — compare overall_decode_tps" -ForegroundColor Cyan
    if ($ollamaSummary) { Write-Host "  Ollama: $ollamaSummary" }
    Write-Host "  Tarafer: see overall_decode_tps= in profile above" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════" -ForegroundColor Cyan
} finally {
    Pop-Location
    Stop-OllamaModels | Out-Null
    if (Test-Path Env:TARAFER_Q35_FAST) { Remove-Item Env:TARAFER_Q35_FAST -EA SilentlyContinue }
}
