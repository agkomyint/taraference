# One-click interactive chat launcher for the speed-qualified Tara 1.5 Base v4
# laptop smoke checkpoint. This is a chat launcher, not a fixed-token benchmark.

[CmdletBinding()]
param(
    [int]$N = 256,
    # 4096 permits a useful multi-turn conversation. Larger contexts retain more history but
    # decode more slowly and consume more KV memory.
    [int]$Ctx = 4096
)

$ErrorActionPreference = "Stop"
$chatLauncher = Join-Path $PSScriptRoot "chat-moe-750.ps1"

if (-not (Test-Path -LiteralPath $chatLauncher)) {
    throw "Chat launcher not found: $chatLauncher"
}

Write-Host "Starting Tara 1.5 Base v4 interactive chat..." -ForegroundColor Cyan
Write-Host "Commands: /reset clears context | /quit exits" -ForegroundColor DarkGray
Write-Host "Multi-turn memory: ON (history retained until /reset or context fills)." -ForegroundColor DarkGray
Write-Host "Note: chat honors EOS; short replies are not valid 750 tok/s benchmarks." -ForegroundColor DarkGray
Write-Host ""

& $chatLauncher -Which moe1bv4 -N $N -Ctx $Ctx -Decode flash -VocabLimit 8192 -TopK 1
exit $LASTEXITCODE
