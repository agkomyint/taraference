# Test Tara MoE speed pack for ~750 single-stream decode tok/s on taraference.
#
# Usage:
#   .\scripts\test-moe-750.ps1
#   .\scripts\test-moe-750.ps1 -Runs 5 -N 256
#   .\scripts\test-moe-750.ps1 -FullVocab          # full 32k head (slower)
#   .\scripts\test-moe-750.ps1 -Pack "D:\path\to\pack"
#
# Expectation (RTX 3050 Ti Laptop, warm CUDA graph):
#   cold first run  ~600 t/s
#   warm runs       ~760–770 t/s  (target ≥ 750)

[CmdletBinding()]
param(
    [string]$Pack = "D:\Tara_HQ\departments\taraference_750_department\exports\tara-moe-400-speed768-q4pack",
    [string]$TokenizerGguf = "",
    [string]$Tarafer = "",
    [string]$Prompt = "hello world speed test please continue writing more tokens for benchmark now",
    [int]$N = 256,
    [int]$Ctx = 384,
    [int]$Runs = 4,
    [switch]$FullVocab,
    [int]$VocabLimit = 8192,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "Cargo.toml"))) {
    $RepoRoot = (Get-Location).Path
}

function Find-FirstExisting([string[]]$Paths) {
    foreach ($p in $Paths) {
        if ($p -and (Test-Path -LiteralPath $p)) { return (Resolve-Path -LiteralPath $p).Path }
    }
    return $null
}

if (-not $Tarafer) {
    $Tarafer = Find-FirstExisting @(
        (Join-Path $RepoRoot "target\release\tarafer.exe"),
        (Join-Path $RepoRoot "target-candidate\release\tarafer.exe")
    )
}
if (-not $Tarafer) {
    Write-Host "tarafer.exe not found. Building release..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    try {
        cargo build --release -p taraference
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    } finally {
        Pop-Location
    }
    $Tarafer = Join-Path $RepoRoot "target\release\tarafer.exe"
}

if (-not $TokenizerGguf) {
    $TokenizerGguf = Find-FirstExisting @(
        (Join-Path $RepoRoot "models\tara-sprint-80m-Q8_0.gguf"),
        (Join-Path $RepoRoot "models\tara-sprint-50m-Q8_0.gguf"),
        (Join-Path $RepoRoot "models\tara-sprint-150m-Q8_0.gguf"),
        "D:\Tara_HQ\departments\taraference_750_department\exports\tara-sprint-80m-real\tara-sprint-80m-Q8_0.gguf"
    )
}

if (-not (Test-Path -LiteralPath $Pack)) {
    throw "MoE pack not found: $Pack`nExport speed pack first (tara-moe-400-speed768-q4pack)."
}
if (-not (Test-Path -LiteralPath (Join-Path $Pack "meta.json"))) {
    throw "Not a MoE pack (missing meta.json): $Pack"
}
if (-not $TokenizerGguf -or -not (Test-Path -LiteralPath $TokenizerGguf)) {
    throw "Tokenizer GGUF not found. Pass -TokenizerGguf or place tara-sprint-*-Q8_0.gguf under models\"
}
if (-not (Test-Path -LiteralPath $Tarafer)) {
    throw "tarafer binary missing: $Tarafer"
}

# Env for MoE serve
$env:TARAFER_TOKENIZER_GGUF = $TokenizerGguf
Remove-Item Env:TARAFER_MOE_FIXED -ErrorAction SilentlyContinue
Remove-Item Env:TARAFER_MOE_TOPK -ErrorAction SilentlyContinue
Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue

if ($FullVocab) {
    $env:TARAFER_FULL_VOCAB = "1"
    Remove-Item Env:TARAFER_VOCAB_LIMIT -ErrorAction SilentlyContinue
    Remove-Item Env:TARAFER_SPEED -ErrorAction SilentlyContinue
    $vocabNote = "full vocab (slower)"
} else {
    Remove-Item Env:TARAFER_FULL_VOCAB -ErrorAction SilentlyContinue
    # Speed packs auto shortlist; force limit explicitly for clarity.
    $env:TARAFER_VOCAB_LIMIT = "$VocabLimit"
    $vocabNote = "active_vocab=$VocabLimit"
}

Write-Host ""
Write-Host "=== MoE 750 speed test ===" -ForegroundColor Cyan
Write-Host "tarafer   : $Tarafer"
Write-Host "pack      : $Pack"
Write-Host "tokenizer : $TokenizerGguf"
Write-Host "prompt    : $Prompt"
Write-Host "n=$N  ctx=$Ctx  runs=$Runs  $vocabNote"
Write-Host "target    : warm decode >= 750 tok/s"
Write-Host ""

$rates = New-Object System.Collections.Generic.List[double]

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host "--- run $i/$Runs ---" -ForegroundColor DarkCyan
    $out = & $Tarafer $Pack --prompt $Prompt -n $N --ctx $Ctx 2>&1
    $text = ($out | Out-String)

    $line = ($out | Where-Object { $_ -match 'decode\s+([\d.]+)\s+tok/s' } | Select-Object -Last 1)
    if (-not $line) {
        Write-Host $text
        throw "No decode tok/s line in run $i (see output above)"
    }

    if ($line -match 'decode\s+([\d.]+)\s+tok/s') {
        $tps = [double]$Matches[1]
        $rates.Add($tps) | Out-Null
        $color = if ($tps -ge 750) { "Green" } elseif ($tps -ge 600) { "Yellow" } else { "Red" }
        Write-Host ("  decode {0:F1} tok/s" -f $tps) -ForegroundColor $color
    }

    # Show approx / flags lines if present
    $out | Where-Object {
        $_ -match 'approximation|active_vocab|cuda_graph|GPU device|weight_gib|experts='
    } | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
}

$cold = $rates[0]
$warm = @()
if ($rates.Count -gt 1) {
    $warm = $rates.GetRange(1, $rates.Count - 1)
} else {
    $warm = $rates
}
$warmBest = ($warm | Measure-Object -Maximum).Maximum
$warmAvg = ($warm | Measure-Object -Average).Average

Write-Host ""
Write-Host "=== summary ===" -ForegroundColor Cyan
Write-Host ("cold (run1) : {0:F1} tok/s" -f $cold)
Write-Host ("warm best   : {0:F1} tok/s" -f $warmBest)
Write-Host ("warm avg    : {0:F1} tok/s" -f $warmAvg)
Write-Host ("all runs    : {0}" -f (($rates | ForEach-Object { "{0:F1}" -f $_ }) -join ", "))

if ($warmBest -ge 750) {
    Write-Host ""
    Write-Host "PASS: warm best >= 750 tok/s" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "FAIL: warm best {0:F1} < 750 tok/s" -f $warmBest -ForegroundColor Red
    Write-Host "Tips: close other GPU apps; re-run (first run pays CUDA-graph capture); check 3050 Ti not thermal-throttled."
    exit 1
}
