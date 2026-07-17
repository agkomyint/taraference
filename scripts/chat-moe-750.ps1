# Interactive chat with Tara MoE packs on taraference.
#
# Defaults to the newest available pack under taraference_750_department/exports.
# Prefer speed750 for ~750 tok/s; use moe500 for the latest capacity checkpoint.
#
# Usage:
#   .\scripts\chat-moe-750.ps1
#   .\scripts\chat-moe-750.ps1 -Which moe500
#   .\scripts\chat-moe-750.ps1 -Which speed750
#   .\scripts\chat-moe-750.ps1 -Pack "D:\path\to\q4pack"
#   .\scripts\chat-moe-750.ps1 -N 128 -Ctx 4096
#   .\scripts\chat-moe-750.ps1 -N 64 -Ctx 8192 -FullVocab
#
# In chat:
#   type a message, Enter
#   /reset   clear history (use when near ctx full)
#   /quit    exit

[CmdletBinding()]
param(
    # auto | moe1b | speed750 | moe500 | moe400
    [ValidateSet("auto", "moe1b", "speed750", "moe500", "moe400")]
    [string]$Which = "auto",

    [string]$Pack = "",
    [string]$TokenizerGguf = "",
    [string]$Tarafer = "",
    [int]$N = 128,
    [int]$Ctx = 4096,
    [switch]$FullVocab,
    [int]$VocabLimit = 8192,
    [switch]$NoCudaGraph,
    [string]$Decode = "flash"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "Cargo.toml"))) {
    $RepoRoot = (Get-Location).Path
}

$ExportRoot = "D:\Tara_HQ\departments\taraference_750_department\exports"

function Find-FirstExisting([string[]]$Paths) {
    foreach ($p in $Paths) {
        if ($p -and (Test-Path -LiteralPath $p)) {
            return (Resolve-Path -LiteralPath $p).Path
        }
    }
    return $null
}

function Resolve-MoEPack([string]$WhichName, [string]$Explicit) {
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit)) {
            throw "MoE pack not found: $Explicit"
        }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }

    $byName = @{
        "moe1b" = @(
            (Join-Path $ExportRoot "tara-moe-1b-a100-laptop-mock-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-smoke-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-real-q4pack")
        )
        "speed750" = @(
            (Join-Path $ExportRoot "tara-moe-speed750-q4pack")
        )
        "moe500" = @(
            (Join-Path $ExportRoot "tara-moe-500-a100-smoke30-q4pack")
            (Join-Path $ExportRoot "tara-moe-500-a100-real-q4pack")
            (Join-Path $ExportRoot "tara-moe-500-a100-q4pack")
        )
        "moe400" = @(
            (Join-Path $ExportRoot "tara-moe-400-a120-q4pack")
            (Join-Path $ExportRoot "tara-moe-400-speed768-q4pack")
        )
    }

    if ($WhichName -ne "auto") {
        $hit = Find-FirstExisting $byName[$WhichName]
        if (-not $hit) {
            throw "No pack found for -Which $WhichName under $ExportRoot"
        }
        return $hit
    }

    # auto: flagship 1B if present, else newest 100M-active, then speed750
    $autoOrder = @(
        $byName["moe1b"] +
        $byName["moe500"] +
        $byName["speed750"] +
        $byName["moe400"]
    )
    $hit = Find-FirstExisting $autoOrder
    if (-not $hit) {
        throw "No MoE pack found. Expected under $ExportRoot (moe500 / speed750 / moe400)."
    }
    return $hit
}

if (-not $Tarafer) {
    $Tarafer = Find-FirstExisting @(
        (Join-Path $RepoRoot "target\release\tarafer.exe"),
        (Join-Path $RepoRoot "target-candidate\release\tarafer.exe")
    )
}
if (-not $Tarafer) {
    Write-Host "Building tarafer (release)..." -ForegroundColor Yellow
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
        (Join-Path $ExportRoot "tara-sprint-80m-real\tara-sprint-80m-Q8_0.gguf")
    )
}

$Pack = Resolve-MoEPack -WhichName $Which -Explicit $Pack

if (-not (Test-Path -LiteralPath (Join-Path $Pack "meta.json"))) {
    throw "Not a Tara MoE pack (missing meta.json): $Pack"
}
if (-not $TokenizerGguf -or -not (Test-Path -LiteralPath $TokenizerGguf)) {
    throw "Tokenizer GGUF not found. Pass -TokenizerGguf path\to\tara-sprint-*-Q8_0.gguf"
}
if (-not (Test-Path -LiteralPath $Tarafer)) {
    throw "tarafer.exe not found: $Tarafer"
}

# Human-readable arch blurb from meta.json
$metaNote = ""
$qualityNote = "Model may be under-trained — quality can be weak."
try {
    $meta = Get-Content (Join-Path $Pack "meta.json") -Raw | ConvertFrom-Json
    $metaNote = ("L={0} d={1} ff={2} experts={3} top_k={4} quant={5}" -f `
        $meta.n_layer, $meta.n_embd, $meta.expert_ff, $meta.n_experts, $meta.router_top_k, $meta.quant)
    if ($Pack -match "smoke") {
        $qualityNote = "SMOKE checkpoint (e.g. ~30 train steps) — quality is not product-ready; good for engine/chat plumbing."
    } elseif ($Pack -match "speed750") {
        $qualityNote = "Speed-750 arch (~750 tok/s target on 3050 Ti). Quality still early."
    } elseif ($Pack -match "moe-1b|1b-a100") {
        $qualityNote = "Flagship 1B-A100 (~1B total / ~100M active). 750 target; engine ~500 tok/s baseline today."
    } elseif ($Pack -match "moe-500") {
        $qualityNote = "MoE-500 (~0.47B total / ~100M active). Same active band as 1B-A100; engine ~500 tok/s class today."
    }
} catch {
    $metaNote = "(meta unreadable)"
}

$env:TARAFER_TOKENIZER_GGUF = $TokenizerGguf
$env:TARAFER_N_CTX = "$Ctx"
Remove-Item Env:TARAFER_MOE_FIXED -ErrorAction SilentlyContinue
Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue
Remove-Item Env:TARAFER_STRICT_CTX -ErrorAction SilentlyContinue

if ($FullVocab) {
    $env:TARAFER_FULL_VOCAB = "1"
    Remove-Item Env:TARAFER_VOCAB_LIMIT -ErrorAction SilentlyContinue
    $vocabNote = "full vocab 32k"
} else {
    Remove-Item Env:TARAFER_FULL_VOCAB -ErrorAction SilentlyContinue
    $env:TARAFER_VOCAB_LIMIT = "$VocabLimit"
    $vocabNote = "active vocab $VocabLimit (faster)"
}

$approxTurns = [math]::Max(1, [int][math]::Floor($Ctx / [math]::Max(1, ($N + 40))))
$packLeaf = Split-Path $Pack -Leaf

Write-Host ""
Write-Host "=== Tara MoE interactive chat ===" -ForegroundColor Cyan
Write-Host "which     : $Which  →  $packLeaf"
Write-Host "model     : $Pack"
Write-Host "arch      : $metaNote"
Write-Host "tokenizer : $TokenizerGguf"
Write-Host "binary    : $Tarafer"
Write-Host "max_new   : $N   ctx: $Ctx   decode: $Decode"
Write-Host "vocab     : $vocabNote"
Write-Host ("budget    : ~{0} turns if each reply uses ~{1} tokens (rough)" -f $approxTurns, $N) -ForegroundColor DarkGray
Write-Host ""
Write-Host "Commands: /reset  clear history   |   /quit  exit" -ForegroundColor DarkGray
Write-Host "Tip: if context full, type /reset or use -N 64 -Ctx 8192" -ForegroundColor DarkGray
Write-Host "Tip: -Which moe1b (flagship) | moe500 (same active) | speed750 (proven 750, smaller)" -ForegroundColor DarkGray
Write-Host "Note: $qualityNote" -ForegroundColor DarkYellow
Write-Host "First reply may be slower (CUDA graph warm-up)." -ForegroundColor DarkGray
Write-Host ""

$launchArgs = @(
    $Pack,
    "-n", "$N",
    "--ctx", "$Ctx",
    "--decode", $Decode
)
if ($NoCudaGraph) {
    $launchArgs += "--no-cuda-graph"
}

& $Tarafer @launchArgs
exit $LASTEXITCODE
