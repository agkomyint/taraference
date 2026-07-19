# Interactive chat with Tara MoE packs on taraference — **100M-active product default**.
#
# Default run (no args):
#   - speed-qualified 1B-A100-v4 L12 pack first (~98M active), not thin speed750
#   - flash decode + CUDA graphs + vocab 8k + top_k=1 (max TPS for that SKU)
#   - ctx 1024, flash decode, CUDA graph, real top-1 routing
#
# Usage:
#   .\scripts\chat-moe-750.ps1
#   .\scripts\chat-moe-750.ps1 -Which moe1b         # force 1B-A100 shape (~100M active)
#   .\scripts\chat-moe-750.ps1 -Which moe1bv2       # 1B-A83 v2 smoke twin (83M active)
#   .\scripts\chat-moe-750.ps1 -Which moe1bv4       # 1.042B / 98M-active speed-qualified v4
#   .\scripts\chat-moe-750.ps1 -Which speed750      # thin pack, highest raw tok/s (~750)
#   .\scripts\chat-moe-750.ps1 -Pack "D:\path\to\q4pack"
#   .\scripts\chat-moe-750.ps1 -N 256 -Ctx 1024
#   .\scripts\chat-moe-750.ps1 -FullVocab           # slower full 32k head
#   .\scripts\chat-moe-750.ps1 -SpeedPack           # prefer speed750 over 100M-active
#
# In chat:
#   type a message, Enter
#   /reset   clear history (use when near ctx full — keeps tok/s high)
#   /quit    exit

[CmdletBinding()]
param(
    # auto = product default | moe1bv4 | moe1bv2 | moe1b | speed750 | moe500 | moe400
    [ValidateSet("auto", "moe1bv4", "moe1bv2", "moe1b", "speed750", "moe500", "moe400")]
    [string]$Which = "auto",

    [string]$Pack = "",
    [string]$TokenizerGguf = "",
    [string]$Tarafer = "",
    [int]$N = 128,
    # Product context; v4 L12 crossed 750 warm at short/mid context on the local GPU.
    [int]$Ctx = 1024,
    [switch]$FullVocab,
    [int]$VocabLimit = 8192,
    [switch]$NoCudaGraph,
    # flash >> fastv2 for MoE long/mid context on this stack
    [string]$Decode = "flash",
    # Prefer thin speed750 pack (~750 tps) instead of default 100M-active
    [switch]$SpeedPack,
    # Force top_k (default 1 = max TPS for active SKU; still real router)
    [int]$TopK = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "Cargo.toml"))) {
    $RepoRoot = (Get-Location).Path
}

$ExportRoot = "D:\Tara_HQ\departments\taraference_750_department\exports"
$script:PackSelectNote = ""

function Find-FirstExisting([string[]]$Paths) {
    foreach ($p in $Paths) {
        if ($p -and (Test-Path -LiteralPath $p)) {
            return (Resolve-Path -LiteralPath $p).Path
        }
    }
    return $null
}

function Get-PackStamp([string]$PackDir) {
    $meta = Join-Path $PackDir "meta.json"
    if (Test-Path -LiteralPath $meta) {
        return (Get-Item -LiteralPath $meta).LastWriteTimeUtc
    }
    return (Get-Item -LiteralPath $PackDir).LastWriteTimeUtc
}

function Select-NewestPack([string[]]$Candidates) {
    $best = $null
    $bestStamp = [datetime]::MinValue
    foreach ($p in $Candidates) {
        if (-not $p -or -not (Test-Path -LiteralPath $p)) { continue }
        $meta = Join-Path $p "meta.json"
        if (-not (Test-Path -LiteralPath $meta)) { continue }
        $stamp = Get-PackStamp $p
        if ($stamp -ge $bestStamp) {
            $bestStamp = $stamp
            $best = (Resolve-Path -LiteralPath $p).Path
        }
    }
    if ($best) {
        $script:PackSelectNote = "newest by meta.json mtime ($bestStamp UTC)"
    }
    return $best
}

function Get-KnownPackMap {
    return @{
        "moe1bv4" = @(
            (Join-Path $ExportRoot "tara-moe-1b-a100-v4-l12-laptop-mock-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-v4-l12-q4pack")
        )
        "moe1bv2" = @(
            (Join-Path $ExportRoot "tara-moe-1b-a83-v2-laptop-mock-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a83-v2-q4pack")
        )
        "moe1b" = @(
            (Join-Path $ExportRoot "tara-moe-1b-a100-laptop-mock-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-smoke-q4pack")
            (Join-Path $ExportRoot "tara-moe-1b-a100-real-q4pack")
        )
        "speed750" = @(
            (Join-Path $ExportRoot "tara-moe-speed750-q4pack")
            (Join-Path $ExportRoot "tara-moe-400-speed768-q4pack")
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
}

function Get-AllExportQ4Packs {
    $found = New-Object System.Collections.Generic.List[string]
    if (-not (Test-Path -LiteralPath $ExportRoot)) { return @() }

    $byName = Get-KnownPackMap
    foreach ($key in @("moe1bv4", "moe1bv2", "moe1b", "moe500", "speed750", "moe400")) {
        foreach ($p in $byName[$key]) {
            if ($p -and (Test-Path -LiteralPath $p) -and (Test-Path -LiteralPath (Join-Path $p "meta.json"))) {
                [void]$found.Add((Resolve-Path -LiteralPath $p).Path)
            }
        }
    }

    Get-ChildItem -LiteralPath $ExportRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'q4pack' } |
        ForEach-Object {
            $meta = Join-Path $_.FullName "meta.json"
            if (Test-Path -LiteralPath $meta) {
                $full = (Resolve-Path -LiteralPath $_.FullName).Path
                if (-not ($found -contains $full)) {
                    [void]$found.Add($full)
                }
            }
        }

    return $found.ToArray()
}

function Resolve-MoEPack([string]$WhichName, [string]$Explicit, [bool]$PreferSpeedPack) {
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit)) {
            throw "MoE pack not found: $Explicit"
        }
        $script:PackSelectNote = "explicit -Pack"
        return (Resolve-Path -LiteralPath $Explicit).Path
    }

    $byName = Get-KnownPackMap

    if ($WhichName -ne "auto") {
        $hit = Select-NewestPack $byName[$WhichName]
        if (-not $hit) {
            throw "No pack found for -Which $WhichName under $ExportRoot"
        }
        $script:PackSelectNote = "-Which $WhichName → $($script:PackSelectNote)"
        return $hit
    }

    if ($PreferSpeedPack) {
        # Optional: thin speed packs first (~750 raw tok/s).
        $speedFirst = @(
            $byName["speed750"] +
            $byName["moe1b"] +
            $byName["moe500"] +
            $byName["moe400"]
        )
        $hit = Find-FirstExisting $speedFirst
        if ($hit) {
            $script:PackSelectNote = "speed-pack auto (speed750 → 1b → 500 → 400)"
            return $hit
        }
    }

    # Default product: ~100M active (1B-A100 mock / moe500), then thinner packs.
    $active100mFirst = @(
        $byName["moe1bv4"] +
        $byName["moe1b"] +
        $byName["moe500"] +
        $byName["speed750"] +
        $byName["moe400"]
    )
    $hit = Find-FirstExisting $active100mFirst
    if ($hit) {
        $script:PackSelectNote = "100M-active default (moe1bv4 → moe1b → moe500 → speed750 → moe400)"
        return $hit
    }

    $all = Get-AllExportQ4Packs
    $hit = Select-NewestPack $all
    if (-not $hit) {
        throw "No MoE pack found. Expected *q4pack with meta.json under $ExportRoot"
    }
    $script:PackSelectNote = "auto newest → $($script:PackSelectNote)"
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

$Pack = Resolve-MoEPack -WhichName $Which -Explicit $Pack -PreferSpeedPack $SpeedPack.IsPresent

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
$tpsHint = "~500+ class (100M active) or ~750 class (speed pack)"
try {
    $meta = Get-Content (Join-Path $Pack "meta.json") -Raw | ConvertFrom-Json
    $metaNote = ("L={0} d={1} ff={2} experts={3} top_k={4} quant={5}" -f `
        $meta.n_layer, $meta.n_embd, $meta.expert_ff, $meta.n_experts, $meta.router_top_k, $meta.quant)
    if ($Pack -match "a100-v4-l12") {
        $qualityNote = "V4 L12 laptop smoke twin: speed-qualified architecture; train full E=17 for product quality."
        $tpsHint = "measured warm 790–792 tok/s (1.042B total / 98.3M active shape)"
    } elseif ($Pack -match "speed750|speed768") {
        $qualityNote = "Speed pack — highest raw tok/s (~750 warm on 3050 Ti). Quality still early."
        $tpsHint = "target warm ~750 tok/s (thin MoE)"
    } elseif ($Pack -match "a83-v2") {
        $qualityNote = "A83-v2 laptop smoke twin (same active shape as future E=27 1B). Smoke quality only."
        $tpsHint = "measured warm up to ~678 tok/s; target 750 after engine fusion"
    } elseif ($Pack -match "laptop-mock") {
        $qualityNote = "Laptop mock 1B-A100 shape (~100M active). Smoke quality; max-TPS flags on."
        $tpsHint = "target warm ~500–580 tok/s short/mid (cool GPU)"
    } elseif ($Pack -match "smoke") {
        $qualityNote = "SMOKE checkpoint — plumbing/speed, not product quality."
        $tpsHint = "~500 class if 100M-active shape"
    } elseif ($Pack -match "moe-1b|1b-a100") {
        $qualityNote = "Flagship 1B-A100 (~1B total / ~100M active)."
        $tpsHint = "target ~500–560 tok/s short/mid"
    } elseif ($Pack -match "moe-500") {
        $qualityNote = "MoE-500 (~100M active). Same active band as 1B."
        $tpsHint = "target ~500 class"
    }
} catch {
    $metaNote = "(meta unreadable)"
}

# ── Max-TPS environment (always on for this script unless overridden) ──
$env:TARAFER_TOKENIZER_GGUF = $TokenizerGguf
$env:TARAFER_N_CTX = "$Ctx"
# Real router, fewer experts = less BW (do NOT use TARAFER_MOE_FIXED for product claims)
Remove-Item Env:TARAFER_MOE_FIXED -ErrorAction SilentlyContinue
Remove-Item Env:TARAFER_STRICT_CTX -ErrorAction SilentlyContinue
# Interactive chat: honor EOS so replies stop naturally (speed benches use IGNORE_EOS)
Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue

if ($TopK -ge 1) {
    $env:TARAFER_MOE_TOPK = "$TopK"
} else {
    Remove-Item Env:TARAFER_MOE_TOPK -ErrorAction SilentlyContinue
}
# Same top_k=1 shortcut path used by speed ladder
$env:TARAFER_SPEED = "1"

if ($FullVocab) {
    $env:TARAFER_FULL_VOCAB = "1"
    Remove-Item Env:TARAFER_VOCAB_LIMIT -ErrorAction SilentlyContinue
    $vocabNote = "full vocab 32k (SLOWER)"
} else {
    Remove-Item Env:TARAFER_FULL_VOCAB -ErrorAction SilentlyContinue
    $env:TARAFER_VOCAB_LIMIT = "$VocabLimit"
    $vocabNote = "active vocab $VocabLimit (max TPS)"
}

$approxTurns = [math]::Max(1, [int][math]::Floor($Ctx / [math]::Max(1, ($N + 40))))
$packLeaf = Split-Path $Pack -Leaf
$graphNote = if ($NoCudaGraph) { "OFF" } else { "ON (max TPS)" }

Write-Host ""
Write-Host "=== Tara MoE chat (max tok/s) ===" -ForegroundColor Cyan
Write-Host "which     : $Which"
Write-Host "selected  : $packLeaf  ($script:PackSelectNote)" -ForegroundColor Green
Write-Host "model     : $Pack"
Write-Host "arch      : $metaNote"
Write-Host "tokenizer : $TokenizerGguf"
Write-Host "binary    : $Tarafer"
Write-Host "max_new   : $N   ctx: $Ctx   decode: $Decode   cuda_graph: $graphNote"
Write-Host "vocab     : $vocabNote"
Write-Host "router    : top_k=$TopK  TARAFER_SPEED=1  (real routing, not fixed experts)"
Write-Host "expect    : $tpsHint" -ForegroundColor Green
Write-Host ("budget    : ~{0} turns if each reply uses ~{1} tokens (rough)" -f $approxTurns, $N) -ForegroundColor DarkGray
Write-Host ""
Write-Host "Commands: /reset  clear history   |   /quit  exit" -ForegroundColor DarkGray
Write-Host "Tip: /reset often → shorter KV → higher tok/s" -ForegroundColor DarkGray
Write-Host "Tip: default = ~100M active | -SpeedPack or -Which speed750 = thin ~750 pack" -ForegroundColor DarkGray
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
