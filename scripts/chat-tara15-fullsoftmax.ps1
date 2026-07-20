param(
    [string]$Pack = "D:\Tara_HQ\departments\taraference_750_department\exports\local-moe-100m-fullsoftmax-q4pack",
    [string]$TokenizerGguf = "D:\taraference\models\tara-sprint-80m-Q8_0.gguf",
    [int]$Ctx = 1024,
    [int]$MaxNew = 256,
    [string]$Prompt = "",
    [switch]$NoCudaGraph,
    [switch]$UseShortlist
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo "target\release\tarafer.exe"
$metaPath = Join-Path $Pack "meta.json"

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Missing release binary: $exe`nBuild it with: cd D:\taraference; cargo build --release -p taraference"
}
if (-not (Test-Path -LiteralPath $Pack -PathType Container)) {
    throw "Missing Tara MoE pack: $Pack"
}
if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
    throw "Not a Tara MoE pack (missing meta.json): $Pack"
}
if (-not (Test-Path -LiteralPath $TokenizerGguf -PathType Leaf)) {
    throw "Missing tokenizer GGUF: $TokenizerGguf"
}

$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$routerMode = if ($meta.router_weight_mode) { $meta.router_weight_mode } else { "selected_softmax (legacy default)" }

Write-Host "Taraference Tara 1.5 MoE chat" -ForegroundColor Cyan
Write-Host "  Pack:           $Pack"
Write-Host "  Quant:          $($meta.quant ?? 'q8_0')"
Write-Host "  Architecture:   L$($meta.n_layer), E$($meta.n_experts), top-$($meta.router_top_k), d=$($meta.n_embd)"
Write-Host "  Router weights: $routerMode"
Write-Host "  Context:        $Ctx"
Write-Host "  CUDA graphs:    $(-not $NoCudaGraph)"
Write-Host ""

if ($routerMode -ne "full_softmax") {
    Write-Warning "This pack is not marked full_softmax. Use a corrected Tara 1.5 export for task-trained top-1 routing."
}

$env:TARAFER_TOKENIZER_GGUF = $TokenizerGguf
$env:TARAFER_N_CTX = $Ctx.ToString()
$env:TARAFER_STRICT_CTX = "1"
if ($UseShortlist) {
    Remove-Item Env:TARAFER_FULL_VOCAB -ErrorAction SilentlyContinue
} else {
    $env:TARAFER_FULL_VOCAB = "1"
}

$cliArgs = @($Pack, "--ctx", $Ctx, "--max-new", $MaxNew)
if ($NoCudaGraph) {
    $cliArgs += "--no-cuda-graph"
}
if ($Prompt) {
    $cliArgs += @("--prompt", $Prompt)
} else {
    Write-Host "Interactive mode started. Type /exit or press Ctrl+C to stop." -ForegroundColor Green
    Write-Host ""
}

Push-Location $repo
try {
    & $exe @cliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "tarafer exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
