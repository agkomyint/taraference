param(
    [string]$Pack = "D:\Tara_HQ\departments\taraference_750_department\exports\local-moe-1m-memorize-q4pack",
    [string]$TokenizerGguf = "D:\taraference\models\tara-sprint-80m-Q8_0.gguf",
    [string]$Prompt = "Question: What is Tara? Answer:",
    [int]$MaxNew = 32,
    [int]$Ctx = 64,
    [switch]$Interactive,
    [switch]$NoCudaGraph
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo "target\release\tarafer.exe"
$metaPath = Join-Path $Pack "meta.json"

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Missing Taraference release binary: $exe"
}
if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
    throw "Missing Q4 MoE pack or meta.json: $Pack"
}
if (-not (Test-Path -LiteralPath $TokenizerGguf -PathType Leaf)) {
    throw "Missing tokenizer GGUF: $TokenizerGguf"
}

$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
Write-Host "Tara 1M Q4 GPU test" -ForegroundColor Cyan
Write-Host "  Runtime: $exe"
Write-Host "  Pack: $Pack"
Write-Host "  Model: L$($meta.n_layer), E$($meta.n_experts), top-$($meta.router_top_k), d=$($meta.n_embd)"
Write-Host "  Router: $($meta.router_weight_mode)"
Write-Host ""

$env:TARAFER_TOKENIZER_GGUF = $TokenizerGguf
$env:TARAFER_FULL_VOCAB = "1"
$env:TARAFER_N_CTX = $Ctx.ToString()
$env:TARAFER_STRICT_CTX = "1"

$cliArgs = @($Pack, "--ctx", $Ctx, "--max-new", $MaxNew)
if ($NoCudaGraph) {
    $cliArgs += "--no-cuda-graph"
}
if (-not $Interactive) {
    $cliArgs += @("--prompt", $Prompt)
} else {
    Write-Host "Interactive mode. Try: Question: What is Tara? Answer:" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop.`n"
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
