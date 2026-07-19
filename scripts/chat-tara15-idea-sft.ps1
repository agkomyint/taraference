[CmdletBinding()]
param(
    [string]$Prompt = "",
    [int]$N = 128,
    [int]$Ctx = 1024,
    [switch]$Benchmark,
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$tarafer = Join-Path $repo "target\release\tarafer.exe"
$pack = Join-Path $repo "models\tara15-idea-sft-checkpoint61035-q8pack"
$tokenizer = Join-Path $pack "tokenizer.gguf"
$system = "You are Tara, a helpful creative assistant. Give exactly one useful, original idea that uses the user's meaningful words. Be concise and include a concrete first step."

foreach ($path in @($tarafer, $pack, $tokenizer)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}

$names = @(
    "TARAFER_TOKENIZER_GGUF", "TARAFER_FULL_VOCAB", "TARAFER_VOCAB_LIMIT",
    "TARAFER_MOE_WARPS", "TARAFER_N_CTX", "TARAFER_TARA141_SFT",
    "TARAFER_IGNORE_EOS"
)
$saved = @{}
foreach ($name in $names) {
    $item = Get-Item "Env:$name" -ErrorAction SilentlyContinue
    $saved[$name] = if ($null -ne $item) { $item.Value } else { $null }
}

try {
    $env:TARAFER_TOKENIZER_GGUF = $tokenizer
    $env:TARAFER_FULL_VOCAB = "1"
    $env:TARAFER_MOE_WARPS = "4"
    $env:TARAFER_N_CTX = "$Ctx"
    $env:TARAFER_TARA141_SFT = "1"
    Remove-Item Env:TARAFER_VOCAB_LIMIT -ErrorAction SilentlyContinue

    if ($Benchmark) {
        $env:TARAFER_IGNORE_EOS = "1"
        $benchmarkPrompt = if ($Prompt) { $Prompt } else { "beginner balcony gardening weekend" }
        $rendered = "<|system|>$system<|/system|>`n<|user|>$benchmarkPrompt<|/user|>`n<|assistant|>"
        for ($i = 1; $i -le $Runs; $i++) {
            Write-Host "=== benchmark $i/${Runs}: fixed $N tokens ===" -ForegroundColor Cyan
            & $tarafer $pack --prompt $rendered -n $N --ctx $Ctx --decode flash
            if ($LASTEXITCODE -ne 0) { throw "tarafer benchmark $i failed" }
        }
    } elseif ($Prompt) {
        Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue
        $rendered = "<|system|>$system<|/system|>`n<|user|>$Prompt<|/user|>`n<|assistant|>"
        & $tarafer $pack --prompt $rendered -n $N --ctx $Ctx --decode flash `
            --temperature 0.7 --top-p 0.9 --top-k 128 --repetition-penalty 1.1 --seed 42
        if ($LASTEXITCODE -ne 0) { throw "tarafer inference failed" }
    } else {
        Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue
        Write-Host "" 
        Write-Host "=== Tara Idea Generator ===" -ForegroundColor Cyan
        Write-Host "Each prompt is independent; conversation history is never retained." -ForegroundColor DarkGray
        Write-Host "Type /exit to quit." -ForegroundColor DarkGray
        while ($true) {
            $ideaPrompt = (Read-Host "`nIdea prompt").Trim()
            if ($ideaPrompt -in @("/exit", "exit", "/quit", "quit")) { break }
            if (-not $ideaPrompt) { continue }
            $rendered = "<|system|>$system<|/system|>`n<|user|>$ideaPrompt<|/user|>`n<|assistant|>"
            Write-Host ""
            & $tarafer $pack --prompt $rendered -n $N --ctx $Ctx --decode flash `
                --temperature 0.7 --top-p 0.9 --top-k 128 --repetition-penalty 1.1 --seed 42
            if ($LASTEXITCODE -ne 0) { throw "tarafer inference failed" }
        }
    }
} finally {
    foreach ($name in $names) {
        if ($null -eq $saved[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $saved[$name]
        }
    }
}
