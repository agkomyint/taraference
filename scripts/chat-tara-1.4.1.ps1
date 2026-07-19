# Interactive Tara 1.4.1 Base launcher for its dedicated top-2 MoE backend.
# Preserves the model's trained routing and full 32K vocabulary.

[CmdletBinding()]
param(
    [string]$Prompt = "",
    [int]$N = 256,
    [int]$Ctx = 1024,
    [switch]$Sft,
    [switch]$Q4Fast,
    [switch]$Benchmark,
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$samplingBuild = Join-Path $repo "target-sampling\release\tarafer.exe"
$tarafer = if (Test-Path -LiteralPath $samplingBuild) {
    $samplingBuild
} else {
    Join-Path $repo "target\release\tarafer.exe"
}
$exportRoot = "D:\Tara_HQ\departments\taraference_750_department\exports"
$pack = if ($Sft) {
    if ($Q4Fast) { throw "No Q4 SFT pack is qualified; omit -Q4Fast for SFT quality testing." }
    Join-Path $exportRoot "tara-1.4.1-alpaca-sft-q8pack"
} elseif ($Q4Fast) {
    Join-Path $exportRoot "tara-1.4.1-base-q4pack"
} else {
    Join-Path $exportRoot "tara-1.4.1-base-q8pack"
}
$tokenizer = Join-Path $repo "models\tara-sprint-80m-Q8_0.gguf"

foreach ($path in @($tarafer, $pack, $tokenizer)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}

# Save caller state because environment variables are process-wide in PowerShell.
$names = @(
    "TARAFER_TOKENIZER_GGUF", "TARAFER_FULL_VOCAB", "TARAFER_VOCAB_LIMIT",
    "TARAFER_SPEED", "TARAFER_MOE_TOPK", "TARAFER_MOE_WARPS",
    "TARAFER_IGNORE_EOS", "TARAFER_N_CTX", "TARAFER_TARA141_SFT"
)
$saved = @{}
foreach ($name in $names) {
    $item = Get-Item "Env:$name" -ErrorAction SilentlyContinue
    $saved[$name] = if ($null -ne $item) { $item.Value } else { $null }
}

try {
    $env:TARAFER_TOKENIZER_GGUF = $tokenizer
    $env:TARAFER_FULL_VOCAB = "1"
    $env:TARAFER_MOE_WARPS = "4" # measured best for d=448, FFN=1536, top-2
    $env:TARAFER_N_CTX = "$Ctx"
    Remove-Item Env:TARAFER_VOCAB_LIMIT,Env:TARAFER_SPEED,Env:TARAFER_MOE_TOPK -ErrorAction SilentlyContinue
    if ($Sft) {
        $env:TARAFER_TARA141_SFT = "1"
    } else {
        Remove-Item Env:TARAFER_TARA141_SFT -ErrorAction SilentlyContinue
    }

    Write-Host "" 
    $variant = if ($Sft) { "Alpaca Assistant SFT" } else { "Base" }
    $quant = if ($Q4Fast) { "Q4 fast" } else { "Q8 quality reference" }
    Write-Host "=== Tara 1.4.1 $variant — dedicated top-2 MoE ===" -ForegroundColor Cyan
    Write-Host "pack     : $pack"
    Write-Host "routing  : real top-2 (trained behavior)"
    Write-Host "vocab    : full 32768"
    Write-Host "context  : $Ctx   max_new: $N"
    Write-Host "quant    : $quant"
    Write-Host "sampling : temperature=0.7 top_p=0.9 repetition_penalty=1.1 seed=42" -ForegroundColor Green

    if ($Benchmark) {
        $env:TARAFER_IGNORE_EOS = "1"
        $benchPrompt = if ($Prompt) { $Prompt } else { "A small creative project can begin with" }
        Write-Host "mode     : fixed-length benchmark ($Runs runs)" -ForegroundColor Yellow
        Write-Host ""
        for ($i = 1; $i -le $Runs; $i++) {
            Write-Host "--- run $i/$Runs ---" -ForegroundColor DarkCyan
            & $tarafer $pack --prompt $benchPrompt -n $N --ctx $Ctx --decode flash
            if ($LASTEXITCODE -ne 0) { throw "tarafer benchmark run $i failed" }
        }
    } elseif ($Prompt) {
        Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue
        $effectivePrompt = if ($Sft) {
            "<|system|>You are Tara, a helpful assistant. Answer directly and concisely.<|/system|>`n<|user|>$Prompt<|/user|>`n<|assistant|>"
        } else {
            $Prompt
        }
        Write-Host "mode     : one-shot $(if ($Sft) { 'SFT response' } else { 'base continuation' })"
        Write-Host ""
        & $tarafer $pack --prompt $effectivePrompt -n $N --ctx $Ctx --decode flash --temperature 0.7 --top-p 0.9 --top-k 128 --repetition-penalty 1.1 --seed 42
        if ($LASTEXITCODE -ne 0) { throw "tarafer one-shot failed" }
    } else {
        Remove-Item Env:TARAFER_IGNORE_EOS -ErrorAction SilentlyContinue
        Write-Host "mode     : interactive (base model, not an instruction-tuned assistant)"
        Write-Host "commands : /reset | /quit"
        Write-Host ""
        & $tarafer $pack -n $N --ctx $Ctx --decode flash --temperature 0.7 --top-p 0.9 --top-k 128 --repetition-penalty 1.1 --seed 42
        if ($LASTEXITCODE -ne 0) { throw "tarafer chat failed" }
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
