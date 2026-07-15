<#
.SYNOPSIS
  One-command setup for taraference (Windows).

.DESCRIPTION
  No flags required for the default path:
    install Rust, check MSVC/CUDA/GPU, cargo build --release, download models.

  Optional:
    -SkipModels
    -Force   (re-download models)

.EXAMPLE
  .\scripts\install.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$ModelsDir = Join-Path $Root "models"

function Step($m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  !!  $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  XX  $m" -ForegroundColor Red }

function Have($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

Write-Host "taraference setup"
Write-Host "repo: $Root"

# ── Rust ────────────────────────────────────────────────────────────────────
Step "Rust"
if (Have "cargo") {
    Ok (& cargo --version)
} else {
    Warn "installing rustup…"
    $rustup = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustup -UseBasicParsing
    & $rustup -y --default-toolchain stable
    $cargoBin = Join-Path (if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }) "bin"
    $env:Path = "$cargoBin;" + $env:Path
    if (-not (Have "cargo")) {
        Fail "cargo not on PATH — open a new terminal and re-run"
        exit 1
    }
    Ok (& cargo --version)
}

# ── MSVC ────────────────────────────────────────────────────────────────────
Step "C++ linker (MSVC)"
if (Have "link") {
    Ok "link.exe on PATH"
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $ok = $false
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($inst) { Ok "VS C++ tools: $inst"; $ok = $true }
    }
    if (-not $ok) {
        Warn "Install Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/  (Desktop development with C++)"
        if (Have "winget") {
            Warn "trying winget install of VS Build Tools…"
            try {
                winget install -e --id Microsoft.VisualStudio.2022.BuildTools `
                    --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" `
                    --accept-package-agreements --accept-source-agreements
            } catch { Warn "winget: $_" }
        }
    }
}

# ── GPU / CUDA (info only for build) ────────────────────────────────────────
Step "GPU / CUDA (for running models)"
if (Have "nvidia-smi") {
    Ok "nvidia-smi found"
    & nvidia-smi -L 2>$null
} else {
    Warn "no nvidia-smi — build may work; inference needs NVIDIA GPU + driver"
}
if (Have "nvcc") {
    Ok ((& nvcc --version) | Select-String "release" | ForEach-Object { $_.Line.Trim() })
} elseif ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
    Ok "CUDA_PATH=$env:CUDA_PATH"
    $env:Path = "$env:CUDA_PATH\bin;" + $env:Path
} else {
    Warn "CUDA toolkit not detected — runtime needs CUDA 13.x + NVRTC"
    Warn "https://developer.nvidia.com/cuda-downloads"
}

# ── Build ───────────────────────────────────────────────────────────────────
Step "Build (cargo build --release)"
& cargo build --release -p taraference
if ($LASTEXITCODE -ne 0) { Fail "build failed"; exit $LASTEXITCODE }
$exe = Join-Path $Root "target\release\taraference.exe"
if (-not (Test-Path $exe)) { Fail "missing $exe"; exit 1 }
Ok $exe

# ── Models ──────────────────────────────────────────────────────────────────
if (-not $SkipModels) {
    Step "Download models → models\"
    $args = @("--download", "all", "--models-dir", $ModelsDir)
    if ($Force) { $args += "--force" }
    & $exe @args
    if ($LASTEXITCODE -ne 0) { Fail "download failed"; exit $LASTEXITCODE }
} else {
    Warn "skipped model download"
}

Step "Done"
Write-Host @"
  Binary:  $exe
  Models:  $ModelsDir

  Chat:
    $exe models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

  Server:
    $exe models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve 3000
"@
Ok "setup finished"
