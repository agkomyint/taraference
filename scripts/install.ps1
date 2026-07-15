<#
.SYNOPSIS
  Production-oriented setup for taraference (Windows).

.DESCRIPTION
  Checks / installs what can be automated:
    - Rust (rustup)
    - MSVC C++ build tools (via winget, if missing)
    - NVIDIA driver / CUDA toolkit presence (install toolkit via winget if missing)
  Then builds release binary and optionally downloads GGUF models.

  CUDA Toolkit and GPU drivers often need an interactive installer and a reboot.
  This script never force-reboots; it prints clear next steps when blocked.

.PARAMETER SkipModels
  Do not download GGUF weights.

.PARAMETER SkipBuild
  Only check/install toolchains; do not cargo build.

.PARAMETER Models
  Which models to download: all | 0.5b | 3b  (default: all)

.PARAMETER Force
  Re-download models even if present.

.PARAMETER ModelsDir
  Directory for GGUFs (default: <repo>/models)

.EXAMPLE
  .\scripts\install.ps1
  .\scripts\install.ps1 -Models 0.5b
  .\scripts\install.ps1 -SkipModels
#>

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$SkipBuild,
    [ValidateSet("all", "0.5b", "3b")]
    [string]$Models = "all",
    [switch]$Force,
    [string]$ModelsDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not $ModelsDir) {
    $ModelsDir = Join-Path $Root "models"
}

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Write-Ok($msg) { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "  XX  $msg" -ForegroundColor Red }

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Ensure-Rust {
    Write-Step "Rust toolchain"
    if (Test-Command "cargo") {
        $v = & cargo --version 2>&1
        Write-Ok $v
        return
    }
    Write-Warn "cargo not found — installing rustup (default stable toolchain)"
    $rustup = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustup -UseBasicParsing
    & $rustup -y --default-toolchain stable
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
    $env:Path = "$cargoHome\bin;" + $env:Path
    if (-not (Test-Command "cargo")) {
        Write-Fail "cargo still not on PATH. Open a new terminal and re-run this script."
        exit 1
    }
    Write-Ok (& cargo --version)
}

function Ensure-Msvc {
    Write-Step "MSVC C++ linker (required for cargo on Windows)"
    if (Test-Command "link") {
        Write-Ok "link.exe found on PATH"
        return
    }
    # vswhere is the reliable detector
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($inst) {
            Write-Ok "Visual Studio C++ tools present: $inst"
            Write-Warn "If cargo fails to link, open 'x64 Native Tools Command Prompt' or run VsDevCmd.bat"
            return
        }
    }
    if (Test-Command "winget") {
        Write-Warn "Installing Visual Studio 2022 Build Tools (C++) via winget — may need admin / UAC"
        try {
            winget install -e --id Microsoft.VisualStudio.2022.BuildTools `
                --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" `
                --accept-package-agreements --accept-source-agreements
            Write-Ok "Build Tools install requested. Re-open the terminal if cargo cannot find link.exe."
        } catch {
            Write-Warn "winget install failed: $_"
            Write-Warn "Install manually: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            Write-Warn "Workload: Desktop development with C++"
        }
    } else {
        Write-Fail "No MSVC linker and no winget. Install Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    }
}

function Ensure-Nvidia {
    Write-Step "NVIDIA GPU driver"
    if (-not (Test-Command "nvidia-smi")) {
        Write-Fail "nvidia-smi not found. Install a recent NVIDIA Game Ready / Studio driver."
        Write-Warn "https://www.nvidia.com/Download/index.aspx"
        return $false
    }
    $smi = & nvidia-smi 2>&1 | Out-String
    $line = ($smi -split "`n" | Where-Object { $_ -match "CUDA Version" } | Select-Object -First 1)
    Write-Ok "nvidia-smi OK"
    if ($line) { Write-Host "      $line" }
    return $true
}

function Ensure-CudaToolkit {
    Write-Step "CUDA Toolkit (driver API + NVRTC — required at runtime)"
    $hasNvcc = Test-Command "nvcc"
    $cudaRoot = $env:CUDA_PATH
    if (-not $cudaRoot) {
        $candidates = @(
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
        )
        foreach ($c in $candidates) {
            if (Test-Path (Join-Path $c "bin\nvrtc64*.dll")) { $cudaRoot = $c; break }
            if (Test-Path (Join-Path $c "bin")) { $cudaRoot = $c; break }
        }
    }

    if ($hasNvcc) {
        Write-Ok (& nvcc --version | Select-String "release" | ForEach-Object { $_.Line.Trim() })
    } elseif ($cudaRoot) {
        Write-Ok "CUDA_PATH-like install at $cudaRoot"
        $env:Path = "$cudaRoot\bin;" + $env:Path
        $env:CUDA_PATH = $cudaRoot
    } else {
        Write-Warn "CUDA Toolkit not detected (need NVRTC for runtime kernel compile)."
        Write-Warn "This project targets CUDA 13.x (cudarc feature cuda-13020)."
        if (Test-Command "winget") {
            Write-Warn "Attempting winget install of CUDA toolkit (large; may require admin)..."
            try {
                winget install -e --id Nvidia.CUDA --accept-package-agreements --accept-source-agreements
            } catch {
                Write-Warn "winget CUDA install failed: $_"
            }
        }
        Write-Warn "Manual: https://developer.nvidia.com/cuda-downloads  (Windows x86_64)"
        Write-Warn "After install, restart the shell so PATH/CUDA_PATH update."
        return $false
    }

    # Spot-check NVRTC DLL (name varies by version)
    $bin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { $null }
    if ($bin -and (Test-Path $bin)) {
        $nvrtc = Get-ChildItem $bin -Filter "nvrtc*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($nvrtc) { Write-Ok "NVRTC: $($nvrtc.Name)" }
        else { Write-Warn "nvrtc*.dll not found under $bin — runtime compile may fail" }
    }
    return $true
}

function Build-Release {
    Write-Step "cargo build --release"
    if (-not (Test-Command "cargo")) {
        Write-Fail "cargo missing"
        exit 1
    }
    & cargo build --release -p taraference
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "build failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    $exe = Join-Path $Root "target\release\taraference.exe"
    if (Test-Path $exe) { Write-Ok $exe }
    else { Write-Fail "binary not found at $exe"; exit 1 }
}

function Download-Models {
    Write-Step "Download GGUF models → $ModelsDir"
    $exe = Join-Path $Root "target\release\taraference.exe"
    if (-not (Test-Path $exe)) {
        Write-Fail "build the project first (missing $exe)"
        exit 1
    }
    $args = @("--download", $Models, "--models-dir", $ModelsDir)
    if ($Force) { $args += "--force" }
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "model download failed"
        exit $LASTEXITCODE
    }
}

# ── main ────────────────────────────────────────────────────────────────────
Write-Host "taraference install (Windows)" -ForegroundColor White
Write-Host "repo: $Root"

Ensure-Rust
Ensure-Msvc
$gpu = Ensure-Nvidia
$cuda = Ensure-CudaToolkit

if (-not $gpu) {
    Write-Fail "No usable NVIDIA GPU driver — inference will not run."
    Write-Warn "You can still compile if CUDA toolkit + MSVC are present."
}

if (-not $SkipBuild) {
    if (-not $cuda) {
        Write-Warn "CUDA toolkit incomplete — build may still succeed; runtime will fail without NVRTC."
    }
    Build-Release
}

if (-not $SkipModels -and -not $SkipBuild) {
    Download-Models
} elseif (-not $SkipModels -and $SkipBuild) {
    Write-Warn "Skipping model download (need a built binary). Run without -SkipBuild, or: cargo run --release -- --download"
}

Write-Step "Summary"
Write-Host @"
  Build:   cargo build --release
  Binary:  target\release\taraference.exe
  Models:  $ModelsDir

  Run chat:
    .\target\release\taraference.exe models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

  Serve API:
    .\target\release\taraference.exe models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve 3000

  Profile:
    .\target\release\taraference.exe models\Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile

  Re-download models only:
    .\target\release\taraference.exe --download
"@

if (-not $gpu -or -not $cuda) {
    Write-Warn "Fix GPU driver / CUDA Toolkit, then re-run this script or just cargo build --release"
    exit 2
}

Write-Ok "Install finished."
exit 0
