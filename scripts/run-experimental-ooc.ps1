$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot

function Find-TaraMoePack {
    $searchRoots = @(
        (Join-Path $repoRoot "models"),
        $repoRoot
    )

    foreach ($root in $searchRoots) {
        if (-not (Test-Path -LiteralPath $root -PathType Container)) {
            continue
        }

        $metadataFiles = Get-ChildItem -LiteralPath $root -Filter "meta.json" -File -Recurse -ErrorAction SilentlyContinue
        foreach ($metadataFile in $metadataFiles) {
            try {
                $metadata = Get-Content -LiteralPath $metadataFile.FullName -Raw | ConvertFrom-Json
                if ($metadata.format -in @("tara_moe_q8_v1", "tara_moe_q4_v1")) {
                    return $metadataFile.Directory.FullName
                }
            }
            catch {
                continue
            }
        }
    }

    return $null
}

Push-Location -LiteralPath $repoRoot
try {
    Clear-Host
    Write-Host "Taraference Experimental Chat" -ForegroundColor Cyan
    Write-Host ""

    $modelPack = Find-TaraMoePack
    if ($null -eq $modelPack) {
        Write-Host "No Tara MoE model pack was found automatically."
        $modelPack = Read-Host "Paste the path to your Tara MoE pack folder"
    }

    $modelPack = $modelPack.Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $modelPack -PathType Container)) {
        throw "Model pack folder not found: $modelPack"
    }

    $modelPack = (Resolve-Path -LiteralPath $modelPack).Path
    $metadataPath = Join-Path $modelPack "meta.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        throw "This is not a Tara MoE pack (meta.json is missing): $modelPack"
    }

    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.format -notin @("tara_moe_q8_v1", "tara_moe_q4_v1")) {
        throw "Unsupported model pack format: $($metadata.format)"
    }

    Write-Host "Model: $modelPack" -ForegroundColor Green
    Write-Host "Building and starting chat..." -ForegroundColor Yellow
    Write-Host "In chat, use /reset to clear history and /quit to exit."
    Write-Host ""

    & cargo run --release -p taraference -- `
        $modelPack `
        --experimental `
        --no-cuda-graph `
        --ctx 1024

    if ($LASTEXITCODE -ne 0) {
        throw "tarafer exited with code $LASTEXITCODE"
    }
}
catch {
    Write-Host ""
    Write-Host "Could not start chat:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}
finally {
    Pop-Location
}
