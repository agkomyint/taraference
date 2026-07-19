# Rebuild the taraference-native Q4 pack from the official Tara 1.4.1 release.

[CmdletBinding()]
param(
    [string]$Source = "D:\Tara_HQ\artifacts\release\tara1.4.1",
    [string]$Destination = "D:\Tara_HQ\departments\taraference_750_department\exports\tara-1.4.1-base-q4pack",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$exporter = "D:\Tara_HQ\departments\taraference_750_department\scripts\export_moe_q4_pack.py"

foreach ($path in @($Source, $exporter)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}

& $Python $exporter --src $Source --dst $Destination --arch tara_moe_141 --mode all
if ($LASTEXITCODE -ne 0) { throw "Tara 1.4.1 export failed" }

Write-Host "Tara 1.4.1 Q4 pack ready: $Destination" -ForegroundColor Green
Write-Host "Launch: D:\taraference\scripts\chat-tara-1.4.1.ps1" -ForegroundColor Cyan
