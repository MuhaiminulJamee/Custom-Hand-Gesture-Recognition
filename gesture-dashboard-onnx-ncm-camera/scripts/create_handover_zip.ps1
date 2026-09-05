$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$ParentDirectory = Split-Path -Parent $ProjectDirectory
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ArchivePath = Join-Path $ParentDirectory "gesture-dashboard-onnx-ncm-camera-handover-$Timestamp.zip"
$TemporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$StagingDirectory = [System.IO.Path]::GetFullPath(
    (Join-Path $TemporaryRoot "gesture-dashboard-onnx-ncm-camera-handover-$Timestamp")
)
$PackageDirectory = Join-Path $StagingDirectory "gesture-dashboard-onnx-ncm-camera"

if (-not $StagingDirectory.StartsWith($TemporaryRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe handover staging path: $StagingDirectory"
}
if (Test-Path -LiteralPath $StagingDirectory) {
    throw "Handover staging path already exists: $StagingDirectory"
}

New-Item -ItemType Directory -Path $PackageDirectory -Force | Out-Null
$ExcludedNames = @(
    ".venv", "node_modules", ".runtime", ".next", ".vinext", "dist",
    ".pytest_cache", "__pycache__", "feedback", "tsconfig.tsbuildinfo"
)

Get-ChildItem -LiteralPath $ProjectDirectory -Force | Where-Object {
    $_.Name -notin $ExcludedNames
} | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $PackageDirectory -Recurse -Force
}

Compress-Archive -LiteralPath $PackageDirectory -DestinationPath $ArchivePath -CompressionLevel Optimal
Remove-Item -LiteralPath $StagingDirectory -Recurse -Force
Write-Host "Created ONNX + NCM camera handover package:" -ForegroundColor Green
Write-Host $ArchivePath
