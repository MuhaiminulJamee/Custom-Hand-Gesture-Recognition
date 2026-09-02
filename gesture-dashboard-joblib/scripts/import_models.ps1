param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$ModelsDirectory = Join-Path $ProjectDirectory "models"
$PythonExecutable = Join-Path $ProjectDirectory ".venv\Scripts\python.exe"
$ResolvedSource = (Resolve-Path -LiteralPath $SourceDirectory).Path
$ResolvedModels = (Resolve-Path -LiteralPath $ModelsDirectory).Path

if (-not (Test-Path -LiteralPath $ResolvedSource -PathType Container)) {
    throw "Source folder does not exist: $ResolvedSource"
}
if ($ResolvedSource -eq $ResolvedModels) {
    throw "The source is already the dashboard models folder. Nothing to import."
}

$Patterns = @(
    "gesture_joblib_runtime_config.json",
    "gesture_primary_model.joblib",
    "gesture_online_state.joblib",
    "gesture_online_replay_cache.npz",
    "gesture_online_validation_cache.npz",
    "gesture_online_untouched_test_cache.npz",
    "gesture_online_confirmed_features.npz",
    "*models.joblib",
    "*model_comparison.csv",
    "*test_metrics.csv",
    "*classification_report.csv",
    "*mlp_history.csv"
)

$Files = foreach ($Pattern in $Patterns) {
    Get-ChildItem -LiteralPath $ResolvedSource -File -Filter $Pattern -ErrorAction SilentlyContinue
}
$Files = $Files | Sort-Object -Property FullName -Unique

if (-not $Files) {
    throw "No recognized v18_17 model exports were found in $ResolvedSource"
}

Write-Host "Importing trusted notebook artifacts..." -ForegroundColor Cyan
$ImportStamp = Get-Date -Format "yyyyMMddTHHmmss"
$BackupDirectory = Join-Path $ModelsDirectory ("import_backups\" + $ImportStamp)
foreach ($File in $Files) {
    $Destination = Join-Path $ResolvedModels $File.Name
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        New-Item -ItemType Directory -Force -Path $BackupDirectory | Out-Null
        Copy-Item -LiteralPath $Destination -Destination (Join-Path $BackupDirectory $File.Name)
    }
    $TemporaryDestination = $Destination + ".importing"
    Copy-Item -LiteralPath $File.FullName -Destination $TemporaryDestination -Force
    Move-Item -LiteralPath $TemporaryDestination -Destination $Destination -Force
    Write-Host "  copied $($File.Name)"
}

$ConfigPath = Join-Path $ResolvedModels "gesture_joblib_runtime_config.json"
$Bundle = Get-ChildItem -LiteralPath $ResolvedModels -File -Filter "*models.joblib" -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Warning "gesture_joblib_runtime_config.json is still missing."
}
if (-not $Bundle -and -not (Test-Path -LiteralPath (Join-Path $ResolvedModels "gesture_primary_model.joblib"))) {
    Write-Warning "No Python classifier was found. The recommended *models.joblib bundle is still missing."
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python environment missing. Run scripts\setup_windows.ps1 before importing models."
}

Write-Host "Running the frozen quality report..." -ForegroundColor Cyan
& $PythonExecutable (Join-Path $PSScriptRoot "qualify_release.py")
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Release qualification could not be completed. Review the output above."
}

Write-Host "Signing the imported runtime artifact set..." -ForegroundColor Cyan
& $PythonExecutable (Join-Path $PSScriptRoot "build_artifact_manifest.py")
if ($LASTEXITCODE -ne 0) {
    throw "Artifact manifest generation failed."
}

Write-Host "Model import finished. Restart the dashboard to load the new files." -ForegroundColor Green
