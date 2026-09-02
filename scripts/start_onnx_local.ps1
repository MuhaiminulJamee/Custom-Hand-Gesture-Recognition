param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$RuntimeDirectory = Join-Path $ProjectDirectory ".runtime"
$PythonExecutable = Join-Path $ProjectDirectory ".venv\Scripts\python.exe"
$VinextScript = Join-Path $ProjectDirectory "node_modules\vinext\dist\cli.js"

& (Join-Path $PSScriptRoot "stop_onnx_local.ps1")

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Independent ONNX Python environment missing. Run setup_onnx_dashboard.bat first."
}
if (-not (Test-Path -LiteralPath $VinextScript)) {
    throw "Independent frontend environment missing. Run setup_onnx_dashboard.bat first."
}

New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
$MatplotlibCache = Join-Path $RuntimeDirectory "matplotlib"
New-Item -ItemType Directory -Force -Path $MatplotlibCache | Out-Null
$env:MPLCONFIGDIR = $MatplotlibCache
$env:NODE_OPTIONS = "--max-old-space-size=2048"
$env:NEXT_PUBLIC_GESTURE_API_URL = "http://127.0.0.1:8100"
$env:GESTURE_ENVIRONMENT = "production"
$env:GESTURE_REQUIRE_ARTIFACT_MANIFEST = "true"
$env:GESTURE_ALLOW_FORCE_LEARNING = "false"
$env:GESTURE_ALLOW_ARTIFACT_RELOAD = "false"
$BackendOutput = Join-Path $RuntimeDirectory "backend.stdout.log"
$BackendError = Join-Path $RuntimeDirectory "backend.stderr.log"
$FrontendOutput = Join-Path $RuntimeDirectory "frontend.stdout.log"
$FrontendError = Join-Path $RuntimeDirectory "frontend.stderr.log"
$NodeCommand = Get-Command node -ErrorAction Stop

Write-Host "Running ONNX deployment preflight..." -ForegroundColor Cyan
& $PythonExecutable (Join-Path $PSScriptRoot "onnx_preflight.py")
if ($LASTEXITCODE -ne 0) {
    throw "ONNX preflight failed. Review the output above."
}

Write-Host "Building the independent ONNX dashboard..." -ForegroundColor Cyan
& $NodeCommand.Source $VinextScript "build"
if ($LASTEXITCODE -ne 0) {
    throw "Frontend build failed. Review the output above."
}

$Backend = Start-Process -FilePath $PythonExecutable `
    -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8100" `
    -WorkingDirectory $ProjectDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $BackendOutput -RedirectStandardError $BackendError

$QuotedVinextScript = '"' + $VinextScript + '"'
$Frontend = Start-Process -FilePath $NodeCommand.Source `
    -ArgumentList $QuotedVinextScript, "start", "--hostname", "127.0.0.1", "--port", "3100" `
    -WorkingDirectory $ProjectDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $FrontendOutput -RedirectStandardError $FrontendError

Set-Content -LiteralPath (Join-Path $RuntimeDirectory "backend.pid") -Value $Backend.Id
Set-Content -LiteralPath (Join-Path $RuntimeDirectory "frontend.pid") -Value $Frontend.Id

$BackendReady = $false
$FrontendReady = $false
for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    Start-Sleep -Milliseconds 500
    try { $BackendReady = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8100/readyz" -TimeoutSec 1).StatusCode -eq 200 } catch { $BackendReady = $false }
    try { $FrontendReady = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:3100" -TimeoutSec 1).StatusCode -eq 200 } catch { $FrontendReady = $false }
    if ($BackendReady -and $FrontendReady) { break }
}

if (-not ($BackendReady -and $FrontendReady)) {
    Write-Host "One of the independent ONNX servers did not become ready." -ForegroundColor Red
    Write-Host "Review logs in $RuntimeDirectory"
    & (Join-Path $PSScriptRoot "stop_onnx_local.ps1")
    exit 1
}

Write-Host "Gesture Control Lab ONNX is running." -ForegroundColor Green
Write-Host "Dashboard: http://127.0.0.1:3100"
Write-Host "API docs:  http://127.0.0.1:8100/docs"
Write-Host "The Joblib dashboard may continue running independently on ports 3000/8000."
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:3100"
}
