param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$RuntimeDirectory = Join-Path $ProjectDirectory ".runtime"
$PythonExecutable = Join-Path $ProjectDirectory ".venv\Scripts\python.exe"
$VinextScript = Join-Path $ProjectDirectory "node_modules\vinext\dist\cli.js"

# Stop only processes previously started by this dashboard.
& (Join-Path $PSScriptRoot "stop_local.ps1")

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python environment missing. Run scripts\setup_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $VinextScript)) {
    throw "Frontend dependencies missing. Run scripts\setup_windows.ps1 first."
}

New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
$MatplotlibCache = Join-Path $RuntimeDirectory "matplotlib"
New-Item -ItemType Directory -Force -Path $MatplotlibCache | Out-Null
$env:MPLCONFIGDIR = $MatplotlibCache
$env:NODE_OPTIONS = "--max-old-space-size=2048"
$env:GESTURE_ENVIRONMENT = "production"
$env:GESTURE_REQUIRE_ARTIFACT_MANIFEST = "true"
$env:GESTURE_ALLOW_FORCE_LEARNING = "false"
$env:GESTURE_ALLOW_ARTIFACT_RELOAD = "false"
$BackendOutput = Join-Path $RuntimeDirectory "backend.stdout.log"
$BackendError = Join-Path $RuntimeDirectory "backend.stderr.log"
$FrontendOutput = Join-Path $RuntimeDirectory "frontend.stdout.log"
$FrontendError = Join-Path $RuntimeDirectory "frontend.stderr.log"
$NodeCommand = Get-Command node -ErrorAction Stop

Write-Host "Running production preflight..." -ForegroundColor Cyan
& $PythonExecutable (Join-Path $PSScriptRoot "production_preflight.py")
if ($LASTEXITCODE -ne 0) {
    throw "Production preflight failed. Re-import the trusted model artifacts and review the output above."
}

# The Vinext development server keeps client, RSC, and SSR compilers resident.
# Build first and use its much lighter production server for the live PC test.
Write-Host "Building the dashboard frontend..." -ForegroundColor Cyan
& $NodeCommand.Source $VinextScript "build"
if ($LASTEXITCODE -ne 0) {
    throw "Frontend build failed. Review the build output above."
}

$Backend = Start-Process -FilePath $PythonExecutable `
    -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $ProjectDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $BackendOutput -RedirectStandardError $BackendError

$QuotedVinextScript = '"' + $VinextScript + '"'
$Frontend = Start-Process -FilePath $NodeCommand.Source `
    -ArgumentList $QuotedVinextScript, "start", "--hostname", "127.0.0.1", "--port", "3000" `
    -WorkingDirectory $ProjectDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $FrontendOutput -RedirectStandardError $FrontendError

Set-Content -LiteralPath (Join-Path $RuntimeDirectory "backend.pid") -Value $Backend.Id
Set-Content -LiteralPath (Join-Path $RuntimeDirectory "frontend.pid") -Value $Frontend.Id

$BackendReady = $false
$FrontendReady = $false
for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
    Start-Sleep -Milliseconds 500
    try { $BackendReady = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/readyz" -TimeoutSec 1).StatusCode -eq 200 } catch { $BackendReady = $false }
    try { $FrontendReady = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:3000" -TimeoutSec 1).StatusCode -eq 200 } catch { $FrontendReady = $false }
    if ($BackendReady -and $FrontendReady) { break }
}

if (-not ($BackendReady -and $FrontendReady)) {
    Write-Host "One of the local servers did not become ready." -ForegroundColor Red
    Write-Host "Review logs in $RuntimeDirectory"
    & (Join-Path $PSScriptRoot "stop_local.ps1")
    exit 1
}

Write-Host "Gesture Control Lab is running." -ForegroundColor Green
Write-Host "Dashboard: http://127.0.0.1:3000"
Write-Host "API docs:  http://127.0.0.1:8000/docs"
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:3000"
}
