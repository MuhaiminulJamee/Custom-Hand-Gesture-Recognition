$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectDirectory ".venv"
$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"

Write-Host "Preparing the independent Gesture Control Lab ONNX package..." -ForegroundColor Cyan

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "A 64-bit Windows installation is required."
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            & $PyLauncher.Source -3.12 -m venv $VirtualEnvironment
        } else {
            & $PyLauncher.Source -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" 2>$null
            if ($LASTEXITCODE -ne 0) {
                throw "Install 64-bit Python 3.11 or 3.12, then rerun setup."
            }
            & $PyLauncher.Source -3.11 -m venv $VirtualEnvironment
        }
    } else {
        $PythonCommand = Get-Command python -ErrorAction Stop
        & $PythonCommand.Source -c "import sys; assert sys.version_info[:2] in {(3, 11), (3, 12)} and sys.maxsize > 2**32"
        if ($LASTEXITCODE -ne 0) {
            throw "Install 64-bit Python 3.11 or 3.12, then rerun setup."
        }
        & $PythonCommand.Source -m venv $VirtualEnvironment
    }
}

& $PythonExecutable -m pip install --upgrade pip
& $PythonExecutable -m pip install -r (Join-Path $ProjectDirectory "requirements.lock.txt")
& $PythonExecutable (Join-Path $PSScriptRoot "check_onnx_runtime.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install the Microsoft Visual C++ 2015-2022 x64 Redistributable:" -ForegroundColor Yellow
    Write-Host "https://aka.ms/vc14/vc_redist.x64.exe"
    throw "ONNX Runtime native dependency check failed."
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectDirectory "models\hand_landmarker.task"))) {
    & $PythonExecutable (Join-Path $PSScriptRoot "download_hand_landmarker.py")
}

Push-Location $ProjectDirectory
try {
    $env:CI = "true"
    $PnpmCommand = Get-Command pnpm -ErrorAction SilentlyContinue
    if ($null -ne $PnpmCommand) {
        & $PnpmCommand.Source install --frozen-lockfile
    } else {
        $CorepackCommand = Get-Command corepack -ErrorAction Stop
        & $CorepackCommand.Source pnpm install --frozen-lockfile
    }
} finally {
    Pop-Location
}

& $PythonExecutable (Join-Path $PSScriptRoot "build_artifact_manifest.py")
& $PythonExecutable (Join-Path $PSScriptRoot "onnx_preflight.py")
if ($LASTEXITCODE -ne 0) {
    throw "ONNX deployment preflight failed."
}

Write-Host "ONNX setup complete." -ForegroundColor Green
Write-Host "Start it with start_onnx_dashboard.bat"
Write-Host "Dashboard: http://127.0.0.1:3100"
Write-Host "API docs:  http://127.0.0.1:8100/docs"
