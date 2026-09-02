$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectDirectory ".venv"
$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"

Write-Host "Preparing Gesture Control Lab..." -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PythonCommand) {
        & py -3.11 -m venv $VirtualEnvironment
    } else {
        $PythonCommand = Get-Command python -ErrorAction Stop
        & $PythonCommand.Source -m venv $VirtualEnvironment
    }
}

& $PythonExecutable -m pip install --upgrade pip
$LockedRequirements = Join-Path $ProjectDirectory "requirements.lock.txt"
if (Test-Path -LiteralPath $LockedRequirements) {
    & $PythonExecutable -m pip install -r $LockedRequirements
} else {
    & $PythonExecutable -m pip install -r (Join-Path $ProjectDirectory "requirements.txt")
}
& $PythonExecutable (Join-Path $PSScriptRoot "download_hand_landmarker.py")

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

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Copy the notebook exports into: $(Join-Path $ProjectDirectory 'models')"
Write-Host "Then double-click start_dashboard.bat"
