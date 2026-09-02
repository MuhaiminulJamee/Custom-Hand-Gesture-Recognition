$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectDirectory ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Run setup_onnx_dashboard.bat first."
}

Push-Location $ProjectDirectory
try {
    & $PythonExecutable -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Python dependency check failed." }
    & $PythonExecutable -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Python tests failed." }
    & $PythonExecutable (Join-Path $PSScriptRoot "onnx_preflight.py")
    if ($LASTEXITCODE -ne 0) { throw "ONNX preflight failed." }
    & ".\node_modules\.bin\tsc.cmd" --noEmit
    if ($LASTEXITCODE -ne 0) { throw "TypeScript check failed." }
    & ".\node_modules\.bin\eslint.cmd" . --ignore-pattern dist --ignore-pattern .next
    if ($LASTEXITCODE -ne 0) { throw "ESLint failed." }
    $env:NEXT_PUBLIC_GESTURE_API_URL = "http://127.0.0.1:8100"
    & node ".\node_modules\vinext\dist\cli.js" build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
} finally {
    Pop-Location
}

Write-Host "Independent ONNX release verification passed." -ForegroundColor Green
