$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$RuntimeDirectory = Join-Path $ProjectDirectory ".runtime"

foreach ($Name in @("backend", "frontend")) {
    $PidPath = Join-Path $RuntimeDirectory "$Name.pid"
    if (-not (Test-Path -LiteralPath $PidPath)) { continue }
    $ProcessId = [int](Get-Content -LiteralPath $PidPath -Raw)
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $Process) {
        try {
            Stop-Process -InputObject $Process -Force -ErrorAction Stop
        } catch {
            # Some Windows/Node combinations throw a PowerShell null-reference
            # error for hidden child processes. Fall back to this exact recorded
            # PID and its process tree; no unrelated processes are targeted.
            & taskkill.exe /PID $ProcessId /T /F | Out-Null
            if ($LASTEXITCODE -ne 0 -and $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
                throw "Could not stop $Name server (PID $ProcessId)."
            }
        }
        Write-Host "Stopped $Name server (PID $ProcessId)."
    }
    Remove-Item -LiteralPath $PidPath -Force
}
