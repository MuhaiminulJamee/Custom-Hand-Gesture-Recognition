@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\verify_ncm_onnx_release.ps1"
pause
