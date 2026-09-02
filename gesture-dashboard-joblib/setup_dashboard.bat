@echo off
setlocal
title Gesture Control Lab - Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
if errorlevel 1 (
  echo.
  echo Setup failed. Read the message above, then see README.md.
  pause
  exit /b 1
)
echo.
echo Setup finished successfully.
pause
