@echo off
setlocal

echo Stopping the voice dashboard and its services...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Stop-Voice-Dashboard.ps1"
echo Stopped.
pause
