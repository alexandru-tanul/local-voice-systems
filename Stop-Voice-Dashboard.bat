@echo off
setlocal

echo Stopping the voice dashboard and its services...

powershell.exe -NoProfile -Command "try { Invoke-RestMethod -Method Post -ContentType 'application/json' -Body '{}' -Uri 'http://127.0.0.1:8790/api/stop-system' -TimeoutSec 15 | Out-Null } catch {}" >nul 2>nul

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8790" ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>nul
)

echo Stopped.
pause
