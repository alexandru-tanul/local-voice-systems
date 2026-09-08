@echo off
setlocal
cd /d "%~dp0"

echo Creating the speech recognition environment in relay_env...
if not exist "relay_env\Scripts\python.exe" (
    where py >nul 2>nul && (py -3 -m venv relay_env) || (python -m venv relay_env)
)
if not exist "relay_env\Scripts\python.exe" (
    echo Python was not found. Install Python 3.11 or newer and run this again.
    pause
    exit /b 1
)
"relay_env\Scripts\python.exe" -m pip install --upgrade pip
"relay_env\Scripts\python.exe" -m pip install -r requirements-relay.txt
if errorlevel 1 (
    echo Installation failed. Check the messages above.
    pause
    exit /b 1
)
echo Speech recognition environment ready. The Whisper model downloads on first use.
pause
