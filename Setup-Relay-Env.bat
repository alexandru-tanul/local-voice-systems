@echo off
setlocal
cd /d "%~dp0"

echo Creating the speech recognition environment in relay_env...
if not exist "relay_env\Scripts\python.exe" (
    where py >nul 2>nul && (py -3.12 -m venv relay_env) || (python -m venv relay_env)
)
if not exist "relay_env\Scripts\python.exe" (
    echo Python was not found. Install 64-bit Python 3.12 and run this again.
    pause
    exit /b 1
)
"relay_env\Scripts\python.exe" -c "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo This setup needs 64-bit Python 3.12. Rename relay_env and run this again after installing it.
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
