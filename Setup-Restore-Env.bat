@echo off
setlocal
cd /d "%~dp0"

echo Creating the audio restoration environment in restore_env. It downloads about 1 GB.
if not exist "restore_env\Scripts\python.exe" (
    where py >nul 2>nul && (py -3.12 -m venv restore_env) || (python -m venv restore_env)
)
if not exist "restore_env\Scripts\python.exe" (
    echo Python was not found. Install 64-bit Python 3.12 and run this again.
    pause
    exit /b 1
)
"restore_env\Scripts\python.exe" -c "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo This setup needs 64-bit Python 3.12. Rename restore_env and run this again after installing it.
    pause
    exit /b 1
)
"restore_env\Scripts\python.exe" -m pip install --upgrade pip
"restore_env\Scripts\python.exe" -m pip install -r requirements-restore.txt
if errorlevel 1 goto failed
rem LavaSR names its Vocos fork by branch, which pip would not match to the pinned commit above.
"restore_env\Scripts\python.exe" -m pip install --no-deps "LavaSR @ git+https://github.com/ysharma3501/LavaSR.git@33ac040892519c1bb4aed7eb32e79af51cc29e2a"
if errorlevel 1 goto failed
echo Audio restoration ready. LavaSR downloads its models (about 100 MB) the first time it runs.
pause
exit /b 0

:failed
echo Installation failed. Check the messages above.
pause
exit /b 1
