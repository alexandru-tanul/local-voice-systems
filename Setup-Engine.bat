@echo off
setlocal
cd /d "%~dp0"

rem The setup script needs Python 3.12 on Windows, the same Python the dashboard uses.
set "PYTHON="
if exist "relay_env\Scripts\python.exe" set "PYTHON=relay_env\Scripts\python.exe"
rem Each is tried, as py can lack 3.12 and python can be the Microsoft Store's placeholder.
if not defined PYTHON (
    py -3.12 -c "" >nul 2>nul && set "PYTHON=py -3.12"
)
if not defined PYTHON (
    python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul && set "PYTHON=python"
)
if not defined PYTHON (
    echo Python was not found. Install 64-bit Python 3.12 and run this again.
    pause
    exit /b 1
)
%PYTHON% scripts\setup_engine.py %*
set "CODE=%errorlevel%"
pause
exit /b %CODE%
