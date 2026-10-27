@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ACE] Run setup.bat first to create the Python 3.11 environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "source\main.py" %*
set "ACE_RUN_EXIT=%ERRORLEVEL%"
if not "%ACE_RUN_EXIT%"=="0" echo [ACE] Execution failed. Read the message above.
pause
exit /b %ACE_RUN_EXIT%
