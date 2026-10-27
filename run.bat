@echo off
setlocal
cd /d "%~dp0"
set "ACE_PYTHON=.venv\Scripts\python.exe"
if not exist "%ACE_PYTHON%" set "ACE_PYTHON=..\.venv_yolo\Scripts\python.exe"
if not exist "%ACE_PYTHON%" (
  echo [ACE] Run setup.bat first to create the Python 3.11 environment.
  pause
  exit /b 1
)
"%ACE_PYTHON%" "source\main.py" %*
set "ACE_RUN_EXIT=%ERRORLEVEL%"
if not "%ACE_RUN_EXIT%"=="0" echo [ACE] Execution failed. Read the message above.
pause
exit /b %ACE_RUN_EXIT%
