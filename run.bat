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
rem Match the teacher's numbered .jpg names without changing the original photos silently.
"%ACE_PYTHON%" "tools\사진변환.py"
if errorlevel 1 (
  echo [ACE] Photo conversion failed. Check tools\사진변환.py.
  pause
  exit /b 1
)
"%ACE_PYTHON%" "source\main-3.py" %*
set "ACE_RUN_EXIT=%ERRORLEVEL%"
if not "%ACE_RUN_EXIT%"=="0" echo [ACE] Execution failed. Read the message above.
pause
exit /b %ACE_RUN_EXIT%
