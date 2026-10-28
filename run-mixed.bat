@echo off
setlocal
cd /d "%~dp0"
set "ACE_PYTHON=.venv\Scripts\python.exe"
if not exist "%ACE_PYTHON%" set "ACE_PYTHON=..\.venv_yolo\Scripts\python.exe"
if not exist "%ACE_PYTHON%" (
  echo [ACE] setup.bat first: Python 3.11 environment is required.
  pause
  exit /b 1
)
rem Run the real models on mixed JPG samples, then compare each cat/dog count with manual truth.
"%ACE_PYTHON%" "source\main-mixed.py" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [ACE] Mixed test not passed. Check outputs\mixed_samples\혼합_테스트_요약.md.
pause
exit /b %RESULT%
