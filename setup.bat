@echo off
rem [설명] 처음 실행할 때 Python 3.11 가상환경을 준비합니다.
setlocal
cd /d "%~dp0"
echo [ACE] Python 3.11 environment setup

rem Prefer the Windows Python launcher.
py -3.11 -c "import struct; assert struct.calcsize('P') == 8" >nul 2>&1
if not errorlevel 1 (
  py -3.11 setup_env.py %*
  goto done
)

rem Also support installations not registered with the launcher.
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  "%LocalAppData%\Programs\Python\Python311\python.exe" setup_env.py %*
  goto done
)
python -c "import sys,struct; assert sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8" >nul 2>&1
if not errorlevel 1 (
  python setup_env.py %*
  goto done
)

echo 64-bit Python 3.11 was not found.
echo Install Python 3.11 from https://www.python.org/downloads/windows/
echo Enable the Python launcher or Add python.exe to PATH, then retry.
pause
exit /b 1

:done
set "ACE_SETUP_EXIT=%ERRORLEVEL%"
if not "%ACE_SETUP_EXIT%"=="0" echo [ACE] Setup failed. Read the error above.
pause
exit /b %ACE_SETUP_EXIT%
