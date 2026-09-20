@echo off
REM Drag and drop any DST/PES file onto this file to run FillFinder.
cd /d "%~dp0"
if exist venv\Scripts\python.exe (set PY=venv\Scripts\python.exe) else (set PY=python)
if "%~1"=="" (
  echo Drag a design file onto run.bat, or: run.bat design.dst
  pause
  exit /b
)
%PY% fillfinder.py "%~1" %2 %3 %4 %5 %6 %7 %8 %9
pause
