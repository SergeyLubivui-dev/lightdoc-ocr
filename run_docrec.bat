@echo off
setlocal
set IMG=%~1
if "%IMG%"=="" set IMG=%~dp0samples\synth1.jpg
set OUT=%~2
if "%OUT%"=="" set OUT=%~dp0out
"%~dp0.venv\Scripts\python.exe" "%~dp0recognize.py" "%IMG%" --out "%OUT%" %3 %4
