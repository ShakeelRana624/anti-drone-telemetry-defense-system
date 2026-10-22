@echo off
setlocal
title LIMONS Anti-Drone Telemetry Defense System
color 0A

:: Ensure working directory is the script's root directory
cd /d "%~dp0"

echo ===============================================================================
echo            LIMONS ANTI-DRONE TELEMETRY DEFENSE SYSTEM
echo                    Autonomous Surveillance Suite
echo ===============================================================================
echo.

:: 1. Check if Python is installed and accessible in PATH
python --version >nul 2>&1
if errorlevel 1 goto python_missing
goto python_ok

:python_missing
color 0C
echo ===============================================================================
echo  [ERROR] Python is NOT installed or NOT added to PATH!
echo ===============================================================================
echo  Please follow these simple steps:
echo   1. Download Python 3.10 or 3.11 from https://www.python.org/downloads/
echo   2. Run the installer and CRITICALLY check the box:
echo      [X] Add python.exe to PATH
echo   3. Restart this script after installation.
echo ===============================================================================
echo.
pause
exit /b 1

:python_ok
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Detected %PY_VER%

:: 2. Check if required AI, Vision, and Dashboard libraries are installed
echo [*] Checking required dependencies...
python -c "import flask, cv2, yaml, numpy, ultralytics" >nul 2>&1
if not errorlevel 1 goto deps_ready

color 0E
echo.
echo ===============================================================================
echo  [FIRST TIME SETUP] Missing required libraries detected!
echo  The system is now automatically installing all dependencies.
echo  This only happens ONCE on first run. Please wait 2 to 3 minutes...
echo ===============================================================================
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

echo.
echo [OK] All dependencies installed successfully!
color 0A
goto deps_ready

:install_failed
color 0C
echo.
echo ===============================================================================
echo  [ERROR] Failed to install dependencies!
echo  Please verify your internet connection and try running this script again.
echo ===============================================================================
echo.
pause
exit /b 1

:deps_ready
echo [OK] All dependencies verified and ready.

:: 3. Launch Dashboard and auto-open Browser
echo.
echo ===============================================================================
echo  Starting Tactical Defense Dashboard at http://127.0.0.1:5000 ...
echo  Your default web browser will open automatically in 2 seconds.
echo  To shut down the system anytime, close this window or press CTRL+C.
echo ===============================================================================
echo.

start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://127.0.0.1:5000"
python dashboard/app.py

echo.
echo [NOTICE] Dashboard process stopped.
pause
