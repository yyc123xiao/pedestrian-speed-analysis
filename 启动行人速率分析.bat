@echo off
cd /d "%~dp0"

echo Person Speed Analyzer
echo.
echo Checking Python...
python --version
if errorlevel 1 (
    echo Python not found! Install Python 3.8+ first.
    pause
    exit
)

echo.
echo This script requires: ultralytics, opencv-python, matplotlib
echo Press ENTER to start...
pause >nul

echo Starting...
python video_speed_gui.py
echo.
echo Done.
pause
