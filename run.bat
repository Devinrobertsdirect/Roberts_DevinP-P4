@echo off
echo Starting Hand Gesture Control...
echo.

REM Try to find Python
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" main.py
    goto :end
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python main.py
    goto :end
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py main.py
    goto :end
)

echo ERROR: Could not find Python installation.
echo Please add Python to your PATH or edit this script with the correct path.
pause
exit /b 1

:end
pause

