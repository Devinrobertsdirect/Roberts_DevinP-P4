@echo off
REM Build script for Windows executable
REM This creates a single .exe file that can run on any Windows machine

echo Installing PyInstaller if needed...
python -m pip install pyinstaller --quiet

echo.
echo Building Windows executable...
echo This may take 5-10 minutes, please wait...
echo.

pyinstaller --name=HandGestureControl --onefile --noconsole --clean --noconfirm ^
    --hidden-import=cv2 ^
    --hidden-import=mediapipe ^
    --hidden-import=mediapipe.python.solutions ^
    --hidden-import=mediapipe.python.solutions.hands ^
    --hidden-import=numpy ^
    --hidden-import=pandas ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn.neighbors ^
    --hidden-import=pyautogui ^
    --hidden-import=PIL ^
    --hidden-import=PIL.Image ^
    --hidden-import=PIL.ImageTk ^
    --hidden-import=tkinter ^
    --hidden-import=tkinter.ttk ^
    --collect-submodules=mediapipe ^
    --collect-submodules=cv2 ^
    main.py

echo.
echo ========================================
echo Build complete!
echo.
echo Your executable is located at:
echo dist\HandGestureControl.exe
echo.
echo You can distribute this file to any Windows computer.
echo ========================================
pause

