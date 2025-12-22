@echo off
REM Build script with console enabled for debugging
REM This will show error messages if the app fails to start

echo Building with console enabled for debugging...
echo.

"C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe" -m PyInstaller ^
    --name=HandGestureControl ^
    --onefile ^
    --console ^
    --clean ^
    --noconfirm ^
    --hidden-import=cv2 ^
    --hidden-import=cv2.cv2 ^
    --hidden-import=mediapipe ^
    --hidden-import=mediapipe.python.solutions.hands ^
    --hidden-import=numpy ^
    --hidden-import=pyautogui ^
    --hidden-import=PIL ^
    --hidden-import=PIL.Image ^
    --hidden-import=PIL.ImageTk ^
    --hidden-import=tkinter ^
    --collect-all=cv2 ^
    --collect-all=mediapipe ^
    main.py

echo.
echo Build complete. Check dist\HandGestureControl.exe
echo Console window will show if there are any startup errors.
pause

