#!/bin/bash
# Build script for macOS executable
# This creates a .app bundle that can run on any Mac

echo "Installing PyInstaller if needed..."
pip3 install pyinstaller --quiet

echo ""
echo "Building macOS application..."
echo "This may take 5-10 minutes, please wait..."
echo ""

pyinstaller --name=HandGestureControl --onefile --windowed --clean --noconfirm \
    --hidden-import=cv2 \
    --hidden-import=mediapipe \
    --hidden-import=mediapipe.python.solutions \
    --hidden-import=mediapipe.python.solutions.hands \
    --hidden-import=numpy \
    --hidden-import=pandas \
    --hidden-import=sklearn \
    --hidden-import=sklearn.neighbors \
    --hidden-import=pyautogui \
    --hidden-import=PIL \
    --hidden-import=PIL.Image \
    --hidden-import=PIL.ImageTk \
    --hidden-import=tkinter \
    --hidden-import=tkinter.ttk \
    --collect-submodules=mediapipe \
    --collect-submodules=cv2 \
    main.py

echo ""
echo "========================================"
echo "Build complete!"
echo ""
echo "Your application is located at:"
echo "dist/HandGestureControl.app"
echo ""
echo "You can distribute this .app bundle to any Mac."
echo "========================================"

