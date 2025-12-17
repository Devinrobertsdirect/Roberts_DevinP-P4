#!/usr/bin/env python3
"""
Build script for creating executable from Python application.
Supports both Windows (.exe) and macOS (.app) builds.

Usage:
    python build_executable.py
"""

import PyInstaller.__main__
import sys
import os
import platform

def build_executable():
    """Build executable using PyInstaller"""
    
    # Get platform-specific settings
    system = platform.system()
    
    # Common PyInstaller arguments
    args = [
        'main.py',
        '--name=HandGestureControl',
        '--onefile',  # Single executable file
        '--windowed',  # No console window (use --noconsole for Windows, --windowed for macOS)
        '--clean',  # Clean PyInstaller cache
        '--noconfirm',  # Overwrite output without asking
    ]
    
    # Platform-specific adjustments
    if system == 'Windows':
        args.append('--noconsole')  # Windows: hide console
        # Add icon if you have one (optional)
        # args.append('--icon=icon.ico')
    elif system == 'Darwin':  # macOS
        args.append('--windowed')  # macOS: creates .app bundle
        # Add icon if you have one (optional)
        # args.append('--icon=icon.icns')
    
    # Add hidden imports (packages that PyInstaller might miss)
    hidden_imports = [
        'cv2',
        'mediapipe',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.hands',
        'mediapipe.framework.formats',
        'numpy',
        'pandas',
        'sklearn',
        'sklearn.neighbors',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'pyautogui',
        'tkinter',
        'tkinter.ttk',
        'queue',
        'threading',
    ]
    
    for imp in hidden_imports:
        args.append(f'--hidden-import={imp}')
    
    # Add data files if needed (MediaPipe models are usually bundled, but include just in case)
    # Uncomment if you need to include specific data files:
    # args.append('--add-data=path/to/data;data')  # Windows format
    # args.append('--add-data=path/to/data:data')  # macOS/Linux format
    
    # Collect submodules
    args.append('--collect-submodules=mediapipe')
    args.append('--collect-submodules=cv2')
    
    print(f"Building executable for {system}...")
    print(f"Arguments: {' '.join(args)}")
    print("\nThis may take several minutes. Please wait...\n")
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("\n" + "="*50)
    if system == 'Windows':
        print("Build complete! Executable should be in: dist/HandGestureControl.exe")
    elif system == 'Darwin':
        print("Build complete! Application should be in: dist/HandGestureControl.app")
    else:
        print("Build complete! Executable should be in: dist/HandGestureControl")
    print("="*50)

if __name__ == '__main__':
    try:
        import PyInstaller
    except ImportError:
        print("ERROR: PyInstaller is not installed!")
        print("Please install it first:")
        print("  pip install pyinstaller")
        sys.exit(1)
    
    build_executable()

