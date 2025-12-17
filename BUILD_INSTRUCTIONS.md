# Building Executable from Python Application

This guide will help you create standalone executables (.exe for Windows, .app for macOS) that can run without Python installed.

## Prerequisites

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Ensure all dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Build (Recommended)

Simply run the build script:
```bash
python build_executable.py
```

The executable will be created in the `dist/` folder.

## Manual Build (Windows)

1. **Open Command Prompt or PowerShell in the project folder**

2. **Run PyInstaller:**
   ```bash
   pyinstaller --name=HandGestureControl --onefile --noconsole --clean --noconfirm ^
       --hidden-import=cv2 --hidden-import=mediapipe --hidden-import=numpy ^
       --hidden-import=pyautogui --hidden-import=PIL --hidden-import=tkinter ^
       --collect-submodules=mediapipe --collect-submodules=cv2 main.py
   ```

3. **Find your executable:**
   - Location: `dist/HandGestureControl.exe`
   - You can distribute this single .exe file

## Manual Build (macOS)

1. **Open Terminal in the project folder**

2. **Run PyInstaller:**
   ```bash
   pyinstaller --name=HandGestureControl --onefile --windowed --clean --noconfirm \
       --hidden-import=cv2 --hidden-import=mediapipe --hidden-import=numpy \
       --hidden-import=pyautogui --hidden-import=PIL --hidden-import=tkinter \
       --collect-submodules=mediapipe --collect-submodules=cv2 main.py
   ```

3. **Find your application:**
   - Location: `dist/HandGestureControl.app`
   - This is a macOS application bundle you can distribute

## Build Options Explained

- `--onefile`: Creates a single executable file (easier to distribute)
- `--windowed` / `--noconsole`: Hides the console window
- `--clean`: Cleans PyInstaller cache before building
- `--noconfirm`: Overwrites existing build without asking
- `--hidden-import`: Includes packages PyInstaller might miss
- `--collect-submodules`: Ensures all submodules are included

## Alternative: Directory Build (for debugging)

If you encounter issues with the one-file build, try a directory build:

```bash
# Remove --onefile flag
pyinstaller --name=HandGestureControl --windowed main.py
```

This creates a folder with the executable and all dependencies. Useful for debugging missing files.

## Troubleshooting

### "Module not found" errors

If you get import errors, add more hidden imports:
```bash
pyinstaller ... --hidden-import=MODULE_NAME ...
```

### Large file size

The executable will be large (100-300 MB) because it bundles:
- Python interpreter
- All dependencies (OpenCV, MediaPipe, NumPy, etc.)
- MediaPipe models

This is normal for applications with computer vision libraries.

### Camera access issues

- **Windows**: Should work automatically
- **macOS**: Users may need to grant camera permissions in System Preferences > Security & Privacy
- Make sure to test camera access after building

### Performance

The first startup may be slower as the executable extracts temporary files. Subsequent runs are faster.

## Distributing Your Application

1. **Windows:**
   - Distribute: `dist/HandGestureControl.exe`
   - Users can double-click to run
   - No Python installation needed

2. **macOS:**
   - Distribute: `dist/HandGestureControl.app`
   - Users may need to right-click and select "Open" the first time (Gatekeeper)
   - Alternatively, create a .dmg file for easier distribution

3. **Requirements for users:**
   - Webcam
   - Windows 10+ or macOS 10.13+
   - No Python installation needed!

## Testing the Build

After building, test the executable:
1. Move it to a different folder (not where you built it)
2. Run it from there
3. Verify camera access works
4. Test all gestures and features

## Advanced: Custom Icon

To add a custom icon:

1. **Windows:** Create `icon.ico` file
2. **macOS:** Create `icon.icns` file

Then add to build command:
```bash
--icon=icon.ico  # Windows
--icon=icon.icns  # macOS
```

## Notes

- First build takes 5-10 minutes
- Executable size: ~200-300 MB (normal for apps with OpenCV/MediaPipe)
- MediaPipe models are automatically bundled
- Test thoroughly before distributing to users

