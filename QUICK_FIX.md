# Quick Fix for "Failed to Start Embedded Python Interpreter"

## Immediate Solution: Build with Console Enabled

The fastest way to see what's wrong is to build with a console window that will show the actual error:

**Run this command:**
```bash
"C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe" -m PyInstaller --name=HandGestureControl --onefile --console --clean --noconfirm --collect-all=cv2 --collect-all=mediapipe main.py
```

This will create `HandGestureControl.exe` that shows a console window when you run it, displaying the actual error message.

## Alternative: Directory Build (More Reliable)

Instead of onefile, try a directory build which is often more reliable:

```bash
"C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe" -m PyInstaller --name=HandGestureControl --onedir --windowed --clean --noconfirm --collect-all=cv2 --collect-all=mediapipe main.py
```

This creates a `dist/HandGestureControl/` folder with the executable and all dependencies. The executable will be at `dist/HandGestureControl/HandGestureControl.exe`.

## Common Causes:

1. **Missing DLLs** - MediaPipe or OpenCV DLLs not bundled correctly
2. **UPX Compression** - Can corrupt executables (fixed in new spec file)
3. **Antivirus** - May block PyInstaller executables
4. **Missing hidden imports** - Added more in the fixed spec file

## Try the Console Build First!

The console build will tell us exactly what's wrong so we can fix it properly.

