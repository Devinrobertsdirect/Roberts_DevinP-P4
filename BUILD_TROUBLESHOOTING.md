# Troubleshooting PyInstaller Build Issues

## Error: "failed to start embedded python interpreter"

This error usually indicates missing dependencies or issues with how PyInstaller bundled the application.

### Solution 1: Try Directory Build (for debugging)

Instead of `--onefile`, use a directory build to see what's missing:

```bash
pyinstaller --name=HandGestureControl --windowed --clean --noconfirm main.py
```

This creates a `dist/HandGestureControl/` folder. Run the `.exe` from there and check for missing DLL errors.

### Solution 2: Enable Console for Debugging

Modify the spec file to set `console=True` temporarily to see error messages:

```python
console=True,  # Enable console to see errors
```

Rebuild and run to see the actual error message.

### Solution 3: Check for Missing DLLs

Common missing DLLs:
- `python310.dll` (should be auto-included)
- OpenCV DLLs
- MediaPipe DLLs

### Solution 4: Try Different Build Options

If onefile fails, try:
```bash
pyinstaller --name=HandGestureControl --windowed --onedir main.py
```

This creates a folder with the executable and all dependencies.

### Solution 5: Verify Dependencies

Make sure all imports work in Python:
```bash
python test_imports.py
```

### Solution 6: Check Windows Defender/Antivirus

Sometimes antivirus software blocks PyInstaller executables. Try:
- Temporarily disable antivirus
- Add exception for the dist folder
- Use Windows Defender exclusion

### Solution 7: Use Virtual Environment

Build from a clean virtual environment with only required packages:
```bash
python -m venv build_env
build_env\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller ...
```

### Solution 8: Check Python Version

Ensure you're using a compatible Python version (3.7-3.11 recommended for PyInstaller).

If the issue persists, try building with console enabled first to see the actual error message.

