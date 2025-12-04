# VS Code Setup Guide for Hand Gesture Control

This guide helps you configure Visual Studio Code to properly recognize `cv2` and `numpy` imports.

## ✅ Quick Fix - Select Python Interpreter

1. **Open VS Code** in this project folder
2. **Press `Ctrl+Shift+P`** (or `Cmd+Shift+P` on Mac)
3. **Type**: `Python: Select Interpreter`
4. **Choose one of these**:

   **Option 1: System Python (Recommended - packages already installed)**
   ```
   C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe
   ```

   **Option 2: Virtual Environment (packages also installed here)**
   ```
   .\env\Scripts\python.exe
   ```

5. **Reload VS Code window** if needed (Ctrl+Shift+P → "Developer: Reload Window")

## ✅ Verify Installation

Run the test script to verify imports:

1. Open `test_imports.py` in VS Code
2. Press `F5` or click the Run button
3. All imports should show ✓ checkmarks

Or run in terminal:
```bash
python test_imports.py
```

## ✅ VS Code Configuration Files Created

The following configuration files have been created for you:

- **`.vscode/settings.json`** - Sets default Python interpreter path
- **`.vscode/launch.json`** - Debug configurations for running Python files

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution 1**: Check which Python interpreter VS Code is using
- Look at the bottom-right corner of VS Code (should show Python version)
- Click it to change interpreter
- Select the system Python path above

**Solution 2**: Verify packages are installed
```bash
python -c "import cv2; print('cv2 works!')"
python -c "import numpy; print('numpy works!')"
```

**Solution 3**: Install packages if missing
```bash
python -m pip install opencv-python numpy
```

### Issue: Red squiggly lines under imports

1. Wait a few seconds - VS Code might still be analyzing
2. Check Python interpreter is selected correctly
3. Reload VS Code window (Ctrl+Shift+P → "Developer: Reload Window")

### Issue: Packages work in terminal but not in VS Code

VS Code might be using a different Python than your terminal. Fix:
1. Select interpreter manually (Ctrl+Shift+P → "Python: Select Interpreter")
2. Use the exact path shown above

## 📝 Current Configuration

- **Default Interpreter**: System Python 3.10
- **Location**: `C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe`
- **Packages Installed**: ✅ All required packages are installed in both system Python and virtual environment

## 🚀 Next Steps

Once imports are working:
1. Open `main.py`
2. Press `F5` to run with debugging
3. Or use the Run button in VS Code

The application should start and show the camera feed!


