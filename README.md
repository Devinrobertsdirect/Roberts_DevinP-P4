# Hand Gesture Control R&D Prototype

A complete Python prototype demonstrating hand gesture recognition and OS control using MediaPipe Hands, OpenCV, and pyautogui.

## Features

- **Real-time hand tracking** using MediaPipe Hands with 21-landmark detection
- **Heuristic gesture recognition** (pinch, fist, open hand, index pointing)
- **OS control** - mouse movement and clicks via pyautogui
- **Live UI** - Tkinter interface showing camera feed, gesture labels, confidence, and FPS
- **Data logging** - CSV logger for collecting labeled gesture samples
- **ML training stub** - KNN classifier for future machine learning integration

## Installation

1. **Install Python dependencies:**

   **Option A: Use the installation script (Windows)**
   ```bash
   install_dependencies.bat
   ```

   **Option B: Manual installation**
   
   First, find your Python executable. Common locations on Windows:
   - `C:\Users\YourUsername\AppData\Local\Programs\Python\Python310\python.exe`
   - Or add Python to your PATH and use:
   
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   
   Or install individually:
   ```bash
   python -m pip install opencv-python mediapipe numpy pandas scikit-learn pyautogui pillow
   ```
   
   **Option C: If Python is not in PATH**
   
   Use the full path to Python. For example:
   ```bash
   "C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
   ```

2. **Configure Visual Studio Code:**
   
   See `VSCODE_SETUP.md` for detailed instructions, or quickly:
   - Press `Ctrl+Shift+P` → Type "Python: Select Interpreter"
   - Choose: `C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe`
   - Test imports by running `test_imports.py`

3. **Note on permissions:**
   - **macOS**: Grant camera and accessibility permissions in System Preferences
   - **Windows**: pyautogui should work without additional setup
   - **Linux**: May require additional permissions for mouse/keyboard control

## Usage

**Option A: Use the run script (Windows)**
```bash
run.bat
```

**Option B: Run directly with Python**
```bash
python main.py
```

Or if Python is not in PATH:
```bash
"C:\Users\devin\AppData\Local\Programs\Python\Python310\python.exe" main.py
```

Press `q` or close the window to quit.

## Module Overview

- **camera.py** - Webcam capture wrapper using OpenCV
- **tracker.py** - MediaPipe Hands wrapper producing 21-landmark arrays
- **gestures.py** - Feature extraction, heuristic recognizers, data logger, and ML training stub
- **controller.py** - Maps gestures to OS actions (mouse/keyboard) via pyautogui
- **ui.py** - Minimal Tkinter UI with live camera, landmark overlay, gesture display, and debug log
- **main.py** - Orchestrates everything and runs the application

## Gesture Mappings

- **Index pointing** (open hand with extended index) → Move mouse cursor
- **Pinch** (thumb + index close) → Left-click and drag
- **Fist** → Right-click
- **Open hand** → Stop dragging

## Testing Individual Modules

Each module can be tested independently:

```bash
python camera.py      # Test webcam capture
python tracker.py     # Test hand tracking
python gestures.py    # Test gesture heuristics
python controller.py  # Test mouse control
```

## Data Collection

The `DataLogger` class in `gestures.py` can be used to collect labeled samples for training. Samples are saved to `gesture_samples.csv`.

To train a KNN model from collected samples:
```python
from gestures import Trainer
trainer = Trainer()
trainer.train_knn(n_neighbors=3)
```

## Requirements

- Python 3.7+
- Webcam
- Windows/macOS/Linux

## Notes

- This is an R&D prototype with verbose comments for educational purposes
- Thresholds in gesture detection may need tuning for different camera resolutions
- pyautogui fail-safe is disabled by default (can be re-enabled in `controller.py`)

