"""
Quick test script to verify cv2 and numpy imports work correctly.
Run this in VS Code to test your Python environment configuration.
"""

print("Testing imports...")
print("-" * 50)

try:
    import cv2
    print(f"✓ cv2 imported successfully (version: {cv2.__version__})")
except ImportError as e:
    print(f"✗ Failed to import cv2: {e}")

try:
    import numpy as np
    print(f"✓ numpy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")

try:
    import mediapipe as mp
    print(f"✓ mediapipe imported successfully")
except ImportError as e:
    print(f"✗ Failed to import mediapipe: {e}")

try:
    import pandas as pd
    print(f"✓ pandas imported successfully (version: {pd.__version__})")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Failed to import scikit-learn: {e}")

try:
    import pyautogui
    print(f"✓ pyautogui imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pyautogui: {e}")

try:
    from PIL import Image
    print(f"✓ pillow (PIL) imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pillow: {e}")

print("-" * 50)
print("\nImport test complete!")
print("\nIf all imports succeeded, your environment is configured correctly.")
print("If you see any ✗ errors, check your Python interpreter in VS Code:")
print("  - Press Ctrl+Shift+P")
print("  - Type 'Python: Select Interpreter'")
print("  - Choose: C:\\Users\\devin\\AppData\\Local\\Programs\\Python\\Python310\\python.exe")



