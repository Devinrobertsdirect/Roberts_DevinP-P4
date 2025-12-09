# ui.py

# Modern Tkinter UI with improved styling and layout
# Shows live feed, gesture label, confidence, FPS, and debug log.

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time
import threading
import queue


class SimpleUI:
    def __init__(self, width=1000, height=700, title="Hand Gesture Control R&D"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg='#1e1e1e')  # Dark theme background
        self.width = width
        self.height = height
        
        # Configure modern styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), background='#1e1e1e', foreground='#ffffff')
        style.configure('Info.TLabel', font=('Segoe UI', 10), background='#2d2d2d', foreground='#ffffff')
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'), background='#2d2d2d', foreground='#4CAF50')
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar
        title_frame = tk.Frame(main_container, bg='#1e1e1e')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = tk.Label(title_frame, text="🖐️ Hand Gesture Recognition System", 
                               font=('Segoe UI', 18, 'bold'), 
                               bg='#1e1e1e', fg='#4CAF50')
        title_label.pack(side=tk.LEFT)
        
        # Video panel with border
        video_frame = tk.Frame(main_container, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, bg='#000000', text="Initializing camera...")
        self.video_label.pack(padx=5, pady=5)
        
        # Status panel with modern layout
        status_frame = tk.Frame(main_container, bg='#2d2d2d', relief=tk.FLAT)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create info boxes
        info_container = tk.Frame(status_frame, bg='#2d2d2d')
        info_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Gesture display (larger, prominent)
        gesture_box = tk.Frame(info_container, bg='#3d3d3d', relief=tk.RAISED, bd=1)
        gesture_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(gesture_box, text="GESTURE", font=('Segoe UI', 9, 'bold'), 
                bg='#3d3d3d', fg='#888888').pack(anchor='w', padx=10, pady=(10, 2))
        self.gesture_var = tk.StringVar(value="None Detected")
        self.gesture_label = tk.Label(gesture_box, textvariable=self.gesture_var, 
                                font=('Segoe UI', 14, 'bold'), 
                                bg='#3d3d3d', fg='#4CAF50', anchor='w')
        self.gesture_label.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Confidence display
        conf_box = tk.Frame(info_container, bg='#3d3d3d', relief=tk.RAISED, bd=1)
        conf_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(conf_box, text="CONFIDENCE", font=('Segoe UI', 9, 'bold'), 
                bg='#3d3d3d', fg='#888888').pack(anchor='w', padx=10, pady=(10, 2))
        self.conf_var = tk.StringVar(value="0%")
        self.conf_label = tk.Label(conf_box, textvariable=self.conf_var, 
                             font=('Segoe UI', 14, 'bold'), 
                             bg='#3d3d3d', fg='#FFC107', anchor='w')
        self.conf_label.pack(anchor='w', padx=10, pady=(0, 10))
        
        # FPS display
        fps_box = tk.Frame(info_container, bg='#3d3d3d', relief=tk.RAISED, bd=1)
        fps_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(fps_box, text="FPS", font=('Segoe UI', 9, 'bold'), 
                bg='#3d3d3d', fg='#888888').pack(anchor='w', padx=10, pady=(10, 2))
        self.fps_var = tk.StringVar(value="0.0")
        fps_label = tk.Label(fps_box, textvariable=self.fps_var, 
                            font=('Segoe UI', 14, 'bold'), 
                            bg='#3d3d3d', fg='#2196F3', anchor='w')
        fps_label.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Gesture reference panel
        ref_frame = tk.Frame(main_container, bg='#2d2d2d', relief=tk.FLAT)
        ref_frame.pack(fill=tk.X, pady=(0, 10))
        
        ref_title = tk.Label(ref_frame, text="📋 Available Gestures", 
                            font=('Segoe UI', 11, 'bold'), 
                            bg='#2d2d2d', fg='#ffffff', anchor='w')
        ref_title.pack(anchor='w', padx=10, pady=(10, 5))
        
        gesture_list = tk.Frame(ref_frame, bg='#2d2d2d')
        gesture_list.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        gestures_info = [
            "👌 OK Sign  |  👍 Thumbs Up  |  ✌️ Peace Sign  |  🤘 Rock On",
            "🤏 Pinch  |  ✋ Open Palm  |  ✊ Fist  |  👆 Point  |  1️⃣-5️⃣ Numbers"
        ]
        
        for info in gestures_info:
            tk.Label(gesture_list, text=info, font=('Segoe UI', 9), 
                    bg='#2d2d2d', fg='#cccccc', anchor='w').pack(anchor='w', pady=2)
        
        # Debug log with scrollbar
        log_frame = tk.Frame(main_container, bg='#2d2d2d', relief=tk.FLAT)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_title = tk.Label(log_frame, text="📝 System Log", 
                            font=('Segoe UI', 11, 'bold'), 
                            bg='#2d2d2d', fg='#ffffff', anchor='w')
        log_title.pack(anchor='w', padx=10, pady=(0, 5))
        
        log_container = tk.Frame(log_frame, bg='#1e1e1e')
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_box = tk.Text(log_container, height=6, state='disabled',
                              bg='#1e1e1e', fg='#00ff00',
                              font=('Consolas', 9),
                              yscrollcommand=scrollbar.set,
                              wrap=tk.WORD)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_box.yview)
        
        # Internal vars
        self.queue = queue.Queue()
        self.running = False

    def update_frame(self, cv2_bgr_image):
        # convert BGR -> RGB -> PIL -> ImageTk
        image_rgb = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_rgb)
        
        # Resize to fit video panel while maintaining aspect ratio
        video_panel_width = min(960, self.width - 40)
        video_panel_height = min(540, self.height - 350)
        
        aspect_ratio = im_pil.width / im_pil.height
        if aspect_ratio > video_panel_width / video_panel_height:
            new_width = video_panel_width
            new_height = int(video_panel_width / aspect_ratio)
        else:
            new_height = video_panel_height
            new_width = int(video_panel_height * aspect_ratio)
        
        im_pil = im_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")

    def set_gesture_text(self, gesture_text):
        # Format gesture text nicely
        if gesture_text == 'none' or gesture_text == 'unknown':
            display_text = "None Detected"
            color = '#888888'
        else:
            # Special handling for common gesture names
            gesture_display_map = {
                'open_hand': 'Open Palm',
                'open_palm': 'Open Palm',
                'index_point': 'Point',
                'pinch': 'Pinch',
                'fist': 'Fist',
                'thumbs_up': 'Thumbs Up',
                'peace_sign': 'Peace Sign',
                'ok_sign': 'OK Sign',
                'rock_on': 'Rock On'
            }
            
            if gesture_text in gesture_display_map:
                display_text = gesture_display_map[gesture_text]
            elif gesture_text.startswith('number_'):
                num = gesture_text.split('_')[1]
                display_text = f"Number {num}"
            else:
                display_text = gesture_text.replace('_', ' ').title()
            color = '#4CAF50'
        
        self.gesture_var.set(display_text)
        # Update color dynamically
        self.gesture_label.config(fg=color)

    def set_confidence(self, conf_text):
        try:
            conf_float = float(conf_text)
            conf_percent = f"{conf_float*100:.0f}%"
            
            # Color based on confidence
            if conf_float >= 0.7:
                color = '#4CAF50'  # Green
            elif conf_float >= 0.4:
                color = '#FFC107'  # Yellow
            else:
                color = '#F44336'  # Red
            
            self.conf_var.set(conf_percent)
            # Update color dynamically
            self.conf_label.config(fg=color)
        except:
            self.conf_var.set("0%")

    def set_fps(self, fps_text):
        self.fps_var.set(fps_text)

    def log(self, text):
        self.log_box.config(state='normal')
        timestamp = time.strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{timestamp}] {text}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')

    def start(self):
        self.running = True

    def run(self):
        """Run the Tkinter mainloop (call this from main thread)"""
        self.running = True
        self.root.mainloop()

    def stop(self):
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
