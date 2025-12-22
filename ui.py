# ui.py

# High-tech Ironman-inspired UI with dark blue aesthetic
# Production-ready design with sleek styling

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time
import threading
import queue


class SimpleUI:
    def __init__(self, width=1000, height=700, title="ACERA Hand Gesture Control"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg='#0a1628')  # Dark navy blue - ironman aesthetic
        self.width = width
        self.height = height
        
        # High-tech color palette
        self.colors = {
            'bg_dark': '#0a1628',      # Deep navy
            'bg_medium': '#1a1f3a',    # Dark blue-gray
            'bg_light': '#2a2f4a',     # Lighter blue-gray
            'accent_cyan': '#00D9FF',  # Bright cyan (ironman arc)
            'accent_blue': '#0066FF',  # Electric blue
            'accent_light': '#4A9EFF', # Light blue
            'text_primary': '#ffffff',
            'text_secondary': '#B0BEC5',  # Light gray-blue
            'text_muted': '#64748B',      # Muted gray
            'success': '#00E676',         # Green-cyan
            'warning': '#FFB300',         # Amber
            'error': '#FF5252'            # Red
        }
        
        # Configure modern styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title bar with high-tech styling
        title_frame = tk.Frame(main_container, bg=self.colors['bg_dark'], height=70)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title with glow effect
        title_container = tk.Frame(title_frame, bg=self.colors['bg_dark'])
        title_container.pack(side=tk.LEFT)
        
        title_label = tk.Label(title_container, 
                              text="⚡ ACERA",
                              font=('Segoe UI', 24, 'bold'),
                              bg=self.colors['bg_dark'],
                              fg=self.colors['accent_cyan'])
        title_label.pack(side=tk.LEFT, padx=(0, 10))
        
        subtitle_label = tk.Label(title_container,
                                 text="Gesture Control",
                                 font=('Segoe UI', 12),
                                 bg=self.colors['bg_dark'],
                                 fg=self.colors['text_secondary'])
        subtitle_label.pack(side=tk.LEFT)
        
        # Action buttons in title bar
        button_frame = tk.Frame(title_frame, bg=self.colors['bg_dark'])
        button_frame.pack(side=tk.RIGHT, padx=10)
        
        # Gesture reference button
        self.gesture_ref_button = tk.Button(button_frame,
                                           text="📋 Gestures",
                                           font=('Segoe UI', 11, 'bold'),
                                           bg=self.colors['bg_medium'],
                                           fg=self.colors['accent_cyan'],
                                           activebackground=self.colors['bg_light'],
                                           activeforeground=self.colors['accent_cyan'],
                                           borderwidth=0,
                                           padx=15,
                                           pady=8,
                                           command=self.on_gesture_ref_click)
        self.gesture_ref_button.pack(side=tk.LEFT, padx=5)
        
        # Test mode button
        self.test_mode_button = tk.Button(button_frame,
                                         text="🧪 Test Mode",
                                         font=('Segoe UI', 11, 'bold'),
                                         bg=self.colors['accent_blue'],
                                         fg='#ffffff',
                                         activebackground=self.colors['accent_light'],
                                         borderwidth=0,
                                         padx=15,
                                         pady=8,
                                         command=self.on_test_mode_click)
        self.test_mode_button.pack(side=tk.LEFT, padx=5)
        
        self.test_mode_handler = None
        self.gesture_ref_handler = None
        
        # Video panel with sleek border
        video_container = tk.Frame(main_container, bg=self.colors['bg_medium'])
        video_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Border effect
        border_frame = tk.Frame(video_container, bg=self.colors['accent_cyan'], height=2)
        border_frame.pack(fill=tk.X)
        
        video_frame = tk.Frame(video_container, bg='#000000')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.video_label = tk.Label(video_frame, bg='#000000', text="Initializing camera...",
                                    fg=self.colors['accent_cyan'],
                                    font=('Segoe UI', 12))
        self.video_label.pack(expand=True)
        
        # Status panel with high-tech cards
        status_frame = tk.Frame(main_container, bg=self.colors['bg_dark'])
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_container = tk.Frame(status_frame, bg=self.colors['bg_dark'])
        info_container.pack(fill=tk.X, padx=0, pady=0)
        
        # Create sleek info cards
        self.gesture_var = tk.StringVar(value="None Detected")
        gesture_card = self.create_info_card(info_container, "GESTURE", self.gesture_var, 
                                            self.colors['accent_cyan'], 0)
        
        self.conf_var = tk.StringVar(value="0%")
        conf_card = self.create_info_card(info_container, "CONFIDENCE", self.conf_var,
                                         self.colors['success'], 1)
        
        self.fps_var = tk.StringVar(value="0.0")
        fps_card = self.create_info_card(info_container, "FPS", self.fps_var,
                                        self.colors['accent_light'], 2)
        
        # Store label references for dynamic color updates
        self.gesture_label = None
        self.conf_label = None
        
        # Debug log with high-tech styling
        log_frame = tk.Frame(main_container, bg=self.colors['bg_dark'])
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_header = tk.Frame(log_frame, bg=self.colors['bg_medium'], height=35)
        log_header.pack(fill=tk.X)
        
        log_title = tk.Label(log_header,
                            text="⚡ SYSTEM LOG",
                            font=('Segoe UI', 11, 'bold'),
                            bg=self.colors['bg_medium'],
                            fg=self.colors['accent_cyan'],
                            anchor='w')
        log_title.pack(side=tk.LEFT, padx=15, pady=8)
        
        log_container = tk.Frame(log_frame, bg=self.colors['bg_medium'])
        log_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_box = tk.Text(log_container,
                              height=6,
                              state='disabled',
                              bg='#0a1628',
                              fg=self.colors['accent_cyan'],
                              font=('Consolas', 9),
                              yscrollcommand=scrollbar.set,
                              wrap=tk.WORD,
                              insertbackground=self.colors['accent_cyan'],
                              selectbackground=self.colors['bg_light'],
                              selectforeground=self.colors['text_primary'])
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar.config(command=self.log_box.yview)
        
        # Internal vars
        self.queue = queue.Queue()
        self.running = False
    
    def create_info_card(self, parent, label_text, value_var, accent_color, index):
        """Create a sleek high-tech info card"""
        card = tk.Frame(parent, bg=self.colors['bg_medium'], relief=tk.FLAT)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)
        
        # Accent border
        border = tk.Frame(card, bg=accent_color, height=3)
        border.pack(fill=tk.X)
        
        content = tk.Frame(card, bg=self.colors['bg_medium'])
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # Label
        label = tk.Label(content,
                        text=label_text,
                        font=('Segoe UI', 9, 'bold'),
                        bg=self.colors['bg_medium'],
                        fg=self.colors['text_muted'],
                        anchor='w')
        label.pack(anchor='w', pady=(0, 8))
        
        # Value
        value_label = tk.Label(content,
                              textvariable=value_var,
                              font=('Segoe UI', 16, 'bold'),
                              bg=self.colors['bg_medium'],
                              fg=accent_color,
                              anchor='w')
        value_label.pack(anchor='w')
        
        # Store references for gesture and confidence
        if label_text == "GESTURE":
            self.gesture_label = value_label
        elif label_text == "CONFIDENCE":
            self.conf_label = value_label
        
        return card
    
    def set_test_mode_handler(self, handler):
        """Set the test mode handler callback"""
        self.test_mode_handler = handler
    
    def set_gesture_ref_handler(self, handler):
        """Set the gesture reference handler callback"""
        self.gesture_ref_handler = handler
    
    def on_test_mode_click(self):
        """Handle test mode button click"""
        if self.test_mode_handler:
            self.test_mode_handler()
    
    def on_gesture_ref_click(self):
        """Handle gesture reference button click"""
        if self.gesture_ref_handler:
            self.gesture_ref_handler()

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
            color = self.colors['text_muted']
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
            color = self.colors['accent_cyan']
        
        self.gesture_var.set(display_text)
        # Update color dynamically
        if self.gesture_label:
            self.gesture_label.config(fg=color)

    def set_confidence(self, conf_text):
        try:
            conf_float = float(conf_text)
            conf_percent = f"{conf_float*100:.0f}%"
            
            # Color based on confidence
            if conf_float >= 0.7:
                color = self.colors['success']  # Green-cyan
            elif conf_float >= 0.4:
                color = self.colors['warning']  # Amber
            else:
                color = self.colors['error']  # Red
            
            self.conf_var.set(conf_percent)
            # Update color dynamically
            if self.conf_label:
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
