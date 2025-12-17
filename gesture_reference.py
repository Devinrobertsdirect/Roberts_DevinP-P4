# gesture_reference.py

# Quick gesture reference popup for test mode and main UI

import tkinter as tk
from tkinter import ttk


class GestureReference:
    """
    High-tech gesture reference popup with quick access to gesture information.
    Ironman-inspired blue aesthetic with sleek design.
    """
    
    def __init__(self, parent):
        self.parent = parent
        self.popup = None
        
        # Gesture information
        self.gestures = {
            'Pinch': {
                'icon': '🤏',
                'description': 'Thumb and index together',
                'action': 'Left Click / Drag',
                'color': '#00D9FF'  # Cyan blue
            },
            'Middle Point': {
                'icon': '🖕',
                'description': 'Middle finger extended, others closed',
                'action': 'Right Click',
                'color': '#FF5252'  # Red for right-click
            },
            'Open Palm': {
                'icon': '✋',
                'description': 'All fingers extended',
                'action': 'Stop Drag',
                'color': '#4CAF50'  # Green
            },
            'Index Point': {
                'icon': '👆',
                'description': 'Index finger extended',
                'action': 'Move Mouse',
                'color': '#2196F3'  # Light blue
            },
            'Thumbs Up': {
                'icon': '👍',
                'description': 'Thumb extended up',
                'action': 'Symbol Gesture',
                'color': '#00BCD4'  # Cyan
            },
            'Peace Sign': {
                'icon': '✌️',
                'description': 'Index and middle extended',
                'action': 'Symbol Gesture',
                'color': '#03A9F4'  # Light blue
            },
            'OK Sign': {
                'icon': '👌',
                'description': 'Thumb and index circle',
                'action': 'Symbol Gesture',
                'color': '#0097A7'  # Teal
            },
            'Rock On': {
                'icon': '🤘',
                'description': 'Index and pinky extended',
                'action': 'Symbol Gesture',
                'color': '#00ACC1'  # Cyan
            }
        }
    
    def show_popup(self, x=None, y=None):
        """Show gesture reference popup"""
        if self.popup is not None:
            self.popup.destroy()
        
        self.popup = tk.Toplevel(self.parent)
        self.popup.title("Gesture Reference")
        self.popup.configure(bg='#0a1628')  # Dark navy blue
        
        # Make it semi-transparent and always on top
        self.popup.attributes('-topmost', True)
        self.popup.overrideredirect(False)
        
        # Position near cursor or center
        if x and y:
            self.popup.geometry(f"+{x}+{y}")
        else:
            self.popup.geometry("+100+100")
        
        # Header with glow effect
        header = tk.Frame(self.popup, bg='#0a1628', height=60)
        header.pack(fill=tk.X)
        
        title_label = tk.Label(header, 
                              text="⚡ GESTURE COMMANDS",
                              font=('Segoe UI', 16, 'bold'),
                              bg='#0a1628',
                              fg='#00D9FF',  # Bright cyan
                              anchor='center')
        title_label.pack(pady=15)
        
        # Close button
        close_btn = tk.Button(header, text="✕", 
                             font=('Segoe UI', 12, 'bold'),
                             bg='#1a1f3a',
                             fg='#ffffff',
                             activebackground='#2a2f4a',
                             borderwidth=0,
                             width=3,
                             command=self.close_popup)
        close_btn.place(relx=0.95, rely=0.5, anchor='center')
        
        # Content frame with scroll
        canvas_frame = tk.Frame(self.popup, bg='#0a1628')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        content_canvas = tk.Canvas(canvas_frame, 
                                   bg='#0a1628',
                                   yscrollcommand=scrollbar.set,
                                   highlightthickness=0)
        content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=content_canvas.yview)
        
        # Content container
        content = tk.Frame(content_canvas, bg='#0a1628')
        content_canvas.create_window((0, 0), window=content, anchor="nw")
        
        # Create gesture cards
        row = 0
        col = 0
        max_cols = 2
        
        for gesture_name, info in self.gestures.items():
            card = self.create_gesture_card(content, gesture_name, info)
            card.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(max_cols):
            content.grid_columnconfigure(i, weight=1)
        
        # Update scroll region
        content.update_idletasks()
        content_canvas.config(scrollregion=content_canvas.bbox("all"))
        
        # Bind mouse wheel
        def on_mousewheel(event):
            content_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        content_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Size constraints
        self.popup.geometry("600x500")
        self.popup.resizable(True, True)
        
        # Bind escape to close
        self.popup.bind("<Escape>", lambda e: self.close_popup())
        
        return self.popup
    
    def create_gesture_card(self, parent, name, info):
        """Create a sleek gesture card with high-tech styling"""
        # Main card frame
        card = tk.Frame(parent, 
                       bg='#1a1f3a',  # Dark blue-gray
                       relief=tk.FLAT,
                       bd=0)
        
        # Glow border effect (simulated with frame)
        border_frame = tk.Frame(card, bg=info['color'], height=3)
        border_frame.pack(fill=tk.X)
        
        # Content frame
        content_frame = tk.Frame(card, bg='#1a1f3a')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=8)
        
        # Icon and name
        header_frame = tk.Frame(content_frame, bg='#1a1f3a')
        header_frame.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        icon_label = tk.Label(header_frame,
                             text=info['icon'],
                             font=('Segoe UI', 24),
                             bg='#1a1f3a',
                             fg=info['color'])
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        name_label = tk.Label(header_frame,
                             text=name,
                             font=('Segoe UI', 14, 'bold'),
                             bg='#1a1f3a',
                             fg='#ffffff',
                             anchor='w')
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Description
        desc_label = tk.Label(content_frame,
                             text=info['description'],
                             font=('Segoe UI', 10),
                             bg='#1a1f3a',
                             fg='#B0BEC5',  # Light gray-blue
                             anchor='w',
                             justify=tk.LEFT)
        desc_label.pack(fill=tk.X, padx=15, pady=(0, 5))
        
        # Action with accent
        action_frame = tk.Frame(content_frame, bg='#1a1f3a')
        action_frame.pack(fill=tk.X, padx=15, pady=(5, 10))
        
        action_label = tk.Label(action_frame,
                               text=f"→ {info['action']}",
                               font=('Segoe UI', 11, 'bold'),
                               bg='#1a1f3a',
                               fg=info['color'],
                               anchor='w')
        action_label.pack(side=tk.LEFT)
        
        return card
    
    def close_popup(self):
        """Close the popup"""
        if self.popup:
            self.popup.destroy()
            self.popup = None

