# test_mode.py

# Testing interface for gesture control with tasks and survey
# High-tech design with gesture reference popup
# Provides visual feedback and tracks user performance

import tkinter as tk
from tkinter import ttk
import time
import json
import os
from gesture_reference import GestureReference


class TestMode:
    """
    Test mode interface with 5 tasks and 5 questions for ACERA project.
    Tests gesture control capabilities and collects telemetry data.
    """
    
    def __init__(self, root, on_task_complete_callback=None, on_test_complete_callback=None):
        self.root = root
        self.on_task_complete = on_task_complete_callback
        self.on_test_complete = on_test_complete_callback
        
        self.current_task = 0
        self.task_start_time = None
        self.task_results = []
        self.survey_answers = {}
        
        self.test_window = None
        self.is_active = False
        
    def show_test_window(self):
        """Show the test mode window"""
        if self.test_window is not None:
            self.test_window.destroy()
        
        self.test_window = tk.Toplevel(self.root)
        self.test_window.title("ACERA Gesture Control Test")
        self.test_window.geometry("900x700")
        self.test_window.configure(bg='#0a1628')  # High-tech dark blue
        self.test_window.protocol("WM_DELETE_WINDOW", self.close_test)
        
        # Initialize gesture reference
        self.gesture_ref = GestureReference(self.test_window)
        
        self.is_active = True
        self.current_task = 0
        self.task_results = []
        self.survey_answers = {}
        
        self.show_task_screen()
    
    def close_test(self):
        """Close test mode"""
        self.is_active = False
        if self.test_window:
            self.test_window.destroy()
            self.test_window = None
    
    def show_task_screen(self):
        """Show current task screen"""
        if not self.test_window:
            return
        
        # Clear window
        for widget in self.test_window.winfo_children():
            widget.destroy()
        
        if self.current_task < len(self.get_tasks()):
            task = self.get_tasks()[self.current_task]
            self.task_start_time = time.time()
            self.render_task(task)
        else:
            self.show_survey()
    
    def get_tasks(self):
        """Return list of 5 test tasks"""
        return [
            {
                'id': 1,
                'title': 'Task 1: Move Mouse',
                'description': 'Move your index finger to move the mouse cursor. Try to hover over the green button below.',
                'instruction': 'Extend your index finger and move it to control the mouse',
                'target_type': 'hover',
                'target_id': 'task1_button'
            },
            {
                'id': 2,
                'title': 'Task 2: Single Click',
                'description': 'Perform a pinch gesture (thumb and index together) to click the button below.',
                'instruction': 'Bring thumb and index finger together to click',
                'target_type': 'click',
                'target_id': 'task2_button'
            },
            {
                'id': 3,
                'title': 'Task 3: Right Click',
                'description': 'Extend only your middle finger (others closed) to perform a right-click on the button below.',
                'instruction': 'Point with middle finger to right-click',
                'target_type': 'right_click',
                'target_id': 'task3_button'
            },
            {
                'id': 4,
                'title': 'Task 4: Drag and Drop',
                'description': 'Use pinch to click and hold, then drag the circle to the target area.',
                'instruction': 'Pinch to click, move while holding, release pinch to drop',
                'target_type': 'drag',
                'target_id': 'task4_drag'
            },
            {
                'id': 5,
                'title': 'Task 5: Multiple Clicks',
                'description': 'Click all three buttons below using pinch gestures.',
                'instruction': 'Click each button in sequence using pinch gestures',
                'target_type': 'multi_click',
                'target_id': 'task5_buttons'
            }
        ]
    
    def render_task(self, task):
        """Render task interface with high-tech styling"""
        # Header with accent border
        header_container = tk.Frame(self.test_window, bg='#1a1f3a')
        header_container.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        border_frame = tk.Frame(header_container, bg='#00D9FF', height=3)
        border_frame.pack(fill=tk.X)
        
        header = tk.Frame(header_container, bg='#1a1f3a', height=80)
        header.pack(fill=tk.X)
        
        title_label = tk.Label(header, text=task['title'], 
                              font=('Segoe UI', 20, 'bold'),
                              bg='#1a1f3a', fg='#00D9FF')
        title_label.pack(pady=(15, 5))
        
        desc_label = tk.Label(header, text=task['description'],
                             font=('Segoe UI', 11),
                             bg='#1a1f3a', fg='#B0BEC5', wraplength=850)
        desc_label.pack()
        
        # Instruction box with high-tech styling
        inst_container = tk.Frame(self.test_window, bg='#0a1628')
        inst_container.pack(fill=tk.X, padx=15, pady=5)
        
        inst_frame = tk.Frame(inst_container, bg='#1a1f3a', relief=tk.FLAT)
        inst_frame.pack(fill=tk.X)
        
        inst_accent = tk.Frame(inst_frame, bg='#4A9EFF', width=5)
        inst_accent.pack(side=tk.LEFT, fill=tk.Y)
        
        inst_label = tk.Label(inst_frame, text=f"⚡ {task['instruction']}",
                             font=('Segoe UI', 12, 'bold'),
                             bg='#1a1f3a', fg='#4A9EFF', anchor='w')
        inst_label.pack(padx=15, pady=12, anchor='w', fill=tk.X)
        
        # Task area with high-tech styling
        task_area = tk.Frame(self.test_window, bg='#0a1628')
        task_area.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Render task-specific interface
        if task['id'] == 1:
            self.render_task1(task_area, task)
        elif task['id'] == 2:
            self.render_task2(task_area, task)
        elif task['id'] == 3:
            self.render_task3(task_area, task)
        elif task['id'] == 4:
            self.render_task4(task_area, task)
        elif task['id'] == 5:
            self.render_task5(task_area, task)
        
        # Progress and controls with high-tech styling
        footer = tk.Frame(self.test_window, bg='#1a1f3a', height=60)
        footer.pack(fill=tk.X, padx=15, pady=10)
        
        left_frame = tk.Frame(footer, bg='#1a1f3a')
        left_frame.pack(side=tk.LEFT, padx=15)
        
        progress_text = f"TASK {task['id']} / 5"
        progress_label = tk.Label(left_frame, text=progress_text,
                                 font=('Segoe UI', 11, 'bold'),
                                 bg='#1a1f3a', fg='#00D9FF')
        progress_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Gesture reference quick button
        if self.gesture_ref:
            gesture_btn = tk.Button(left_frame, text="📋 Gesture List",
                                   command=lambda: self.gesture_ref.show_popup(),
                                   font=('Segoe UI', 10, 'bold'),
                                   bg='#2a2f4a', fg='#00D9FF',
                                   activebackground='#3a3f5a',
                                   activeforeground='#00D9FF',
                                   borderwidth=0,
                                   padx=12,
                                   pady=6)
            gesture_btn.pack(side=tk.LEFT)
        
        skip_btn = tk.Button(footer, text="Skip Task", 
                           command=self.skip_task,
                           font=('Segoe UI', 10, 'bold'),
                           bg='#64748B', fg='#ffffff',
                           activebackground='#74788B',
                           borderwidth=0,
                           padx=15,
                           pady=8)
        skip_btn.pack(side=tk.RIGHT, padx=15)
    
    def render_task1(self, parent, task):
        """Task 1: Move mouse"""
        label = tk.Label(parent, text="Move your index finger to hover over the button:",
                        font=('Segoe UI', 14), bg='#0a1628', fg='#B0BEC5')
        label.pack(pady=50)
        
        button_frame = tk.Frame(parent, bg='#0a1628')
        button_frame.pack(expand=True)
        
        self.task1_button = tk.Button(button_frame, text="HOVER HERE",
                                      font=('Segoe UI', 16, 'bold'),
                                      bg='#1a1f3a', fg='#00D9FF',
                                      activebackground='#2a2f4a',
                                      activeforeground='#00D9FF',
                                      width=20, height=3,
                                      relief=tk.FLAT,
                                      borderwidth=2,
                                      highlightthickness=2,
                                      highlightbackground='#00D9FF')
        self.task1_button.pack(pady=20)
        
        # Track hover via button binding (mouse cursor controlled by gesture)
        self.task1_hovered = False
        self.task1_button.bind("<Enter>", lambda e: self.on_task1_hover())
    
    def on_task1_hover(self):
        """Called when mouse cursor (controlled by gesture) enters task1 button"""
        if not self.task1_hovered and self.current_task == 0 and self.is_active:
            self.task1_hovered = True
            self.task1_button.config(bg='#00D9FF', fg='#0a1628', text="✓ HOVERED!",
                                    highlightbackground='#00D9FF')
            # Wait a moment to show the success, then complete
            self.test_window.after(1000, lambda: self.complete_task(True))
    
    def render_task2(self, parent, task):
        """Task 2: Single click"""
        label = tk.Label(parent, text="Use pinch gesture to click the button:",
                        font=('Segoe UI', 14), bg='#0a1628', fg='#B0BEC5')
        label.pack(pady=50)
        
        button_frame = tk.Frame(parent, bg='#0a1628')
        button_frame.pack(expand=True)
        
        self.task2_button = tk.Button(button_frame, text="CLICK ME",
                                      font=('Segoe UI', 16, 'bold'),
                                      bg='#0066FF', fg='#ffffff',
                                      activebackground='#4A9EFF',
                                      width=20, height=3,
                                      relief=tk.FLAT,
                                      borderwidth=0,
                                      command=lambda: self.complete_task(True))
        self.task2_button.pack(pady=20)
        
        self.task2_clicked = False
    
    def render_task3(self, parent, task):
        """Task 3: Right click"""
        label = tk.Label(parent, text="Extend middle finger (others closed) to right-click:",
                        font=('Segoe UI', 14), bg='#0a1628', fg='#B0BEC5')
        label.pack(pady=50)
        
        button_frame = tk.Frame(parent, bg='#0a1628')
        button_frame.pack(expand=True)
        
        self.task3_button = tk.Button(button_frame, text="RIGHT CLICK ME",
                                      font=('Segoe UI', 16, 'bold'),
                                      bg='#FF5252', fg='#ffffff',
                                      activebackground='#FF7575',
                                      width=20, height=3,
                                      relief=tk.FLAT,
                                      borderwidth=0)
        self.task3_button.pack(pady=20)
        
        # Bind right click
        self.task3_button.bind("<Button-3>", lambda e: self.complete_task(True))
        self.task3_clicked = False
    
    def render_task4(self, parent, task):
        """Task 4: Drag and drop"""
        canvas_frame = tk.Frame(parent, bg='#0a1628')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        label = tk.Label(canvas_frame, text="Drag the circle to the target area using pinch:",
                        font=('Segoe UI', 14), bg='#0a1628', fg='#B0BEC5')
        label.pack()
        
        self.task4_canvas = tk.Canvas(canvas_frame, bg='#1a1f3a', width=800, height=400, 
                                      highlightthickness=2, highlightbackground='#00D9FF')
        self.task4_canvas.pack(pady=20)
        
        # Draw drag source and target with high-tech colors
        self.task4_circle = self.task4_canvas.create_oval(50, 150, 150, 250, 
                                                          fill='#0066FF', outline='#00D9FF', width=3)
        self.task4_target = self.task4_canvas.create_rectangle(600, 100, 750, 300,
                                                               fill='#00E676', outline='#00D9FF', width=3, 
                                                               stipple='gray50')
        
        self.task4_dragging = False
        self.task4_dropped = False
        self.task4_circle_pos = [100, 200]
    
    def render_task5(self, parent, task):
        """Task 5: Multiple clicks"""
        label = tk.Label(parent, text="Click all three buttons using pinch gestures:",
                        font=('Segoe UI', 14), bg='#0a1628', fg='#B0BEC5')
        label.pack(pady=30)
        
        button_frame = tk.Frame(parent, bg='#0a1628')
        button_frame.pack(expand=True)
        
        self.task5_buttons = []
        self.task5_clicked = [False, False, False]
        
        colors = ['#0066FF', '#4A9EFF', '#00D9FF']  # High-tech blues
        for i in range(3):
            btn = tk.Button(button_frame, text=f"Button {i+1}",
                           font=('Segoe UI', 14, 'bold'),
                           bg=colors[i], fg='#ffffff',
                           activebackground=colors[i],
                           width=15, height=2,
                           relief=tk.FLAT,
                           borderwidth=0,
                           command=lambda idx=i: self.handle_task5_click(idx))
            btn.pack(pady=10)
            self.task5_buttons.append(btn)
    
    def handle_task5_click(self, idx):
        """Handle task 5 button click"""
        if not self.task5_clicked[idx]:
            self.task5_clicked[idx] = True
            self.task5_buttons[idx].config(bg='#00E676', state='disabled', fg='#0a1628')
            
            if all(self.task5_clicked):
                self.complete_task(True)
    
    def complete_task(self, success):
        """Mark current task as complete"""
        if self.current_task >= len(self.get_tasks()):
            return
        
        task = self.get_tasks()[self.current_task]
        duration = time.time() - self.task_start_time if self.task_start_time else 0
        
        result = {
            'task_id': task['id'],
            'success': success,
            'duration': duration,
            'timestamp': time.time()
        }
        self.task_results.append(result)
        
        if self.on_task_complete:
            self.on_task_complete(task['id'], success, duration)
        
        # Move to next task
        self.current_task += 1
        self.test_window.after(1500, self.show_task_screen)  # Brief pause before next task
    
    def skip_task(self):
        """Skip current task"""
        self.complete_task(False)
    
    def show_survey(self):
        """Show survey questions"""
        if not self.test_window:
            return
        
        for widget in self.test_window.winfo_children():
            widget.destroy()
        
        # Header with high-tech styling
        header_container = tk.Frame(self.test_window, bg='#1a1f3a')
        header_container.pack(fill=tk.X, padx=15, pady=(20, 15))
        
        border_frame = tk.Frame(header_container, bg='#00D9FF', height=3)
        border_frame.pack(fill=tk.X)
        
        header_frame = tk.Frame(header_container, bg='#1a1f3a')
        header_frame.pack(fill=tk.X, pady=15)
        
        header = tk.Label(header_frame, text="⚡ POST-TEST SURVEY",
                         font=('Segoe UI', 22, 'bold'),
                         bg='#1a1f3a', fg='#00D9FF')
        header.pack()
        
        # Scrollable frame with high-tech styling
        canvas = tk.Canvas(self.test_window, bg='#0a1628', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.test_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#0a1628')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        questions = self.get_survey_questions()
        self.survey_vars = {}
        
        for i, q in enumerate(questions):
            # High-tech card styling
            frame = tk.Frame(scrollable_frame, bg='#1a1f3a', relief=tk.FLAT)
            frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Accent border
            accent = tk.Frame(frame, bg='#4A9EFF', height=2)
            accent.pack(fill=tk.X)
            
            content_frame = tk.Frame(frame, bg='#1a1f3a')
            content_frame.pack(fill=tk.X, padx=2, pady=10)
            
            q_label = tk.Label(content_frame, text=f"{i+1}. {q['question']}",
                              font=('Segoe UI', 13, 'bold'),
                              bg='#1a1f3a', fg='#ffffff', anchor='w', wraplength=800)
            q_label.pack(anchor='w', padx=15, pady=(5, 10))
            
            var = tk.StringVar(value="")
            self.survey_vars[q['id']] = var
            
            if q['type'] == 'scale':
                # 1-5 scale with high-tech styling
                scale_frame = tk.Frame(content_frame, bg='#1a1f3a')
                scale_frame.pack(anchor='w', padx=15, pady=5)
                
                for j in range(1, 6):
                    rb = tk.Radiobutton(scale_frame, text=str(j), variable=var, value=str(j),
                                       font=('Segoe UI', 12, 'bold'), 
                                       bg='#1a1f3a', fg='#B0BEC5',
                                       selectcolor='#00D9FF', 
                                       activebackground='#2a2f4a',
                                       activeforeground='#00D9FF')
                    rb.pack(side=tk.LEFT, padx=12)
                
                labels_frame = tk.Frame(content_frame, bg='#1a1f3a')
                labels_frame.pack(anchor='w', padx=30, pady=(0, 10))
                tk.Label(labels_frame, text=q.get('low_label', 'Poor'), 
                        font=('Segoe UI', 9), bg='#1a1f3a', fg='#64748B').pack(side=tk.LEFT)
                tk.Label(labels_frame, text=q.get('high_label', 'Excellent'), 
                        font=('Segoe UI', 9), bg='#1a1f3a', fg='#64748B').pack(side=tk.RIGHT)
            
            elif q['type'] == 'yesno':
                yesno_frame = tk.Frame(content_frame, bg='#1a1f3a')
                yesno_frame.pack(anchor='w', padx=15, pady=(0, 10))
                
                tk.Radiobutton(yesno_frame, text="Yes", variable=var, value="yes",
                              font=('Segoe UI', 12, 'bold'), 
                              bg='#1a1f3a', fg='#B0BEC5',
                              selectcolor='#00D9FF', 
                              activebackground='#2a2f4a',
                              activeforeground='#00D9FF').pack(side=tk.LEFT, padx=20)
                tk.Radiobutton(yesno_frame, text="No", variable=var, value="no",
                              font=('Segoe UI', 12, 'bold'), 
                              bg='#1a1f3a', fg='#B0BEC5',
                              selectcolor='#00D9FF', 
                              activebackground='#2a2f4a',
                              activeforeground='#00D9FF').pack(side=tk.LEFT, padx=20)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Submit button with high-tech styling
        submit_btn = tk.Button(self.test_window, text="⚡ SUBMIT SURVEY",
                              font=('Segoe UI', 14, 'bold'),
                              bg='#0066FF', fg='#ffffff',
                              activebackground='#4A9EFF',
                              relief=tk.FLAT,
                              borderwidth=0,
                              command=self.submit_survey,
                              width=20, height=2)
        submit_btn.pack(pady=20)
    
    def get_survey_questions(self):
        """Return survey questions"""
        return [
            {
                'id': 'q1',
                'question': 'How easy was it to move the mouse cursor?',
                'type': 'scale',
                'low_label': 'Very Difficult',
                'high_label': 'Very Easy'
            },
            {
                'id': 'q2',
                'question': 'How accurate was the click detection?',
                'type': 'scale',
                'low_label': 'Not Accurate',
                'high_label': 'Very Accurate'
            },
            {
                'id': 'q3',
                'question': 'Did you experience any lag or delay in gesture recognition?',
                'type': 'yesno'
            },
            {
                'id': 'q4',
                'question': 'How intuitive were the gesture controls?',
                'type': 'scale',
                'low_label': 'Not Intuitive',
                'high_label': 'Very Intuitive'
            },
            {
                'id': 'q5',
                'question': 'Would you use this system for everyday computer interaction?',
                'type': 'yesno'
            }
        ]
    
    def submit_survey(self):
        """Submit survey and save results"""
        # Collect answers
        for q_id, var in self.survey_vars.items():
            self.survey_answers[q_id] = var.get()
        
        # Save results to file
        results = {
            'timestamp': time.time(),
            'tasks': self.task_results,
            'survey': self.survey_answers
        }
        
        os.makedirs('test_results', exist_ok=True)
        filename = f"test_results/test_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Show completion message
        for widget in self.test_window.winfo_children():
            widget.destroy()
        
        complete_label = tk.Label(self.test_window, 
                                 text="✅ Test Complete!\n\nThank you for participating.\nResults saved.",
                                 font=('Segoe UI', 16),
                                 bg='#1e1e1e', fg='#4CAF50',
                                 justify=tk.CENTER)
        complete_label.pack(expand=True)
        
        close_btn = tk.Button(self.test_window, text="Close",
                             font=('Segoe UI', 12),
                             bg='#2d2d2d', fg='#ffffff',
                             command=self.close_test,
                             width=15)
        close_btn.pack(pady=20)
        
        if self.on_test_complete:
            self.on_test_complete(results)

