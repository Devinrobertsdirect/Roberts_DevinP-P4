# ui.py

# Minimal Tkinter UI that embeds OpenCV frames (converted to PIL ImageTk).

# Shows live feed, gesture label, confidence, FPS, and debug log.



import tkinter as tk

from PIL import Image, ImageTk

import cv2

import time

import threading

import queue



class SimpleUI:

    def __init__(self, width=800, height=600, title="Hand Control R&D"):

        self.root = tk.Tk()

        self.root.title(title)

        self.width = width

        self.height = height



        # video panel

        self.video_label = tk.Label(self.root)

        self.video_label.pack()



        # info frame

        self.info_frame = tk.Frame(self.root)

        self.info_frame.pack(fill=tk.X)

        self.gesture_var = tk.StringVar(value="Gesture: -")

        self.conf_var = tk.StringVar(value="Confidence: -")

        self.fps_var = tk.StringVar(value="FPS: -")

        tk.Label(self.info_frame, textvariable=self.gesture_var, width=30, anchor='w').pack(side=tk.LEFT, padx=4)

        tk.Label(self.info_frame, textvariable=self.conf_var, width=20, anchor='w').pack(side=tk.LEFT, padx=4)

        tk.Label(self.info_frame, textvariable=self.fps_var, width=12, anchor='w').pack(side=tk.LEFT, padx=4)



        # debug log (simple)

        self.log_box = tk.Text(self.root, height=6, state='disabled')

        self.log_box.pack(fill=tk.X, padx=4, pady=4)



        # internal vars

        self.queue = queue.Queue()

        self.running = False



    def update_frame(self, cv2_bgr_image):

        # convert BGR -> RGB -> PIL -> ImageTk

        image_rgb = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2RGB)

        im_pil = Image.fromarray(image_rgb)

        # resize to fit UI width if needed

        im_pil = im_pil.resize((min(self.width, im_pil.width), int(im_pil.height * min(self.width/im_pil.width, 1.0))))

        imgtk = ImageTk.PhotoImage(image=im_pil)

        self.video_label.imgtk = imgtk

        self.video_label.configure(image=imgtk)



    def set_gesture_text(self, gesture_text):

        self.gesture_var.set(f"Gesture: {gesture_text}")



    def set_confidence(self, conf_text):

        self.conf_var.set(f"Confidence: {conf_text}")



    def set_fps(self, fps_text):

        self.fps_var.set(f"FPS: {fps_text}")



    def log(self, text):

        self.log_box.config(state='normal')

        self.log_box.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")

        self.log_box.see(tk.END)

        self.log_box.config(state='disabled')



    def start(self):

        self.running = True
        # Don't start mainloop here - it should run in the main thread


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

