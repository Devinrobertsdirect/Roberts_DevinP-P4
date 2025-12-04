# main.py

# Orchestrator: camera -> tracker -> gesture detection -> action controller -> UI

import time

import cv2

import numpy as np

import threading

from camera import Webcam

from tracker import HandTracker

from gestures import is_pinch, is_open_hand, is_fist, compute_features, DataLogger

from controller import ActionController

from ui import SimpleUI



# Configuration

FRAME_W = 640

FRAME_H = 480

DEVICE_INDEX = 0



def heuristic_gesture_classifier(landmarks):

    """

    Return (gesture_name, confidence, extra)

    Simple logic combining heuristic detectors. Confidence is heuristic score [0,1]

    """

    # default

    name = 'unknown'

    conf = 0.0

    extra = {}



    pinch, d_pin = is_pinch(landmarks, thresh=0.05)

    openh, avg_open = is_open_hand(landmarks, open_threshold=0.11)

    fist, avg_fist = is_fist(landmarks, fist_threshold=0.06)



    # heuristics priority

    if pinch:

        name = 'pinch'

        # give higher confidence for very close pinch

        conf = float(max(0.6, min(1.0, 0.4 + (0.05 - d_pin) * 20)))

        extra['pinch_distance'] = d_pin

        # determine state (simple toggle): if extremely close -> start drag else single click

        extra['pinch_state'] = 'start' if d_pin < 0.035 else 'hold'



    elif fist:

        name = 'fist'

        conf = float(min(1.0, 0.6 + (0.06 - avg_fist) * 10))



    elif openh:

        # check for index pointing: index finger farther from wrist than middle/pinky to detect pointing

        idx = landmarks[8]

        wrist = landmarks[0]

        idx_dist = np.linalg.norm(idx - wrist)

        avg_other = np.mean([np.linalg.norm(landmarks[i] - wrist) for i in [12,16,20]])

        if idx_dist > avg_other * 1.1:

            name = 'index_point'

            conf = float(min(1.0, 0.6 + (idx_dist - avg_other) * 10))

        else:

            name = 'open'

            conf = float(min(1.0, 0.5 + (avg_open - 0.11) * 5))

    else:

        name = 'unknown'

        conf = 0.1



    return name, conf, extra



def process_camera_loop(ui, cam, tracker, controller, stop_event):
    """Background thread function for camera processing"""
    # Queue log message (thread-safe)
    try:
        ui.queue.put({'type': 'log', 'message': 'Camera thread started. Initializing camera...'}, block=False)
    except:
        pass
    
    try:
        cam.open()
    except Exception as e:
        try:
            ui.queue.put({'type': 'log', 'message': f'Failed to open camera: {e}'}, block=False)
        except:
            pass
        stop_event.set()
        return

    try:
        ui.queue.put({'type': 'log', 'message': 'Camera opened. Starting processing loop.'}, block=False)
    except:
        pass
    
    last_time = time.time()
    fps_smooth = 0.0

    try:
        for frame, ts in cam.frames():
            if stop_event.is_set():
                break

            annotated, hands = tracker.process(frame)  # annotated has MP drawings

            # determine FPS
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 1e-6 else 0.0
            fps_smooth = fps_smooth * 0.8 + fps * 0.2
            last_time = now

            gesture_label = 'none'
            confidence = 0.0

            # process first detected hand only for R&D prototype
            if len(hands) > 0:
                primary = hands[0]
                lms = primary['landmarks']  # normalized 21x3
                # heuristic classifier
                gesture_label, confidence, extra = heuristic_gesture_classifier(lms)
                # call controller to perform action
                try:
                    controller.perform_gesture_action(gesture_label, lms, FRAME_W, FRAME_H, extra=extra)
                except Exception as e:
                    try:
                        ui.queue.put({'type': 'log', 'message': f'Controller error: {e}'}, block=False)
                    except:
                        pass
            else:
                extra = {}

            # Queue updates for UI (thread-safe)
            update_data = {
                'type': 'frame',
                'frame': annotated,
                'gesture': gesture_label,
                'confidence': f"{confidence:.2f}",
                'fps': f"{fps_smooth:.1f}"
            }
            
            try:
                ui.queue.put(update_data, block=False)
            except:
                pass  # Queue full, skip this frame

    except Exception as e:
        try:
            ui.queue.put({'type': 'log', 'message': f'Processing error: {e}'}, block=False)
        except:
            pass
    finally:
        try:
            ui.queue.put({'type': 'log', 'message': 'Camera thread stopping...'}, block=False)
        except:
            pass
        tracker.close()
        cam.release()
        stop_event.set()


def update_ui_from_queue(ui, stop_event):
    """Periodically check queue and update UI (runs in main thread)"""
    if stop_event.is_set():
        return

    # Process all pending updates from queue
    try:
        while True:
            try:
                update = ui.queue.get_nowait()
                if update.get('type') == 'log':
                    ui.log(update['message'])
                elif update.get('type') == 'frame':
                    ui.update_frame(update['frame'])
                    ui.set_gesture_text(update['gesture'])
                    ui.set_confidence(update['confidence'])
                    ui.set_fps(update['fps'])
            except:
                break  # Queue empty
    except:
        pass

    # Schedule next update (use partial to avoid closure issues)
    if not stop_event.is_set():
        ui.root.after(33, lambda: update_ui_from_queue(ui, stop_event))  # ~30 FPS UI updates


def main_loop():
    cam = Webcam(device_index=DEVICE_INDEX, width=FRAME_W, height=FRAME_H)
    tracker = HandTracker()
    controller = ActionController()
    ui = SimpleUI(width=800, height=600)
    datalog = DataLogger(path='gesture_samples.csv')

    stop_event = threading.Event()

    ui.start()
    ui.log("UI initialized. Starting camera thread...")

    # Start camera processing in background thread
    camera_thread = threading.Thread(target=process_camera_loop, 
                                     args=(ui, cam, tracker, controller, stop_event),
                                     daemon=True)
    camera_thread.start()

    # Start UI update loop (runs in main thread) - use a helper to avoid closure issues
    def schedule_update():
        update_ui_from_queue(ui, stop_event)
    ui.root.after(100, schedule_update)
    
    # Handle window close
    def on_closing():
        ui.log("Shutting down...")
        stop_event.set()
        tracker.close()
        cam.release()
        ui.stop()

    ui.root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        # Run Tkinter mainloop in main thread
        ui.run()
    except KeyboardInterrupt:
        ui.log("Interrupted by user.")
        on_closing()
    except Exception as e:
        ui.log(f"Error: {e}")
        on_closing()



if __name__ == "__main__":

    main_loop()

