# controller.py

# Converts gesture detections into OS mouse/keyboard events.

# Uses pyautogui for cross-platform mouse movement/clicks.

# NOTE: pyautogui coordinates use screen pixels



import pyautogui

import numpy as np

import time



pyautogui.FAILSAFE = False  # disable pyautogui corner fail-safe (optional)



class ActionController:

    """

    Maps gestures to actions.

    Basic default mappings:

      - index-pointing (open hand with index extended): move mouse

      - pinch: left-click and hold (drag)

      - fist: right-click

      - two-hand open (not implemented): custom

    """



    def __init__(self, screen_width=None, screen_height=None, smoothing=0.6):

        screen_w, screen_h = pyautogui.size()

        self.screen_width = screen_w if screen_width is None else screen_width

        self.screen_height = screen_h if screen_height is None else screen_height

        self.smoothing = smoothing

        self._prev_mouse = None

        self._dragging = False

        self.last_action_time = 0.0



    def landmark_to_screen(self, landmark, frame_width, frame_height):

        """

        Convert normalized landmark (x,y) to screen pixel coordinates:

          - landmark is normalized x,y in [0,1] relative to frame

          - also flip x if needed according to webcam mirror. Adjust here.

        """

        x_norm, y_norm = float(landmark[0]), float(landmark[1])

        # Mirror correction: many webcams show mirrored image; decide mapping behavior.

        screen_x = int(x_norm * self.screen_width)

        screen_y = int(y_norm * self.screen_height)

        # smoothing

        if self._prev_mouse is None:

            self._prev_mouse = (screen_x, screen_y)

            return screen_x, screen_y

        px, py = self._prev_mouse

        nx = int(px + (screen_x - px) * (1.0 - self.smoothing))

        ny = int(py + (screen_y - py) * (1.0 - self.smoothing))

        self._prev_mouse = (nx, ny)

        return nx, ny



    def move_mouse_to_landmark(self, landmark, frame_w, frame_h):

        x, y = self.landmark_to_screen(landmark, frame_w, frame_h)

        pyautogui.moveTo(x, y, duration=0)  # instant move



    def left_click(self):

        pyautogui.click()



    def right_click(self):

        pyautogui.click(button='right')



    def start_drag(self):

        if not self._dragging:

            pyautogui.mouseDown()

            self._dragging = True



    def stop_drag(self):

        if self._dragging:

            pyautogui.mouseUp()

            self._dragging = False



    def scroll(self, clicks):

        pyautogui.scroll(clicks)



    def type_text(self, text):

        pyautogui.typewrite(text)



    def perform_gesture_action(self, gesture_name, landmarks, frame_w, frame_h, extra=None):

        """

        Map named gestures to actions.

        Gesture names expected: 'pinch', 'open', 'fist', 'point'

        """

        now = time.time()

        # throttle actions slightly to avoid flurry

        if now - self.last_action_time < 0.03:

            return

        self.last_action_time = now



        # Action gestures (control mouse/keyboard)
        if gesture_name == 'point' or gesture_name == 'index_point':

            # move mouse using index tip

            index_tip = landmarks[8]  # normalized

            self.move_mouse_to_landmark(index_tip, frame_w, frame_h)



        elif gesture_name == 'pinch':

            # pinch -> drag (start/stop)

            if extra and extra.get('pinch_state') == 'start':

                self.start_drag()

            elif extra and extra.get('pinch_state') == 'end':

                self.stop_drag()

            else:

                # fallback: single click

                self.left_click()



        elif gesture_name == 'fist':

            # quick right click

            self.right_click()



        elif gesture_name == 'open' or gesture_name == 'open_hand' or gesture_name == 'open_palm':

            # stop drag if open palm/hand

            self.stop_drag()
        
        elif gesture_name.startswith('number_'):
            # Number gestures - could map to keyboard numbers or custom actions
            try:
                number = int(gesture_name.split('_')[1])
                # Example: scroll based on number
                if number == 1:
                    self.scroll(1)
                elif number == 2:
                    self.scroll(-1)
            except:
                pass

        else:

            pass  # Symbol gestures (thumbs_up, peace_sign, ok_sign, rock_on) and unknown gestures don't trigger actions



if __name__ == "__main__":

    # simple test (move mouse to center)

    ctrl = ActionController()

    ctrl.move_mouse_to_landmark((0.5, 0.5), 640, 480)

    print("Moved mouse to center")

