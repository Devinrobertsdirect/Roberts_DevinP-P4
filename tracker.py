# tracker.py

# MediaPipe Hands wrapper. Returns normalized 21-landmark arrays and annotated frames.

import mediapipe as mp

import numpy as np

import cv2



mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils



class HandTracker:

    """

    Wraps MediaPipe Hands and provides:

      - process(frame_bgr) -> (annotated_frame_bgr, hands_data)

    Where hands_data is a list of dicts: {'landmarks': np.array(shape=(21,3)), 'handedness': 'Left'/'Right', 'score': float}

    Landmarks are normalized (x,y,z) with x,y in [0,1] relative to image size.

    """

    def __init__(self, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.hands = mp_hands.Hands(static_image_mode=False,

                                    max_num_hands=max_num_hands,

                                    model_complexity=model_complexity,

                                    min_detection_confidence=min_detection_confidence,

                                    min_tracking_confidence=min_tracking_confidence)



    def _mp_to_array(self, hand_landmarks):

        arr = np.zeros((21, 3), dtype=np.float32)

        for i, lm in enumerate(hand_landmarks.landmark):

            arr[i, 0] = lm.x

            arr[i, 1] = lm.y

            arr[i, 2] = lm.z

        return arr



    def process(self, frame_bgr):

        """

        Input: BGR frame (numpy array)

        Returns: annotated_bgr, hands_data

        """

        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        hands = []

        annotated = frame_bgr.copy()



        if results.multi_hand_landmarks:

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # optionally draw

                mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                arr = self._mp_to_array(hand_landmarks)  # normalized

                handedness_label = None

                if results.multi_handedness:

                    handedness_label = results.multi_handedness[idx].classification[0].label

                    score = results.multi_handedness[idx].classification[0].score

                else:

                    score = 1.0

                hands.append({

                    'landmarks': arr,

                    'handedness': handedness_label,

                    'score': float(score),

                    'landmarks_obj': hand_landmarks  # Store original MediaPipe object for text positioning

                })

        return annotated, hands, results


    def draw_gesture_text(self, frame, gesture_name, confidence, hand_landmarks=None):
        """
        Draw gesture name and confidence as text overlay on the frame.
        If hand_landmarks provided, draw near the hand position.
        """
        h, w = frame.shape[:2]
        
        # Get position - if hand detected, use wrist position, otherwise center-top
        if hand_landmarks is not None:
            # Convert normalized to pixel coordinates
            wrist = hand_landmarks.landmark[0]
            x = int(wrist.x * w)
            y = int(wrist.y * h) - 50  # Above the wrist
        else:
            x = w // 2
            y = 50
        
        # Format text
        gesture_display = gesture_name.replace('_', ' ').title()
        text = f"{gesture_display}"
        conf_text = f"{confidence:.0%}"
        
        # Background rectangle for better visibility
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        conf_w, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw background
        cv2.rectangle(frame, 
                     (x - 10, y - text_h - 40), 
                     (x + max(text_w, conf_w) + 10, y + 10), 
                     (0, 0, 0), -1)
        
        # Draw gesture name (large, bold, color based on confidence)
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 165, 255)
        cv2.putText(frame, text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        # Draw confidence
        cv2.putText(frame, conf_text, (x, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame



    def close(self):

        self.hands.close()



if __name__ == "__main__":

    # quick test (press q to quit)

    from camera import Webcam

    cam = Webcam()

    tracker = HandTracker()

    for frame, ts in cam.frames():

        annotated, hands = tracker.process(frame)

        cv2.imshow("annotated", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    tracker.close()

    cam.release()

    cv2.destroyAllWindows()

