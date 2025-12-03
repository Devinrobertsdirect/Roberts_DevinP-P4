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

                    'score': float(score)

                })

        return annotated, hands



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

