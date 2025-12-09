# gestures.py

# Feature extraction, simple heuristic recognizers, data logging, and a tiny ML training stub.

import numpy as np

import math

import csv

import os

from sklearn.neighbors import KNeighborsClassifier

import pickle



# Landmark indices (MediaPipe)

TIP_IDS = {

    'thumb_tip': 4,

    'index_tip': 8,

    'middle_tip': 12,

    'ring_tip': 16,

    'pinky_tip': 20,

    'wrist': 0

}



def euclidean(a, b):

    return np.linalg.norm(a - b)



def normalized_distance(a, b):

    # Compute euclidean distance in normalized coordinates

    return euclidean(a, b)



def landmark_to_vector(landmarks):

    """

    Flatten 21x3 landmarks into a 63-vector.

    """

    return landmarks.flatten()



def compute_features(landmarks):

    """

    Compute a small set of features useful for simple heuristics and ML:

      - distances between thumb_tip and index_tip (pinch proxy)

      - avg fingertip distance to wrist

      - spread metric (max pairwise fingertip distance)

      - flattened landmarks vector (for ML)

    """

    tips = np.array([landmarks[TIP_IDS['thumb_tip']],

                     landmarks[TIP_IDS['index_tip']],

                     landmarks[TIP_IDS['middle_tip']],

                     landmarks[TIP_IDS['ring_tip']],

                     landmarks[TIP_IDS['pinky_tip']]])

    wrist = landmarks[TIP_IDS['wrist']]

    # distances

    thumb_index = normalized_distance(tips[0], tips[1])

    avg_tip_wrist = float(np.mean([normalized_distance(t, wrist) for t in tips]))

    max_spread = float(np.max([euclidean(t1, t2) for i, t1 in enumerate(tips) for j, t2 in enumerate(tips) if j>i]))

    # flattened raw vector for ML

    flat = landmark_to_vector(landmarks)

    # Compose feature vector (small)

    feat = np.concatenate(([thumb_index, avg_tip_wrist, max_spread], flat))

    return feat



# Heuristic gesture detectors

def is_pinch(landmarks, thresh=0.05):

    """

    Simple pinch detection: thumb_tip and index_tip close in normalized coords.

    Threshold tuned empirically; adjust per camera/resolution.

    """

    d = normalized_distance(landmarks[TIP_IDS['thumb_tip']], landmarks[TIP_IDS['index_tip']])

    return (d < thresh), float(d)



def is_open_hand(landmarks, open_threshold=0.10):

    """

    Open-hand proxy: average distance of fingertips to wrist is large

    """

    tips = np.array([landmarks[TIP_IDS['index_tip']],

                     landmarks[TIP_IDS['middle_tip']],

                     landmarks[TIP_IDS['ring_tip']],

                     landmarks[TIP_IDS['pinky_tip']]])

    wrist = landmarks[TIP_IDS['wrist']]

    avg_dist = float(np.mean([normalized_distance(t, wrist) for t in tips]))

    return (avg_dist > open_threshold), avg_dist



def is_fist(landmarks, fist_threshold=0.06):

    """

    Fist proxy: average fingertip-to-wrist distance small.

    """

    tips = np.array([landmarks[TIP_IDS['index_tip']],

                     landmarks[TIP_IDS['middle_tip']],

                     landmarks[TIP_IDS['ring_tip']],

                     landmarks[TIP_IDS['pinky_tip']]])

    wrist = landmarks[TIP_IDS['wrist']]

    avg_dist = float(np.mean([normalized_distance(t, wrist) for t in tips]))

    return (avg_dist < fist_threshold), avg_dist


def is_thumbs_up(landmarks, thumb_up_threshold=0.15):
    """
    Thumbs up detection: thumb extended upward, other fingers closed.
    Check if thumb tip is above thumb MCP and other fingers are close to wrist.
    """
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    
    # Thumb extended upward (y decreases upward in normalized coords)
    thumb_extended = thumb_tip[1] < thumb_mcp[1] - 0.05
    
    # Other fingers closed (index, middle, ring, pinky tips close to wrist)
    other_fingers = np.array([
        landmarks[8],  # index
        landmarks[12], # middle
        landmarks[16], # ring
        landmarks[20]  # pinky
    ])
    avg_other_dist = float(np.mean([normalized_distance(t, wrist) for t in other_fingers]))
    
    is_closed = avg_other_dist < 0.08
    return (thumb_extended and is_closed), float(thumb_tip[1] - thumb_mcp[1])


def is_peace_sign(landmarks, peace_threshold=0.10):
    """
    Peace sign (V sign): Index and middle finger extended, others closed.
    """
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Index and middle extended
    index_extended = normalized_distance(index_tip, wrist) > peace_threshold
    middle_extended = normalized_distance(middle_tip, wrist) > peace_threshold
    
    # Ring and pinky closed
    ring_closed = normalized_distance(ring_tip, wrist) < 0.07
    pinky_closed = normalized_distance(pinky_tip, wrist) < 0.07
    
    # Index and middle should be separated (not too close)
    index_middle_sep = normalized_distance(index_tip, middle_tip) > 0.03
    
    is_peace = index_extended and middle_extended and ring_closed and pinky_closed and index_middle_sep
    avg_extended = float(np.mean([normalized_distance(index_tip, wrist), normalized_distance(middle_tip, wrist)]))
    return is_peace, avg_extended


def is_ok_sign(landmarks, ok_threshold=0.04):
    """
    OK sign: Thumb and index finger form a circle, other fingers extended.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Thumb and index close (forming circle)
    thumb_index_dist = normalized_distance(thumb_tip, index_tip)
    circle_formed = thumb_index_dist < ok_threshold
    
    # Other fingers extended
    middle_extended = normalized_distance(middle_tip, wrist) > 0.10
    ring_extended = normalized_distance(ring_tip, wrist) > 0.10
    pinky_extended = normalized_distance(pinky_tip, wrist) > 0.10
    
    is_ok = circle_formed and middle_extended and ring_extended and pinky_extended
    return is_ok, float(thumb_index_dist)


def is_rock_on(landmarks, rock_threshold=0.10):
    """
    Rock on: Index and pinky extended, middle and ring closed.
    """
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Index and pinky extended
    index_extended = normalized_distance(index_tip, wrist) > rock_threshold
    pinky_extended = normalized_distance(pinky_tip, wrist) > rock_threshold
    
    # Middle and ring closed
    middle_closed = normalized_distance(middle_tip, wrist) < 0.07
    ring_closed = normalized_distance(ring_tip, wrist) < 0.07
    
    is_rock = index_extended and pinky_extended and middle_closed and ring_closed
    avg_extended = float(np.mean([normalized_distance(index_tip, wrist), normalized_distance(pinky_tip, wrist)]))
    return is_rock, avg_extended


def is_number_gesture(landmarks):
    """
    Detect number gestures (1-5) based on extended fingers.
    Returns: (is_number, number, confidence)
    """
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Threshold for extended finger
    extend_threshold = 0.10
    
    thumb_ext = normalized_distance(thumb_tip, wrist) > extend_threshold
    index_ext = normalized_distance(index_tip, wrist) > extend_threshold
    middle_ext = normalized_distance(middle_tip, wrist) > extend_threshold
    ring_ext = normalized_distance(ring_tip, wrist) > extend_threshold
    pinky_ext = normalized_distance(pinky_tip, wrist) > extend_threshold
    
    # Count extended fingers (thumb counts separately)
    extended = [index_ext, middle_ext, ring_ext, pinky_ext]
    count = sum(extended)
    
    # Number 1-4 based on index-middle-ring-pinky
    if count == 1 and index_ext:
        return True, 1, 0.8
    elif count == 2 and index_ext and middle_ext:
        return True, 2, 0.8
    elif count == 3 and index_ext and middle_ext and ring_ext:
        return True, 3, 0.8
    elif count == 4 and index_ext and middle_ext and ring_ext and pinky_ext:
        return True, 4, 0.8
    elif count == 4 and thumb_ext:  # All 5 fingers including thumb
        return True, 5, 0.8
    
    return False, 0, 0.0



# Simple logger for saving labeled samples (used for training later)

class DataLogger:

    def __init__(self, path='gesture_samples.csv'):

        self.path = path

        # create header if not exists

        if not os.path.exists(self.path):

            with open(self.path, 'w', newline='') as f:

                # small header: label + 63 coordinates flattened

                header = ['label'] + [f'l{i}' for i in range(63)]

                writer = csv.writer(f)

                writer.writerow(header)



    def save_sample(self, landmarks, label):

        feat = landmark_to_vector(landmarks)

        row = [label] + feat.tolist()

        with open(self.path, 'a', newline='') as f:

            writer = csv.writer(f)

            writer.writerow(row)



# Tiny training stub using KNN (fast, no GPU)

class Trainer:

    def __init__(self, samples_path='gesture_samples.csv', model_path='gesture_knn.pkl'):

        self.samples_path = samples_path

        self.model_path = model_path

        self.model = None



    def load_samples(self):

        import pandas as pd

        if not os.path.exists(self.samples_path):

            raise FileNotFoundError("No samples found - record gesture samples first.")

        df = pd.read_csv(self.samples_path)

        X = df.iloc[:, 1:].values.astype(np.float32)

        y = df.iloc[:, 0].values

        return X, y



    def train_knn(self, n_neighbors=3):

        X, y = self.load_samples()

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        knn.fit(X, y)

        self.model = knn

        with open(self.model_path, 'wb') as f:

            pickle.dump(knn, f)

        print(f"Trained KNN saved to {self.model_path}")

        return knn



    def load_model(self):

        if not os.path.exists(self.model_path):

            raise FileNotFoundError("Model not found. Train first.")

        with open(self.model_path, 'rb') as f:

            self.model = pickle.load(f)

        return self.model



    def predict(self, landmarks_flat):

        if self.model is None:

            self.load_model()

        return self.model.predict(landmarks_flat.reshape(1, -1))[0], self.model.predict_proba(landmarks_flat.reshape(1, -1))[0]



if __name__ == "__main__":

    # quick local test of heuristics: provide a dummy landmarks array

    dummy = np.zeros((21,3), dtype=np.float32)

    # set thumb and index close

    dummy[4] = np.array([0.5,0.5,0.0])

    dummy[8] = np.array([0.505,0.5,0.0])

    print("Pinch?", is_pinch(dummy))

    print("Open?", is_open_hand(dummy))

    print("Fist?", is_fist(dummy))

