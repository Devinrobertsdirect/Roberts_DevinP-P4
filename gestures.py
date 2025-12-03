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

