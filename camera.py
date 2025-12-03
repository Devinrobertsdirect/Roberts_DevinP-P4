# camera.py

# Simple webcam wrapper using OpenCV. Provides a generator that yields frames.

import cv2

import time



class Webcam:

    """

    Simple OpenCV webcam wrapper.

    Usage:

        cam = Webcam(0, width=640, height=480)

        for frame, ts in cam.frames():

            # process frame (BGR numpy array)

    """

    def __init__(self, device_index=0, width=640, height=480, fps=30):

        self.device_index = device_index

        self.width = width

        self.height = height

        self.fps = fps

        self.cap = None



    def open(self):

        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)

        if not self.cap.isOpened():

            raise RuntimeError(f"Cannot open webcam index {self.device_index}")

        # apply resolution hints

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # try to set fps, but may be ignored

        self.cap.set(cv2.CAP_PROP_FPS, self.fps)



    def frames(self):

        """

        Generator yielding (frame_bgr, timestamp)

        """

        if self.cap is None:

            self.open()

        while True:

            ret, frame = self.cap.read()

            ts = time.time()

            if not ret:

                break

            yield frame, ts



    def release(self):

        if self.cap:

            self.cap.release()

            self.cap = None



if __name__ == "__main__":

    # quick test runner

    cam = Webcam()

    cam.open()

    for i, (frame, ts) in enumerate(cam.frames()):

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

        if i > 300:

            break

    cam.release()

    cv2.destroyAllWindows()

