"""Thin wrapper around OpenCV's YuNet (detection) and SFace (recognition)."""
import sys

import cv2
import numpy as np

from config import DETECTOR_MODEL, RECOGNIZER_MODEL, DETECT_SCORE, MATCH_THRESHOLD


class FaceEngine:
    def __init__(self):
        for m in (DETECTOR_MODEL, RECOGNIZER_MODEL):
            if not m.exists():
                sys.exit(f"Missing model {m.name}. Run:  python download_models.py")
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_MODEL), "", (320, 320), DETECT_SCORE, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")

    def detect(self, frame):
        """Return an (N, 15) array of faces, or an empty array."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def embed(self, frame, face_row):
        """128-d embedding (shape 1x128) for one detected face."""
        aligned = self.recognizer.alignCrop(frame, face_row)
        return self.recognizer.feature(aligned)

    def best_match(self, embedding, students):
        """Compare against enrolled students.
        Returns (student_tuple or None, best_score)."""
        best, best_score = None, -1.0
        for student in students:
            score = self.recognizer.match(
                embedding, student[3], cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best, best_score = student, score
        if best_score >= MATCH_THRESHOLD:
            return best, best_score
        return None, best_score


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try CAMERA_INDEX = 1 in config.py.")
    return cap
