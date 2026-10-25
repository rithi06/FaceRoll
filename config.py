"""Central settings. Tweak these instead of editing the scripts."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "attendance.db"
EXPORT_DIR = BASE_DIR / "exports"

DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

CAMERA_INDEX = 0            # change to 1 if you have an external webcam
DETECT_SCORE = 0.9          # face detector confidence
MATCH_THRESHOLD = 0.363     # cosine similarity; 0.363 is OpenCV's recommended value
                            # raise (e.g. 0.45) for stricter matching, lower for looser
CONFIRM_FRAMES = 5          # same person must be seen this many frames in a row
ENROLL_SAMPLES = 5          # face captures averaged per student at enrollment
