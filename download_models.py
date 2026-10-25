"""Download the two OpenCV Zoo ONNX models (about 40 MB total)."""
import sys
import urllib.request

from config import DETECTOR_MODEL, RECOGNIZER_MODEL, MODEL_DIR

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {
    DETECTOR_MODEL: f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_MODEL: f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

MODEL_DIR.mkdir(exist_ok=True)
for path, url in FILES.items():
    if path.exists() and path.stat().st_size > 100_000:
        print(f"Already have {path.name}")
        continue
    print(f"Downloading {path.name} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        print(f"Failed: {exc}\nDownload it manually from:\n  {url}\nand save it to {path}")
        sys.exit(1)
    if path.stat().st_size < 100_000:
        path.unlink()
        print(f"{path.name} looks too small (probably an LFS pointer). Download manually from:\n  {url}")
        sys.exit(1)
print("Done.")
