"""Enroll a student:  python enroll.py --roll 23CSD001 --name "Rithika"

Look at the camera, press SPACE to capture a sample (5 by default, vary your
angle/expression slightly), ESC to cancel.
"""
import argparse

import cv2
import numpy as np

import db
from config import CAMERA_INDEX, ENROLL_SAMPLES
from face_engine import FaceEngine, open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True, help="roll number / ID (unique)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    engine = FaceEngine()
    cap = open_camera(CAMERA_INDEX)
    samples = []

    while len(samples) < ENROLL_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.detect(frame)
        view = frame.copy()
        for f in faces:
            x, y, w, h = f[:4].astype(int)
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        msg = f"{args.name}: {len(samples)}/{ENROLL_SAMPLES} - SPACE capture, ESC quit"
        cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Enroll", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Cancelled.")
            break
        if key == 32:
            if len(faces) != 1:
                print(f"Need exactly one face in frame (found {len(faces)}). Try again.")
                continue
            samples.append(engine.embed(frame, faces[0]))
            print(f"Captured {len(samples)}/{ENROLL_SAMPLES}")

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < ENROLL_SAMPLES:
        print("Not enough samples; nothing saved.")
        return

    mean = np.mean(np.vstack(samples), axis=0, keepdims=True).astype(np.float32)
    mean /= np.linalg.norm(mean)
    conn = db.connect()
    db.add_student(conn, args.roll, args.name, mean)
    print(f"Enrolled {args.name} ({args.roll}).")


if __name__ == "__main__":
    main()
