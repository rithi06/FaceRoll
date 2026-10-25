"""Run live attendance:  python attendance.py     (press q to quit)

A student is marked present only after being recognised for CONFIRM_FRAMES
frames in a row, and only once per day.
"""
from collections import defaultdict

import cv2

import db
from config import CAMERA_INDEX, CONFIRM_FRAMES
from face_engine import FaceEngine, open_camera

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 255, 255)


def main():
    engine = FaceEngine()
    conn = db.connect()
    students = db.load_students(conn)
    if not students:
        raise SystemExit("No students enrolled yet. Run enroll.py first.")

    present = db.already_present_today(conn)
    streak = defaultdict(int)        # student_id -> consecutive recognised frames
    banner, banner_ttl = "", 0

    cap = open_camera(CAMERA_INDEX)
    print(f"{len(students)} students loaded. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        seen_ids = set()
        for f in engine.detect(frame):
            x, y, w, h = f[:4].astype(int)
            student, score = engine.best_match(engine.embed(frame, f), students)

            if student is None:
                cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 2)
                cv2.putText(frame, "Unknown", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
                continue

            sid, roll, name, _ = student
            seen_ids.add(sid)
            streak[sid] += 1
            colour = GREEN if sid in present else YELLOW
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            if sid not in present and streak[sid] >= CONFIRM_FRAMES:
                if db.mark_present(conn, sid):
                    banner, banner_ttl = f"Marked present: {name} ({roll})", 60
                    print(banner)
                present.add(sid)

        for sid in list(streak):         # reset streaks for people not seen this frame
            if sid not in seen_ids:
                streak[sid] = 0

        cv2.putText(frame, f"Present today: {len(present)}/{len(students)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        if banner_ttl > 0:
            cv2.putText(frame, banner, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            banner_ttl -= 1

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
