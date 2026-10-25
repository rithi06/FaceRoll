# Face Recognition Attendance System

Real-time attendance using a webcam. Pure OpenCV (YuNet face detection + SFace
face recognition), SQLite storage. No dlib or TensorFlow needed.

## Setup
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_models.py        # one-time, ~40 MB
```

## Use
```bash
python enroll.py --roll 23CSE001 --name "Rithika"    # SPACE x5 to capture, ESC cancels
python attendance.py                              # live recognition, q to quit
python report.py --csv                            # today's report, saved to exports/
python report.py --date 2026-10-24
```

## How it works
1. **Detect**: YuNet finds faces and 5 landmarks in each frame.
2. **Embed**: SFace aligns each face and produces a 128-d vector.
3. **Match**: cosine similarity against each enrolled student's averaged embedding
   (threshold 0.363, OpenCV's recommended value).
4. **Confirm**: a student must be matched for 5 consecutive frames before being marked,
   which avoids false positives from a single noisy frame.
5. **Record**: one row per student per day (`UNIQUE(student_id, day)`).

## Tuning (config.py)
- Misrecognising people -> raise `MATCH_THRESHOLD` (e.g. 0.45) or `CONFIRM_FRAMES`.
- Not recognising enrolled people -> re-enroll in the same lighting, lower the threshold slightly.
- Wrong camera -> change `CAMERA_INDEX`.

## Limitations 
- No liveness detection: a printed photo or phone screen can fool it.
- Accuracy drops in poor lighting or with heavy occlusion.
- Brute-force matching is fine for hundreds of students; use FAISS beyond that.
- Face data is biometric and sensitive: get consent, and keep attendance.db private.

## Ideas to extend
- Flask dashboard showing live attendance and charts
- Mask detection alongside recognition
- Blink-based liveness check
- Email/Telegram summary at end of class
