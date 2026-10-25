"""SQLite storage for students (with face embeddings) and attendance records."""
import sqlite3
from datetime import datetime

import numpy as np

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no    TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    day        TEXT NOT NULL,
    time       TEXT NOT NULL,
    UNIQUE (student_id, day)
);
"""


def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def add_student(conn, roll_no, name, embedding):
    """Insert or replace a student's embedding (re-enrolling updates the face)."""
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO students (roll_no, name, embedding, created_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(roll_no) DO UPDATE SET name = excluded.name,
                                              embedding = excluded.embedding""",
        (roll_no, name, blob, now),
    )
    conn.commit()


def load_students(conn):
    """Return list of (id, roll_no, name, embedding[1x128 float32])."""
    rows = conn.execute("SELECT id, roll_no, name, embedding FROM students").fetchall()
    return [
        (sid, roll, name, np.frombuffer(blob, dtype=np.float32).reshape(1, -1))
        for sid, roll, name, blob in rows
    ]


def mark_present(conn, student_id):
    """Mark attendance once per day. Returns True if newly marked, False if already present."""
    now = datetime.now()
    cur = conn.execute(
        "INSERT OR IGNORE INTO attendance (student_id, day, time) VALUES (?, ?, ?)",
        (student_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
    )
    conn.commit()
    return cur.rowcount == 1


def already_present_today(conn):
    day = datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute("SELECT student_id FROM attendance WHERE day = ?", (day,)).fetchall()
    return {r[0] for r in rows}


def report(conn, day):
    """Every enrolled student with Present/Absent status for the given day."""
    return conn.execute(
        """SELECT s.roll_no, s.name,
                  CASE WHEN a.id IS NULL THEN 'Absent' ELSE 'Present' END,
                  COALESCE(a.time, '')
           FROM students s
           LEFT JOIN attendance a ON a.student_id = s.id AND a.day = ?
           ORDER BY s.roll_no""",
        (day,),
    ).fetchall()
