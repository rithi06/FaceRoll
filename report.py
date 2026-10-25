"""Attendance report:  python report.py [--date YYYY-MM-DD] [--csv]"""
import argparse
import csv
from datetime import datetime

import db
from config import EXPORT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--csv", action="store_true", help="also save to exports/")
    args = ap.parse_args()

    rows = db.report(db.connect(), args.date)
    if not rows:
        raise SystemExit("No students enrolled.")

    print(f"\nAttendance for {args.date}")
    print(f"{'Roll No':<14}{'Name':<24}{'Status':<9}Time")
    print("-" * 56)
    for roll, name, status, t in rows:
        print(f"{roll:<14}{name:<24}{status:<9}{t}")
    n_present = sum(1 for r in rows if r[2] == "Present")
    print("-" * 56)
    print(f"Present: {n_present}/{len(rows)}\n")

    if args.csv:
        EXPORT_DIR.mkdir(exist_ok=True)
        out = EXPORT_DIR / f"attendance_{args.date}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["roll_no", "name", "status", "time"])
            w.writerows(rows)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
