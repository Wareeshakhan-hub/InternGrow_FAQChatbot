"""
logging_utils.py
Quick-win helpers: logs unmatched queries and 👍/👎 feedback to CSV files
under data/, so you have a growing record of:
  1) real gaps in the FAQ dataset (unmatched_log.csv)
  2) which answers users found actually helpful (feedback_log.csv)

Both files are created automatically (with a header row) the first time
something is logged - no setup needed.
"""

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UNMATCHED_LOG = DATA_DIR / "unmatched_log.csv"
FEEDBACK_LOG = DATA_DIR / "feedback_log.csv"

_lock = Lock()  # Streamlit can technically process concurrent sessions


def _append_row(path: Path, header: list, row: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with _lock:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)


def log_unmatched(query: str, mode: str, score: float):
    """Called whenever the bot fails to find a confident match."""
    _append_row(
        UNMATCHED_LOG,
        header=["timestamp", "query", "mode", "best_score"],
        row=[datetime.now().isoformat(timespec="seconds"), query, mode, round(float(score), 3)],
    )


def log_feedback(query: str, mode: str, matched_question: str, answer: str,
                  score: float, feedback: str):
    """Called when the user taps 👍 or 👎 on an answer."""
    _append_row(
        FEEDBACK_LOG,
        header=["timestamp", "query", "mode", "matched_question", "answer", "score", "feedback"],
        row=[
            datetime.now().isoformat(timespec="seconds"),
            query,
            mode,
            matched_question or "",
            answer,
            round(float(score), 3),
            feedback,
        ],
    )
