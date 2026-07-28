"""
review_unmatched.py
Quick-win: "Dataset expand karo" helper.

app.py automatically logs every query the bot could NOT confidently answer
into data/unmatched_log.csv. This script summarizes that log so you can see,
at a glance, which real questions are missing from data/faqs.csv - then you
just add them as new rows and re-run the app.

Run:
    python3 src/review_unmatched.py
"""

from collections import Counter
from pathlib import Path

import pandas as pd

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "unmatched_log.csv"


def main():
    if not LOG_PATH.exists():
        print("No unmatched queries logged yet. Ask the bot a few tricky/off-topic "
              "questions first, then re-run this script.")
        return

    df = pd.read_csv(LOG_PATH)
    print(f"Total unmatched queries logged: {len(df)}\n")

    counts = Counter(df["query"].str.lower().str.strip())
    print("Most frequent unmatched queries:")
    for query, count in counts.most_common(15):
        print(f"  [{count}x] {query}")

    print("\nTip: anything appearing 2+ times is a strong candidate for a new "
          "row in data/faqs.csv (new question + answer + category).")


if __name__ == "__main__":
    main()
