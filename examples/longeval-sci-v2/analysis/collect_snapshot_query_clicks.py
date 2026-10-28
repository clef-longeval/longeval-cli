"""Export stored per-snapshot query click counts without recalculating them."""

import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATABASE = HERE.parent / "longeval.sqlite3"
OUTPUT = HERE / "output" / "snapshot_query_clicks.csv"

SQL = """
SELECT snapshot, query_id, CAST(value AS INTEGER) AS clicks
FROM query_classifications
WHERE classifier = 'clicks'
  AND snapshot IN ('snapshot-1', 'snapshot-2', 'snapshot-3')
ORDER BY snapshot, query_id;
"""


def main():
    print(f"Exporting stored snapshot query clicks from {DATABASE} …", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    try:
        with sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True) as db:
            cursor = db.execute(SQL)
            with temporary.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["snapshot", "query_id", "clicks"])
                writer.writerows(cursor)
        temporary.replace(OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
