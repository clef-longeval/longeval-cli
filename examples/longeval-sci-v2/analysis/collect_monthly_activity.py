"""Monthly clicks by type, before and after excluding late clicks."""

import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATABASE = HERE.parent / "longeval.sqlite3"
OUTPUT = HERE / "output" / "monthly_clicks.csv"

SQL = """
-- Attribute clicks to the originating search month (UTC).
-- Use the stored click-delay verdicts; NOT EXISTS avoids duplicating click rows.
SELECT
    strftime('%Y-%m', r.q_date) AS month,
    i.type,
    COUNT(*) AS clicks,
    SUM(CASE WHEN NOT EXISTS (
        SELECT 1
        FROM interaction_classifications AS c
        WHERE c.interaction_id = i.id
          AND c.classifier = 'click-delay'
          AND c.label = 'late'
          AND c.snapshot IN ('', 'snapshot-3')
    ) THEN 1 ELSE 0 END) AS valid_clicks
FROM interactions AS i
JOIN results AS r ON r.id = i.result_id
WHERE strftime('%Y-%m', r.q_date) IS NOT NULL
GROUP BY month, i.type
ORDER BY month, i.type;
"""


def main():
    print(f"Collecting monthly clicks from {DATABASE} …", flush=True)
    with sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True) as db:
        cursor = db.execute(SQL)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
