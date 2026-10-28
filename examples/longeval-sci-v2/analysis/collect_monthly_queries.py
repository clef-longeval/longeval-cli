"""Monthly search counts per query with stored text-classifier labels."""

import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATABASE = HERE.parent / "longeval.sqlite3"
OUTPUT = HERE / "output" / "monthly_queries.csv"

SQL = """
WITH monthly AS (
    SELECT strftime('%Y-%m', q_date) AS month,
           query_id,
           COUNT(*) AS searches
    FROM results
    WHERE strftime('%Y-%m', q_date) IS NOT NULL
    GROUP BY month, query_id
), labels AS (
    -- Text labels are independent of time. Prefer global verdicts, falling
    -- back to snapshot-3, where the existing notebook stored its run.
    SELECT query_id, classifier,
           COALESCE(MAX(CASE WHEN snapshot = '' THEN label END),
                    MAX(CASE WHEN snapshot = 'snapshot-3' THEN label END)) AS label
    FROM query_classifications
    WHERE snapshot IN ('', 'snapshot-3')
      AND classifier IN ('is-structured', 'is-boolean', 'has-doi', 'word-count')
    GROUP BY query_id, classifier
), query_labels AS (
    SELECT query_id,
           MAX(CASE WHEN classifier = 'is-structured' THEN label END) AS is_structured,
           MAX(CASE WHEN classifier = 'is-boolean' THEN label END) AS is_boolean,
           MAX(CASE WHEN classifier = 'has-doi' THEN label END) AS has_doi,
           CAST(MAX(CASE WHEN classifier = 'word-count' THEN label END)
                AS INTEGER) AS word_count
    FROM labels
    GROUP BY query_id
)
SELECT m.month, m.query_id, m.searches,
       l.is_structured, l.is_boolean, l.has_doi, l.word_count
FROM monthly AS m
LEFT JOIN query_labels AS l ON l.query_id = m.query_id
ORDER BY m.month, m.query_id;
"""


def main():
    print(f"Collecting monthly queries from {DATABASE} …", flush=True)
    with sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True) as db:
        cursor = db.execute(SQL)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(column[0] for column in cursor.description)
            writer.writerows(cursor)
    print(f"Saved monthly queries to {OUTPUT}")


if __name__ == "__main__":
    main()
