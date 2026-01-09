import numpy as np
import psycopg2

# The SQL query to fetch training data
# We unnest the SERP to get the position (idx) and the doc_id
QUERY = """
SELECT
    q.query_id,
    s.search_id,
    u.doc_id,
    u.pos,
    CASE WHEN t.doc_id IS NOT NULL THEN 1.0 ELSE 0.0 END as clicked
FROM search_logs_normalized s
JOIN queries q ON s.query_id = q.query_id
CROSS JOIN LATERAL unnest(s.serp) WITH ORDINALITY AS u(doc_id, pos)
LEFT JOIN tracker_logs t ON s.search_id = t.search_id AND t.doc_id = u.doc_id
LIMIT 20
"""
conn_string = "postgresql://postgres:postgres@localhost:5454/postgres"

def postgres_clax_loader(conn_string, batch_size=1024):
    conn = psycopg2.connect(conn_string)
    # Named cursor creates a server-side streaming cursor
    with conn.cursor(name="clax_stream") as cur:
        cur.execute(QUERY)

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            print(f"Fetched {len(rows)} rows from database.")
            print(rows[0])
            # Extract data into numpy arrays for JAX
            # CLAX requires float32 for clicks and int32 for IDs/Positions
            batch = {
                "query_doc_ids": np.array([r[0] + r[2] for r in rows], dtype=np.int32),
                "positions": np.array([r[3] for r in rows], dtype=np.int32),
                "clicks": np.array([r[4] for r in rows], dtype=np.float32),
                "mask": np.ones(len(rows), dtype=bool),
            }
            yield batch


for i in postgres_clax_loader(conn_string, batch_size=18):
    print(i)
    break