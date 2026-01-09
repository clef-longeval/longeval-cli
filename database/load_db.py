import json
from datetime import datetime

import hydra
import psycopg2
from omegaconf import DictConfig
from psycopg2.extras import execute_values

from src.io import traverse_logs

BATCH_SIZE = 100_000


def sanitize(value):
    """Removes NUL characters that Postgres cannot store in text fields."""
    if isinstance(value, str):
        return value.replace("\0", "").replace("\u0000", "")
    return value


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    conn = psycopg2.connect(**cfg.db)
    cur = conn.cursor()

    # Create tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_logs (
            id SERIAL PRIMARY KEY, 
            log_date TIMESTAMP, 
            search_id TEXT, 
            query TEXT, 
            uid TEXT, 
            serp BIGINT[]
        );
        CREATE TABLE IF NOT EXISTS tracker_logs (
            id SERIAL PRIMARY KEY, 
            log_date TIMESTAMP, 
            search_id TEXT, 
            doc_id TEXT,
            entry_type TEXT, 
            uid TEXT
        );
    """)
    conn.commit()

    batch_search = []
    batch_tracker = []

    print("Starting import...")

    logs_iter = traverse_logs(logs_path=cfg.paths.logs)

    for line in logs_iter:
        try:
            data = json.loads(line)
            dt = datetime.fromtimestamp(data.get("date"))

            # TYPE 1: Search Record
            if "search_id" in data:
                batch_search.append(
                    (
                        dt,
                        sanitize(data.get("search_id")),
                        sanitize(data.get("query")),
                        sanitize(data.get("uid")),
                        data.get("serp"),
                    )
                )

            # TYPE 2: Tracker Record
            elif "trackId" in data:
                track_id = data.get("trackId")
                search_id, doc_id = track_id.split("-")
                batch_tracker.append(
                    (
                        dt,
                        sanitize(search_id),
                        sanitize(doc_id),
                        sanitize(data.get("type")),
                        sanitize(data.get("uid")),
                    )
                )

            # Periodic Bulk Insert
            if len(batch_search) >= BATCH_SIZE:
                execute_values(
                    cur,
                    "INSERT INTO search_logs (log_date, search_id, query, uid, serp) VALUES %s",
                    batch_search,
                )
                batch_search = []
                conn.commit()

            if len(batch_tracker) >= BATCH_SIZE:
                execute_values(
                    cur,
                    "INSERT INTO tracker_logs (log_date, search_id, doc_id, entry_type, uid) VALUES %s",
                    batch_tracker,
                )
                batch_tracker = []
                conn.commit()

        except (json.JSONDecodeError, TypeError, ValueError):
            print("Warning: could not decode JSON line or invalid data:", line)

    # Flush remaining records
    if batch_search:
        execute_values(
            cur,
            "INSERT INTO search_logs (log_date, search_id, query, uid, serp) VALUES %s",
            batch_search,
        )
    if batch_tracker:
        execute_values(
            cur,
            "INSERT INTO tracker_logs (log_date, search_id, doc_id, entry_type, uid) VALUES %s",
            batch_tracker,
        )

    conn.commit()

    print("Import Complete.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
