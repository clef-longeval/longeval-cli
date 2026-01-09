import hydra
import psycopg2
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    conn = psycopg2.connect(**cfg.db)
    cur = conn.cursor()
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
            track_id TEXT, 
            entry_type TEXT, 
            uid TEXT
        );
    """)

    conn.commit()


if __name__ == "__main__":
    main()
