# db.py
import sqlite3
from typing import Any, Dict, Optional

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS works (
  id            TEXT PRIMARY KEY,
  title         TEXT,
  abstract      TEXT,
  createdDate   TEXT,
  updatedDate   TEXT,
  publishedDate TEXT,
  doi           TEXT,
  arxivId       TEXT,
  pubmedId      TEXT,
  magId         TEXT,
  fullText      TEXT
);

CREATE TABLE IF NOT EXISTS authors (
  author_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name      TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS work_authors (
  work_id    TEXT NOT NULL,
  author_id  INTEGER NOT NULL,
  author_pos INTEGER NOT NULL,
  PRIMARY KEY (work_id, author_id, author_pos),
  FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
  FOREIGN KEY (author_id) REFERENCES authors(author_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS oai_ids (
  work_id TEXT NOT NULL,
  oai_id  TEXT NOT NULL,
  PRIMARY KEY (work_id, oai_id),
  FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS links (
  work_id TEXT NOT NULL,
  type    TEXT,
  url     TEXT NOT NULL,
  PRIMARY KEY (work_id, url),
  FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingest_log (
  work_id TEXT PRIMARY KEY,
  status  TEXT NOT NULL,         -- ok | not_found | error
  message TEXT,
  ts      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_works_publishedDate ON works(publishedDate);
CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi);
CREATE INDEX IF NOT EXISTS idx_links_type ON links(type);
CREATE INDEX IF NOT EXISTS idx_ingest_log_status ON ingest_log(status);
"""

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_SQL)
    con.commit()

def _get_str(d: Dict[str, Any], key: str) -> Optional[str]:
    v = d.get(key)
    return None if v is None else str(v)

def upsert_work(con: sqlite3.Connection, work: Dict[str, Any]) -> None:
    work_id = str(work["id"])

    con.execute(
        """
        INSERT INTO works (
          id, title, abstract, createdDate, updatedDate, publishedDate,
          doi, arxivId, pubmedId, magId, fullText
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          title=excluded.title,
          abstract=excluded.abstract,
          createdDate=excluded.createdDate,
          updatedDate=excluded.updatedDate,
          publishedDate=excluded.publishedDate,
          doi=excluded.doi,
          arxivId=excluded.arxivId,
          pubmedId=excluded.pubmedId,
          magId=excluded.magId,
          fullText=excluded.fullText
        """,
        (
            work_id,
            _get_str(work, "title"),
            _get_str(work, "abstract"),
            _get_str(work, "createdDate"),
            _get_str(work, "updatedDate"),
            _get_str(work, "publishedDate"),
            _get_str(work, "doi"),
            _get_str(work, "arxivId"),
            _get_str(work, "pubmedId"),
            _get_str(work, "magId"),
            _get_str(work, "fullText"),
        ),
    )

    # Refresh children on each ingest (safe for updates)
    con.execute("DELETE FROM work_authors WHERE work_id = ?", (work_id,))
    con.execute("DELETE FROM oai_ids WHERE work_id = ?", (work_id,))
    con.execute("DELETE FROM links WHERE work_id = ?", (work_id,))

    # Authors
    authors = work.get("authors") or []
    for pos, a in enumerate(authors):
        name = (a.get("name") or "").strip()
        if not name:
            continue

        con.execute(
            "INSERT INTO authors(name) VALUES (?) ON CONFLICT(name) DO NOTHING",
            (name,),
        )
        author_id = con.execute(
            "SELECT author_id FROM authors WHERE name = ?",
            (name,),
        ).fetchone()[0]

        con.execute(
            "INSERT OR IGNORE INTO work_authors(work_id, author_id, author_pos) VALUES (?, ?, ?)",
            (work_id, author_id, pos),
        )

    # OAI IDs
    for oai in (work.get("oaiIds") or []):
        oai = str(oai).strip()
        if oai:
            con.execute(
                "INSERT OR IGNORE INTO oai_ids(work_id, oai_id) VALUES (?, ?)",
                (work_id, oai),
            )

    # Links
    for link in (work.get("links") or []):
        url = (link.get("url") or "").strip()
        if not url:
            continue
        ltype = link.get("type") or None
        con.execute(
            "INSERT OR IGNORE INTO links(work_id, type, url) VALUES (?, ?, ?)",
            (work_id, ltype, url),
        )

def work_exists(con: sqlite3.Connection, work_id: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM ingest_log WHERE work_id=? AND status='ok'",
        (work_id,),
    ).fetchone()
    return row is not None

def log_status(con: sqlite3.Connection, work_id: str, status: str, message: Optional[str] = None) -> None:
    con.execute(
        """
        INSERT INTO ingest_log(work_id, status, message)
        VALUES (?, ?, ?)
        ON CONFLICT(work_id) DO UPDATE SET
          status=excluded.status,
          message=excluded.message,
          ts=datetime('now')
        """,
        (work_id, status, message),
    )
