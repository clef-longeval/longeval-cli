import sqlite3
from pathlib import Path

db_path = Path("data/interim/core_works.sqlite")

con = sqlite3.connect(db_path)
con.execute("PRAGMA foreign_keys = ON;")

def q(sql):
    return con.execute(sql).fetchone()[0]

works = q("SELECT COUNT(*) FROM works;")
authors = q("SELECT COUNT(*) FROM authors;")
wa = q("SELECT COUNT(*) FROM work_authors;")
oai = q("SELECT COUNT(*) FROM oai_ids;")
links = q("SELECT COUNT(*) FROM links;")

missing_pub = q("""
SELECT SUM(CASE WHEN publishedDate IS NULL OR publishedDate='' THEN 1 ELSE 0 END)
FROM works;
""")

min_pub, max_pub = con.execute("""
SELECT MIN(publishedDate), MAX(publishedDate)
FROM works
WHERE publishedDate IS NOT NULL AND publishedDate != '';
""").fetchone()

orph_wa = q("""
SELECT COUNT(*) FROM work_authors wa
LEFT JOIN works w ON w.id=wa.work_id
WHERE w.id IS NULL;
""")

page_count = q("PRAGMA page_count;")
page_size = q("PRAGMA page_size;")
freelist = q("PRAGMA freelist_count;")
db_size_gb = page_count * page_size / (1024**3)

print("DB:", db_path)
print(f"works: {works:,}")
print(f"authors: {authors:,} | work_authors: {wa:,} | oai_ids: {oai:,} | links: {links:,}")
print(f"missing publishedDate: {missing_pub:,} ({missing_pub/works:.1%})")
print(f"publishedDate range: {min_pub} .. {max_pub}")
print(f"orphan work_authors: {orph_wa:,}")
print(f"db approx size: {db_size_gb:.2f} GB | freelist pages: {freelist:,}")

con.close()
