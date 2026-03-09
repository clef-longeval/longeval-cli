"""
script to export document metadata from a SQLite database to jsonl files.

Given one or more files containing document IDs (one ID per line), the script:
- queries the SQLite DB for the requested document IDs
- reconstructs the document JSON in the expected format
- writes JSONL files with a maximum of N documents per file (default: 1000)

Output format per line:
{"id": "...", "title": ..., "abstract": ..., "authors": [{"name": ...}, ...], ...}
"""

import json
import sqlite3
from pathlib import Path
import hydra
from omegaconf import DictConfig
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_doc_ids(file_path):
    doc_ids = []
    seen = set()
    with open(file_path, 'r') as f:
        for line in f:
            doc_id = line.strip()
            if not doc_id:
                continue
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
    return doc_ids


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def init_db_connection(db_path):
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def fetch_works(con, doc_ids, snapshot_published_date_before):
    """Fetch work records and filter them.

    Returns a tuple ``(docs, stats)`` where ``docs`` is the usual dict of
    work data and ``stats`` is another dict containing counters for the two
    reasons we skip a record:
      * ``too_new``: published after the snapshot cutoff
      * ``short_abstract``: missing or very short abstract (<100 chars)
    """
    if not doc_ids:
        return {}, {"too_new": 0, "short_abstract": 0}

    placeholders = ",".join(["?"] * len(doc_ids))
    rows = con.execute(
        f"""
        SELECT id, title, abstract, createdDate, doi, arxivId, pubmedId, magId,
               publishedDate, updatedDate, fullText
        FROM works
        WHERE id IN ({placeholders})
        """,
        doc_ids,
    ).fetchall()

    docs = {}
    stats = {"too_new": 0, "short_abstract": 0}
    for r in rows:
        (wid, title, abstract, createdDate, doi, arxivId, pubmedId, magId,
         publishedDate, updatedDate, fullText) = r
        
        work_dt = datetime.strptime(publishedDate, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc) if publishedDate else None
        snapshot_dt = datetime.strptime(snapshot_published_date_before, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if snapshot_published_date_before else None
        if work_dt and work_dt > snapshot_dt:
            stats["too_new"] += 1
            continue
        if abstract is None or len(abstract.strip()) < 100:  # Filter out documents with missing or very short abstracts
            stats["short_abstract"] += 1
            continue

        wid = str(wid)
        docs[wid] = {
            "id": wid,
            "title": title,
            "abstract": abstract,
            "authors": [],
            "createdDate": createdDate,
            "doi": doi,
            "arxivId": arxivId,
            "pubmedId": pubmedId,
            "magId": magId,
            "oaiIds": [],
            "links": [],
            "publishedDate": publishedDate,
            "updatedDate": updatedDate,
            "fullText": fullText,
        }
    return docs, stats


def fetch_authors(con, work_ids):
    if not work_ids:
        return {}

    placeholders = ",".join(["?"] * len(work_ids))
    rows = con.execute(
        f"""
        SELECT wa.work_id, wa.author_pos, a.name
        FROM work_authors wa
        JOIN authors a ON a.author_id = wa.author_id
        WHERE wa.work_id IN ({placeholders})
        ORDER BY wa.work_id, wa.author_pos
        """,
        work_ids,
    ).fetchall()

    work_id_to_authors = {}
    for work_id, author_pos, name in rows:
        work_id = str(work_id)
        if work_id not in work_id_to_authors:
            work_id_to_authors[work_id] = []
        work_id_to_authors[work_id].append({"name": name})
    return work_id_to_authors


def fetch_oai_ids(con, work_ids):
    if not work_ids:
        return {}

    placeholders = ",".join(["?"] * len(work_ids))
    rows = con.execute(
        f"""
        SELECT work_id, oai_id
        FROM oai_ids
        WHERE work_id IN ({placeholders})
        ORDER BY work_id
        """,
        work_ids,
    ).fetchall()

    work_id_to_oai = {}
    for work_id, oai_id in rows:
        work_id = str(work_id)
        if work_id not in work_id_to_oai:
            work_id_to_oai[work_id] = []
        work_id_to_oai[work_id].append(oai_id)
    return work_id_to_oai


def fetch_links(con, work_ids):
    if not work_ids:
        return {}

    placeholders = ",".join(["?"] * len(work_ids))
    rows = con.execute(
        f"""
        SELECT work_id, type, url
        FROM links
        WHERE work_id IN ({placeholders})
        ORDER BY work_id
        """,
        work_ids,
    ).fetchall()

    work_id_to_links = {}
    for work_id, link_type, url in rows:
        work_id = str(work_id)
        if work_id not in work_id_to_links:
            work_id_to_links[work_id] = []
        work_id_to_links[work_id].append({"type": link_type, "url": url})
    return work_id_to_links


def attach_children(docs, authors_map, oai_map, links_map):
    for work_id, doc in docs.items():
        doc["authors"] = authors_map.get(work_id, [])
        doc["oaiIds"] = oai_map.get(work_id, [])
        doc["links"] = links_map.get(work_id, [])


def process_batch(batch_doc_ids, db_path, snapshot_published_date_before):
    con = init_db_connection(db_path)
    docs, stats = fetch_works(con, batch_doc_ids, snapshot_published_date_before)
    work_ids = sorted(docs.keys())

    authors_map = fetch_authors(con, work_ids)
    oai_map = fetch_oai_ids(con, work_ids)
    links_map = fetch_links(con, work_ids)

    attach_children(docs, authors_map, oai_map, links_map)

    con.close()
    # return both docs and stats so caller can accumulate
    return docs, stats


def determine_current_output_file(output_folder: Path, output_max_file_size: int = 1000):
    output_file_i = 0
    found_current_output_file = False
    current_output_file_document_num = 0
    while not found_current_output_file:
        current_output_file = output_folder.joinpath(f"documents_{output_file_i:>06}.jsonl")
        if current_output_file.is_file():
            next_output_file = output_folder.joinpath(f"documents_{output_file_i+1:>06}.jsonl")
            if next_output_file.is_file():
                output_file_i += 1
            else:
                counted_lines = 0
                with open(current_output_file, "r", encoding="utf-8") as fp:
                    for _ in fp:
                        counted_lines += 1
                if counted_lines < output_max_file_size:
                    current_output_file_document_num = counted_lines
                    found_current_output_file = True
                else:
                    output_file_i += 1
        else:
            found_current_output_file = True
    return output_file_i, current_output_file, current_output_file_document_num


def export_docs_to_jsonl(db_path, doc_ids, snapshot_published_date_before, output_folder: Path, output_max_file_size: int = 1000, sql_batch_size: int = 2000,
                         found_doc_ids_file: Path = None, missing_doc_ids_file: Path = None):
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file_i, current_output_file, current_output_file_document_num = determine_current_output_file(
        output_folder=output_folder,
        output_max_file_size=output_max_file_size
    )

    if current_output_file_document_num > 0:
        fp = open(current_output_file, "a", encoding="utf-8")
    else:
        fp = open(current_output_file, "w", encoding="utf-8")

    found_fp = open(found_doc_ids_file, "w", encoding="utf-8") if found_doc_ids_file else None
    missing_fp = open(missing_doc_ids_file, "w", encoding="utf-8") if missing_doc_ids_file else None

    # Process all batches in parallel
    all_docs = {}
    total_stats = {"too_new": 0, "short_abstract": 0}
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_batch, batch, str(db_path), snapshot_published_date_before)
            for batch in chunk_list(doc_ids, sql_batch_size)
        ]
        for future in as_completed(futures):
            batch_docs, batch_stats = future.result()
            all_docs.update(batch_docs)
            total_stats["too_new"] += batch_stats["too_new"]
            total_stats["short_abstract"] += batch_stats["short_abstract"]

    found = 0
    missing = 0

    for did in doc_ids:
        did = str(did)
        if did not in all_docs:
            missing += 1
            if missing_fp:
                missing_fp.write(f"{did}\n")
            continue

        if found_fp:
            found_fp.write(f"{did}\n")

        fp.write(json.dumps(all_docs[did], ensure_ascii=False) + "\n")
        found += 1
        current_output_file_document_num += 1

        if current_output_file_document_num >= output_max_file_size:
            fp.close()
            output_file_i += 1
            current_output_file = output_folder.joinpath(f"documents_{output_file_i:>06}.jsonl")
            fp = open(current_output_file, "w", encoding="utf-8")
            current_output_file_document_num = 0

    fp.close()
    if found_fp:
        found_fp.close()
    if missing_fp:
        missing_fp.close()

    # report ignore counts
    print(
        f"Export done | Found: {found} | Missing: {missing} | "
        f"Too-new ignored: {total_stats['too_new']} | "
        f"Short-abstract ignored: {total_stats['short_abstract']} | "
        f"Output folder: {output_folder}"
    )



@hydra.main(version_base=None, config_path="../../config", config_name=None)
def main(cfg: DictConfig):
    """
    Exports documents from a SQLite database into jsonl files PER COLLECTION.

    Args (expected in cfg):
        db_path (str): path to sqlite db
        doc_ids_files (list[str]): each file is one collection (one doc id per line)
        output_folder (str): root folder; each collection gets its own subfolder
        output_max_file_size (int): max docs per jsonl file (default: 1000)
        sql_batch_size (int): ids per SQL chunk
    """
    db_path = Path(cfg.db_path)
    snapshots = cfg.snapshots
    selected_docids_persnapshot_dir = Path(cfg.selected_docids_persnapshot_dir)
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_max_file_size = int(getattr(cfg, "output_max_file_size", 1000))
    sql_batch_size = int(getattr(cfg, "sql_batch_size", 2000))

    for snapshot in snapshots:
        snapshot_name = snapshot.name
        snapshot_published_date_before = snapshot.published_date_before
        doc_ids_file = selected_docids_persnapshot_dir / f"doc_ids-{snapshot_name}.txt"

        collection_output_folder = output_folder / snapshot_name
        collection_output_folder.mkdir(parents=True, exist_ok=True)

        doc_ids = read_doc_ids(str(doc_ids_file))
        print(f"\nCollection: {snapshot_name} | Loaded {len(doc_ids)} doc ids from {doc_ids_file}")

        found_doc_ids_file = collection_output_folder / "found_doc_ids.txt"
        missing_doc_ids_file = collection_output_folder / "missing_doc_ids.txt"

        export_docs_to_jsonl(
            db_path=db_path,
            doc_ids=doc_ids,
            snapshot_published_date_before=snapshot_published_date_before,
            output_folder=collection_output_folder,
            output_max_file_size=output_max_file_size,
            sql_batch_size=sql_batch_size,
            found_doc_ids_file=found_doc_ids_file,
            missing_doc_ids_file=missing_doc_ids_file
        )



if __name__ == "__main__":
    main()
