import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig
import requests
import json
import time
import concurrent.futures
from typing import List
from tqdm import tqdm
from pathlib import Path
import os
import sqlite3
from docs_db_utils import *

def determine_current_output_file(output_folder: str, output_max_file_size:int = 10000):
    output_file_i = 0
    found_current_output_file = False
    current_output_file_document_num = 0
    while not found_current_output_file:
        current_output_file = Path(output_folder).joinpath(f"documents_{output_file_i:>06}.jsonl")
        if current_output_file.is_file():
            next_output_file = Path(output_folder).joinpath(f"documents_{output_file_i+1:>06}.jsonl")
            if next_output_file.is_file(): # CONTINUE, another file exist after current file
                output_file_i += 1
            else:
                # there is no next file, so need to check current file
                # need to check how many documents are already in the file
                counted_lines = 0
                with open(current_output_file, "r") as fp:
                    for line in enumerate(fp):
                        counted_lines += 1
                if counted_lines < output_max_file_size: # BREAK, current file still has space, this will be the most likely exit to this loop
                    current_output_file_document_num = counted_lines
                    found_current_output_file = True
                else: # CONTINUE, current file has no space left, i.e. contains EXACTLY output_max_file_size documents
                    output_file_i += 1
        else: # BREAK, current file does not exist
            # only happens if no file exists yet or last file contains exactly max file size documents
            found_current_output_file = True
    return output_file_i, current_output_file, current_output_file_document_num



def fetch_documents_parallel(document_ids: List[int], api_key: str, output_folder: str,
                             batch_size:int = 10, max_tries: int = 5, max_workers: int = 5, full_text_max_size=400000):
    """
    Fetches documents in parallel using the `get_work_batch` function.
    Writes them into sequentially numbered files in the output folder.

    Args:
        document_ids (List[int]): A list of document IDs to fetch.
        api_key (str): key for accessing the api.
        output_folder (str): location of the folder to store the documents in json lines format.
        doc_ids_tmp_file (str): location of the file to store the progress of which ids have been downloaded
        output_max_file_size (int): how many documents are written in one output file until a new one is created
        batch_size (int): how many documents are requested with one search query
        max_tries (int): how many times a query is repeated in case of time out
        max_workers (int): Maximum number of parallel workers.
        full_text_max_size (int): how many characters the full text field of a document can have until it is ignored

    """
    #results = {}

    # Split document IDs into batches
    def chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    document_id_batches = list(chunk_list(document_ids, batch_size))

    def fetch_and_store_batch(doc_ids):
        try:
            return get_work_batch(doc_ids, api_key=api_key, batch_size=batch_size, max_tries=max_tries)
        except Exception as e:
            print(f"Failed to fetch document {doc_ids}: {e}")
            return None


    commit_every = 200
    pending = 0
    
    db_path = Path(output_folder).joinpath("core_works.sqlite")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    init_db(con)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(fetch_and_store_batch, doc_ids): doc_ids for doc_ids in document_id_batches}
        #futures = [executor.submit(fetch_and_store_batch, batch) for batch in document_id_batches]
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(document_id_batches), desc="Processing document batches"):
        #for future in tqdm(concurrent.futures.as_completed(futures), total=len(document_ids), desc="Processing document batches"):
            doc_ids = future_to_batch[future]
            try:
                document_batch = future.result()["results"]
                print(f"Fetched batch of {len(document_batch)} documents for doc ids {doc_ids}")
                p_docs = process_document_batch(document_batch, full_text_max_size=full_text_max_size)
                # for each doc_id in the batch, check if doc is available and save it if it is
                for doc_id in doc_ids:
                    if doc_id in p_docs:
                        p_doc = p_docs[doc_id]
                        upsert_work(con, p_doc)
                        #results[doc_id] = True
                        # if the current output file has reached its limit, close it and start with the next
                        pending += 1
                        if pending >= commit_every:
                            con.commit()
                            pending = 0
                    else:
                        print(f"Did not find {doc_id} in {len(document_batch)} results")
                        log_status(con, doc_id, "not_found", f"Did not find {doc_id} in {len(document_batch)} results")
                        #results[doc_id] = False
                con.commit()
            except Exception as e:
                print(f"Error processing documents {doc_ids}: {e}")
                for doc_id in doc_ids:
                    log_status(con, doc_id, "error", f"Error processing documents {doc_ids}: {e}")
                    #results[doc_id] = False
    con.close()


def process_document_batch(document_batch, full_text_max_size=400000):
    p_doc_batch = {}
    for doc in document_batch:
        p_doc = process_document(doc, full_text_max_size=full_text_max_size)
        p_doc_batch[p_doc["id"]] = p_doc
    return p_doc_batch


def process_document(document, full_text_max_size=400000):
    processed_document = {"id": document["id"]}
    processed_document["title"] = document.get("title")
    processed_document["abstract"] = document.get("abstract")
    processed_document["authors"] = document.get("authors")
    processed_document["createdDate"] = document.get("createdDate")
    processed_document["doi"] = document.get("doi")
    processed_document["arxivId"] = document.get("arxivId")
    processed_document["pubmedId"] = document.get("pubmedId")
    processed_document["magId"] = document.get("magId")
    processed_document["oaiIds"] = document.get("oaiIds")
    processed_document["links"] = document.get("links")
    processed_document["publishedDate"] = document.get("publishedDate")
    processed_document["updatedDate"] = document.get("updatedDate")
    if document.get("fullText") and len(document["fullText"]) > full_text_max_size:
        processed_document["fullText"] = None
    else:
        processed_document["fullText"] = document.get("fullText")
    return processed_document


def build_query(ids):
    query_template = "id:{}"

    def apply_template(id):
        return query_template.format(id)

    return " OR ".join(map(apply_template, ids))



def get_work_batch(work_ids, api_key, batch_size=10, max_tries=5):
    url_template = "https://api.core.ac.uk/v3/search/works?q={}&limit={}"
    url = url_template.format(build_query(work_ids), batch_size)

    #print(url)

    payload={}
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    succuess = False
    i = 0
    while not succuess and i < max_tries:
        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200:
            doc_batch = response.json()
            succuess = True
        if response.status_code == 429:
            sleep_time = 2**(i+3)
            print(f"Sleeping {sleep_time} - too many requests: {url}")
            time.sleep(sleep_time)
        i += 1
    if not succuess:
        print(f"DL failed after {max_tries} tries: {url}")
        return None
    return doc_batch


def read_search_log(search_log_files):
    print(f"Reading search log files: {search_log_files}")
    doc_id_success = {}
    for search_log_file in search_log_files:
        with open(search_log_file, "r") as fp:
            for line in fp:
                entry = json.loads(line)
                if "trackId" in entry:
                    track_id_data = entry["trackId"].split("-")
                    if len(track_id_data) != 2: # some data might be faulty
                        continue
                    search_id, doc_id = track_id_data
                    try:
                        doc_id = int(doc_id)
                    except:
                        print("Invalid data", entry["trackId"])
                        continue
                    doc_id = str(doc_id)
                    doc_id_success[doc_id] = False
                elif "search_id" in entry:
                    for doc_id in entry["serp"]:
                        doc_id = str(doc_id)
                        doc_id_success[doc_id] = False
                else:
                    assert False, f"Unknown data entry format: {entry}"
    print(f"Extracted {len(doc_id_success)} doc ids from log files {search_log_files}")
    return doc_id_success


def read_doc_id_success(doc_ids_tmp_file):
    """
    Read doc_id_success from file, converting 0/1 to bool
    """
    doc_id_success = {}
    with open(doc_ids_tmp_file, "r") as fp:
        for line in fp:
            doc_id, success = line.split()
            doc_id_success[str(doc_id)] = bool(int(success))
    return doc_id_success


def write_doc_id_success(doc_id_success, doc_ids_tmp_file):
    """
    Write doc_id_success to file, converting bool to int
    """
    with open(doc_ids_tmp_file, "w") as fp:
        for doc_id, success in doc_id_success.items():
            fp.write(f"{doc_id} {int(success)}\n")



@hydra.main(version_base=None, config_path="../../config", config_name=None)
def main(cfg: DictConfig):
    """
    Fetches documents in parallel.
    Writes them into sequentially numbered files in the output folder.

    Args:
        secrets_file (str): file containing the CORE API key.
        output_folder (str): folder that will contain jsonl files to store documents and meta data in, and doc_ids_tmp_file for storing progress of fetching documents
        search_log_files (str): list of CORE log files to scrape document IDs from
        doc_ids_tmp_file (str): location of the file to store the progress of which ids have been downloaded
        output_max_file_size (int): how many documents are written in one output file until a new one is created
        batch_size (int): how many documents are requested with one search query
        max_tries (int): how many times a query is repeated in case of time out
        max_workers (int): Maximum number of parallel workers.
        full_text_max_size (int): how many characters the full text field of a document can have until it is ignored
        use_random_negative_sampling (bool): if true, will attempt to download a random document for each known document by adding +1 to its document ID
        
    example usage:
        python collect.py \
            --config-name core_data_collection \
            secrets_file=secrets.json \
            output_folder=data/core_data/documents/ \
            "search_log_files=['./data/sample_log.jsonl']" \
            output_max_file_size=10000 \
            batch_size=10 \
            max_tries=5 \
            max_workers=5 \
            full_text_max_size=400000 \
            use_random_negative_sampling=False
    """
    api_key = cfg.api_key
    doc_ids_path = cfg.doc_ids_path 
    output_folder = cfg.output_folder
    batch_size = cfg.batch_size
    max_tries = cfg.max_tries
    max_workers = cfg.max_workers
    full_text_max_size = cfg.full_text_max_size
    
    with open(doc_ids_path, "r") as fp:
        doc_ids = [int(line.strip()) for line in fp]

    fetch_documents_parallel(document_ids=doc_ids, 
                            api_key=api_key, 
                            output_folder=output_folder, 
                            batch_size=batch_size,
                            max_tries=max_tries, 
                            max_workers=max_workers, 
                            full_text_max_size=full_text_max_size)


if __name__ == "__main__":
    main()