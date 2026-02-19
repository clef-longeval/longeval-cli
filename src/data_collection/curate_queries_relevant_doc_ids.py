"""
collecting the relevant document ids for the different queries per snapshot 
"""
import hydra
from omegaconf import DictConfig
import requests
import time
from pathlib import Path

import urllib.parse
from collections.abc import Iterable
import csv
import json



def execute_request(url, headers, payload, max_tries=5):
    succuess = False
    i = 0
    while not succuess and i < max_tries:
        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200:
            result = response.json()
            succuess = True
        elif response.status_code == 429:
            retry_after = response.headers.get('X-RateLimit-Retry-After')
            print(f"Rate limit exceeded. Retry after: {retry_after} seconds. URL: {url}")
            if retry_after:
                sleep_time = int(retry_after)
            else:
                sleep_time = 2**(i+3)
            print(f"Sleeping {sleep_time}s - too many requests: {url}")
            time.sleep(sleep_time)
        else:
            print(f"Request failed with status code {response.status_code}. URL: {url}")
            time.sleep(2**(i+3))
        i += 1
    if not succuess:
        print(f"DL failed after {max_tries} tries: {url}")
        return None
    return result



def retrieve_relevant_doc_ids(api_base_url, search_endpoint, api_key, query, limit, max_tries=5):
    url = f'{api_base_url}{search_endpoint}?q={urllib.parse.quote(query)}&limit={limit}&offset=0'

    payload={}
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    result = execute_request(url, headers, payload, max_tries)
    if result is None:
        return None 
    doc_ids = [item['id'] for item in result.get('results', [])]
    return doc_ids

def curate_snapshot_doc_ids(cfg: DictConfig):
    api_base_url = cfg.api_base_url
    search_endpoint = cfg.search_endpoint
    api_key = cfg.api_key
    batch_size = cfg.get('batch_size', 10)
    max_tries = cfg.get('max_tries', 5)
    snapshots = cfg.snapshots
    query_per_snapshot_dir = Path(cfg.query_per_snapshot_dir)
    query_relevant_docids_persnapshot_dir = Path(cfg.query_relevant_docids_persnapshot_dir)
    query_relevant_docids_persnapshot_dir.mkdir(parents=True, exist_ok=True)
    
    

    for snapshot in snapshots:
        snapshot_name = snapshot.name

        print(f"Curating doc IDs for snapshot: {snapshot_name}")

        query_file = query_per_snapshot_dir / f"query-{snapshot_name}.tsv"
        if not query_file.exists():
            print(f"Query file not found for snapshot {snapshot_name}: {query_file}")
            continue
        
        doc_ids = set()
        query_doc_ids_map = []
        with open(query_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header
            queries = [(row[0], row[1]) for row in reader if row]
            for query in queries:
                query_id, query_text = query
                print(f"Retrieving relevant doc IDs for query id: {query_id} and query: {query_text} for snapshot: {snapshot_name}")
                query_doc_ids = retrieve_relevant_doc_ids(api_base_url, search_endpoint, api_key, query_text, limit=1000, max_tries=max_tries)
                query_doc_ids_map.append({"query_id": query_id, "query": query_text, "doc_ids": [] if query_doc_ids is None else query_doc_ids})
                if query_doc_ids is not None:
                    doc_ids.update(set(query_doc_ids))
                print(f"Retrieved {len(query_doc_ids) if query_doc_ids is not None else 0} doc IDs for query id: {query_id} and query: {query_text} for snapshot: {snapshot_name}")
        output_file = query_relevant_docids_persnapshot_dir / f"doc_ids-{snapshot_name}.txt"
        save_doc_ids(doc_ids, output_file)
        print(f"Saved {len(doc_ids)} doc IDs to {output_file}")

        # Save query-specific doc IDs to JSON
        output_file = query_relevant_docids_persnapshot_dir / f"query_doc_ids-{snapshot_name}.json"
        with open(output_file, 'w') as f:
            json.dump(query_doc_ids_map, f, indent=2)
        print(f"Saved query-document IDs mapping to {output_file}")


def save_doc_ids(all_doc_ids, output_file):
    with open(output_file, 'w') as f:
        for doc_id in sorted(all_doc_ids):
            f.write(f"{doc_id}\n")


@hydra.main(version_base=None, config_path="../../config", config_name=None)
def main(cfg: DictConfig):
    print("Starting curation of snapshot document IDs...")
    curate_snapshot_doc_ids(cfg)
    
if __name__ == "__main__":
    main() 
    