"""
collecting the document ids for the different snapshots according the publishedDate 
"""
import hydra
from omegaconf import DictConfig
import requests
import time
from pathlib import Path

import urllib.parse
from collections.abc import Iterable


def build_query(published_date_before, published_date_after) -> str:
    query = f"publishedDate>={published_date_after} AND publishedDate<={published_date_before}"
    return urllib.parse.quote(query)

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
    
def get_number_of_pages(api_base_url, search_endpoint, published_date_before, published_date_after, api_key, batch_size=10, max_tries=5):
    url = f'{api_base_url}{search_endpoint}?q={build_query(published_date_before, published_date_after)}&limit=1&offset=0'
    
    payload={}
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    result = execute_request(url, headers, payload, max_tries)
    
    if result is None:
        return 0 
    total_results = result["totalHits"]
    return (total_results + batch_size - 1) // batch_size

def get_first_work_batch(api_base_url, search_endpoint, published_date_before, published_date_after, api_key, batch_size=10, max_tries=5):
    url = f'{api_base_url}{search_endpoint}?q={build_query(published_date_before, published_date_after)}&limit={batch_size}&scroll=true'

    #print(url)

    payload={}
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    result = execute_request(url, headers, payload, max_tries)
    if result is None:
        return None 
    doc_ids = set([item['id'] for item in result.get('results', [])])
    return doc_ids, result.get('scrollId', None)


def get_work_batch(api_base_url, search_endpoint, api_key, scrollId, max_tries=5):
    url = f'{api_base_url}{search_endpoint}?scroll=true&scrollId={scrollId}'

    payload={}
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    result = execute_request(url, headers, payload, max_tries)
    if result is None:
        return None 
    doc_ids = set([item['id'] for item in result.get('results', [])])
    return doc_ids

def curate_snapshot_doc_ids(cfg: DictConfig):
    api_base_url = cfg.api_base_url
    search_endpoint = cfg.search_endpoint
    api_key = cfg.api_key
    batch_size = cfg.get('batch_size', 10)
    max_tries = cfg.get('max_tries', 5)
    snapshots = cfg.snapshots
    snapshots_doc_ids_folder = Path(cfg.snapshots_doc_ids_folder)
    snapshots_doc_ids_folder.mkdir(parents=True, exist_ok=True)

    for snapshot in snapshots:
        snapshot_name = snapshot.name
        published_date_before = snapshot.published_date_before
        published_date_after = snapshot.published_date_after

        print(f"Curating doc IDs for snapshot: {snapshot_name} ({published_date_after} to {published_date_before})")

        num_pages = get_number_of_pages(api_base_url, search_endpoint, published_date_before, published_date_after, api_key, batch_size, max_tries)
        print(f"Total pages to fetch: {num_pages}")

        all_doc_ids = set()
        checkpoint_interval = 5000
        output_file = snapshots_doc_ids_folder / f"doc_ids-{snapshot_name}.txt"
        
        doc_ids, scrollId = get_first_work_batch(api_base_url, search_endpoint, published_date_before, published_date_after, api_key, batch_size, max_tries)
        if doc_ids is not None:
            all_doc_ids.update(doc_ids)
        print(f"Fetched page 1/{num_pages}, total doc IDs collected: {len(all_doc_ids)}")
        
        if scrollId is None:
            print("No scrollId returned, ending fetch.")
            save_doc_ids(all_doc_ids, output_file)
            continue
                
        for page in range(1, num_pages):
            doc_ids = get_work_batch(api_base_url, search_endpoint, api_key, scrollId, max_tries)
            if doc_ids is not None:
                all_doc_ids.update(doc_ids)
            print(f"Fetched page {page + 1}/{num_pages}, total doc IDs collected: {len(all_doc_ids)}")
            
            # Save checkpoint every 5000 records
            if ((page + 1) * batch_size) % checkpoint_interval == 0 and len(all_doc_ids) > 0:
                save_doc_ids(all_doc_ids, output_file)
                print(f"Checkpoint saved: {len(all_doc_ids)} doc IDs")

        # Final save
        save_doc_ids(all_doc_ids, output_file)
        print(f"Saved {len(all_doc_ids)} doc IDs to {output_file}")

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
    