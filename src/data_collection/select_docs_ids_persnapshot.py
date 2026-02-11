"""
script to select document IDs for each snapshot, and create negative samples of document IDs for each snapshot 
"""

import os
from pathlib import Path
import random
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig


def read_doc_ids(file_path):
    doc_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            doc_id = line.strip()
            if doc_id:
                doc_ids.add(doc_id)
    return doc_ids

def select_negative_samples(negative_doc_ids, sample_size=1000):
    if len(negative_doc_ids) <= sample_size:
        return negative_doc_ids
    return set(random.sample(sorted(negative_doc_ids), sample_size))

def construct_doc_ids_for_snapshot(qrels_snapshot_doc_ids, negative_snapshot_doc_ids, published_date_snapshot_doc_ids, negative_sample_ration=0.5, published_date_ratio=0.5):
    """
    Construct a set of document IDs for a snapshot by combining qrels snapshot doc IDs with selected negative samples and published date samples.
    
    :param qrels_snapshot_doc_ids: document IDs from qrels for the snapshot
    :param negative_snapshot_doc_ids: document IDs that are selected as random negative samples for the snapshot
    :param published_date_snapshot_doc_ids: document IDs that are selected based on published date criteria for the snapshot
    :param negative_sample_ration: ratio of negative samples to select relative to qrels snapshot doc IDs
    :param published_date_ratio: ratio of published date samples to select relative to qrels snapshot doc IDs
    """
    # Determine how many negative samples to select based on the specified ratios
    num_negative_samples = int(len(qrels_snapshot_doc_ids) * negative_sample_ration)
    num_published_date_samples = int(len(qrels_snapshot_doc_ids) * published_date_ratio)

    # Select negative samples and published date samples
    selected_negative_samples = select_negative_samples(negative_snapshot_doc_ids, sample_size=num_negative_samples)
    selected_published_date_samples = select_negative_samples(published_date_snapshot_doc_ids, sample_size=num_published_date_samples)

    # Combine selected samples with qrels snapshot doc IDs
    final_doc_ids = set(qrels_snapshot_doc_ids) | selected_negative_samples | selected_published_date_samples

    return final_doc_ids


def write_docs_set(doc_ids, output_file):
    with open(output_file, 'w') as f:
        for doc_id in sorted(doc_ids):
            f.write(f"{doc_id}\n")
            

@hydra.main(version_base=None, config_path="../../config", config_name=None)
def main(cfg: DictConfig):
    
    
    snapshots = cfg.snapshots
    qrels_snaphosts_dir = Path(cfg.qrels_snapshots_dir)
    negative_snaphosts_dir = Path(cfg.negative_snaphosts_dir)
    published_date_snaphosts_dir = Path(cfg.published_date_snaphosts_dir)
    output_folder = Path(cfg.selected_docids_persnapshot_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    previous_snapshot_doc_ids = set()
    for i in range(len(snapshots)):
        snapshot_name = snapshots[i].name
        qrels_doc_ids = read_doc_ids(qrels_snaphosts_dir / f'doc_ids-{snapshot_name}.txt')
        negative_doc_ids = read_doc_ids(negative_snaphosts_dir / f'doc_ids-{snapshot_name}.txt')
        published_date_doc_ids = read_doc_ids(published_date_snaphosts_dir / f'doc_ids-{snapshot_name}.txt')
        
        final_doc_ids = construct_doc_ids_for_snapshot(qrels_doc_ids, negative_doc_ids, published_date_doc_ids) | previous_snapshot_doc_ids
        previous_snapshot_doc_ids = final_doc_ids
        
        output_file = output_folder / f"doc_ids-{snapshot_name}.txt"
        write_docs_set(final_doc_ids, output_file)
        print(f"Snapshot: {snapshot_name} | Total doc IDs: {len(final_doc_ids)} | Output file: {output_file}")


if __name__ == "__main__":
    main()