# LongEval Sci 2026 


| Script | Library tags | Labels |
|---|---|---|
| `qrels_stats_raw_from_library.py` | `longeval-sci-2026/snapshot-{1,2,3}/raw` | all rows are label 1 |
| `qrels_stats_dctr_from_library.py` | `longeval-sci-2026/snapshot-{1,2,3}/dctr` | graded 0 / 1 / 2 (0 = judged, not relevant) |

| Snapshot | Period |
|---|---|
| 1 | 03-05 |
| 2 | 06-08 |
| 3 | 09-11 |

The test queries (381) come from the same tags, because all three snapshots share one test queries file.

## Setup


On the first run, the library downloads the qrels and queries into `~/.ir_datasets/longeval-sci-2026/`. Later runs use those cached copies, so you need internet access only the first time.

## How to run

Run from the folder that contains the scripts:

```bash
conda activate longeval

python qrels_stats_raw_from_library.py  --out-dir qrels_stats/raw_library
python qrels_stats_dctr_from_library.py --out-dir qrels_stats/dctr_library
```

### Options (both scripts)

| Option | Default | Meaning |
|---|---|---|
| `--out-dir DIR` | `results` | Folder for the CSVs. It is created if it doesn't exist. |
| `--no-queries` | off | Skip the test queries. This drops the coverage columns, the "never judged" row and the query text. |

## Output folders

```
qrels_stats/
├── raw_library/            ← qrels_stats_raw_from_library.py
│   ├── summary_raw.csv
│   ├── overlap_raw.csv
│   └── per_query_raw.csv
├── dctr_library/           ← qrels_stats_dctr_from_library.py
│   ├── summary_dctr.csv
│   ├── overlap_dctr.csv
│   └── per_query_dctr.csv
├── raw/                    ← same stats from the downloaded files (qrels_stats_from_files.py)
└── dctr/                   ← same stats from the downloaded files (qrels_stats_dctr_from_files.py)
```

The `*_library` results are identical to the file-based ones in `raw/` and `dctr/`. The only difference is an extra `tag` column in the summary. Each script also prints its three tables to the terminal.

## Output files

### Raw (`qrels_stats/raw_library/`)

**`summary_raw.csv`** has one row per snapshot, with these columns:

| Column | Meaning |
|---|---|
| `period`, `tag` | Snapshot months and library tag. |
| `judged_queries` | Queries with at least one qrel. |
| `qrels_rows` | Total qrels rows. |
| `unique_docs` | Number of distinct documents. |
| `duplicate_pairs` | Repeated query–doc pairs. A value above 0 points to a data problem. |
| `qrels_per_query_mean`, `_median`, `_max` | Qrels per judged query. |
| `label_<x>` | Row count per label. Raw has only `label_1`. |
| `total_queries` | Number of test queries. |
| `judged_share_%` | Judged queries as a share of the test queries. |
| `judged_not_in_queries_file` | Judged query IDs missing from the test queries. Should be 0. |

**`overlap_raw.csv`** shows which judged queries appear in which snapshots:

| Rows | Meaning |
|---|---|
| `S1 only` … `S2 & S3 only` | Exclusive regions of the Venn diagram. |
| `all three`, `exactly one snapshot`, `exactly two snapshots` | Queries grouped by how many snapshots judge them. |
| `union` | Distinct judged queries across all snapshots. |
| pairwise totals | Size of each pairwise overlap. |
| `test queries never judged` | Test queries with no qrels in any snapshot. |

**`per_query_raw.csv`** has one row per judged query: `text`, `qrels_S1`, `qrels_S2`, `qrels_S3` and `n_snapshots_judged`. It is sorted by the number of snapshots, then by qrels count.

### DCTR (`qrels_stats/dctr_library/`)

DCTR has rows with label 0, which means the query was judged but the document is not relevant. So a query can count in two ways:

- **Judged:** the query has at least one row, with any label.
- **Relevant:** the query has at least one row with label > 0. Only these queries can actually be scored.

**`summary_dctr.csv`** has the same structure as the raw summary, with these differences:

| Column | Meaning |
|---|---|
| `queries_with_relevant` | Queries with at least one row with label > 0. |
| `queries_all_zero` | Judged queries whose rows are all label 0. |
| `label_0`, `label_1`, `label_2` | Row count per label. `other_labels` appears only if unexpected labels exist. |
| `relevant_rows (label>0)`, `relevant_share_%` | Relevant rows, as a count and as a share of all rows. |
| `rows_per_query_*` | All rows per query. |
| `relevant_per_query_*` | Relevant rows per query. |
| `relevant_query_share_%` | Relevant queries as a share of the test queries. |

**`overlap_dctr.csv`** has two columns, one for judged queries (`judged (any label)`) and one for relevant queries (`with >=1 relevant (label>0)`). Its rows are the per-snapshot counts, the pairwise overlaps, `all three`, `exactly two`, `exactly one`, `union`, and the test queries never judged or never with a relevant doc.

**`per_query_dctr.csv`** has one row per judged query: `text`, `rows_S1..3`, `relevant_S1..3`, `n_snapshots_judged` and `n_snapshots_relevant`. It is sorted with queries relevant in the most snapshots first.
