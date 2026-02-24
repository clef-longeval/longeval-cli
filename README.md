# conf26-LongEval-Sci


# Queries
columns
"query_id", "query", "session_id", "turn"

## 1. Crete candidate queries task 1
`get-candidate-queries-task-1.py`

returns:
`candidate_queries_task_1.tsv`


## 2. Filter candidte queries task 1
`filter-candidate-queries-task-1.py`

returns:
`candidate_queries_task_1_new.tsv`


## 3. Parse queries task 3
`parse-queries-task-3.py`

returns: 
`candidate_queries_task_3.tsv`


## 4. Create query file for new queries
`create-query-file-task-3-new-queries.py`

returns:
`candidate_queries_task_3_new.tsv`



# Qrels
columns
"query_id", "q0", "doc_id", "label"

## 1. Raw queries
Each click is marked as relevant:
`qrels-raw.py`

returns:
`data/interim/qrels-1-raw.tsv`
`data/interim/qrels-2-raw.tsv`
`data/interim/qrels-3-raw.tsv`

## 2. DCTR queries
`qrels-dctr.py`

returns:
`data/interim/qrels-1-dctr.tsv`
`data/interim/qrels-2-dctr.tsv` 
`data/interim/qrels-3-dctr.tsv` 


## Validate
Each candidate query of task 1 needs to be in raw


# Export
## Queries
`create-output-files.py`

returns
`data/processed/queries-train.tsv`
`data/processed/queries-test.tsv`


`data/processed/qrels-snapshot-1-dctr.txt`
`data/processed/qrels-snapshot-2-dctr.txt`
`data/processed/qrels-snapshot-3-dctr.txt`

`data/processed/qrels-snapshot-1-raw.txt`
`data/processed/qrels-snapshot-2-raw.txt`
`data/processed/qrels-snapshot-3-raw.txt`
