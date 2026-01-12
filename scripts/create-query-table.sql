-- Create a table with the unique queries


-- allocate more resources
SET work_mem = '8GB';
SET max_parallel_workers_per_gather = 24;

-- Drop the old table to start fresh (assuming it's empty/failed)
-- DROP TABLE IF EXISTS queries;

-- Create table AND fill it in one go
CREATE TABLE queries AS
SELECT DISTINCT
    -- Strict 512 limit prevents the "Index row size" error
    LEFT(TRIM(REGEXP_REPLACE(LOWER(query), '\s+', ' ', 'g')), 512) AS query_text
FROM search_logs
WHERE query IS NOT NULL AND query <> '';

-- Add the ID
ALTER TABLE queries ADD COLUMN query_id SERIAL PRIMARY KEY;

-- Add index
CREATE UNIQUE INDEX idx_queries_text ON queries(query_text);



-- Rebuild the search logs table to reference the query table now
CREATE TABLE search_logs_normalized AS
SELECT
    sl.id AS original_id,  -- Keep the old ID for reference
    sl.log_date,
    sl.search_id,
    q.query_id,            -- Your new normalized ID
    sl.uid,
    sl.serp::text[] as serp -- CASTING to text[] to match tracker_logs
FROM search_logs sl
JOIN queries q
  -- We repeat the normalization logic here to ensure the join matches
  ON q.query_text = LEFT(TRIM(REGEXP_REPLACE(LOWER(query), '\s+', ' ', 'g')), 512);

-- Add Primary Key
ALTER TABLE search_logs_normalized ADD PRIMARY KEY (original_id);



-- Add indices
-- Index for Click Models (Aggregating by Query)
CREATE INDEX idx_norm_query_id ON search_logs_normalized(query_id);
-- Index for Joining (Linking to Tracker Logs)
CREATE INDEX idx_norm_search_id ON search_logs_normalized(search_id);

CREATE INDEX idx_norm_search_id_tracker ON tracker_logs(search_id);
CREATE INDEX idx_doc_id_tracker ON tracker_logs(doc_id);




-- Compare
SELECT
    (SELECT COUNT(*) FROM search_logs) as original_count,
    (SELECT COUNT(*) FROM search_logs_normalized) as new_count,
    (SELECT query_text FROM queries LIMIT 1) as sample_query,
    (SELECT serp FROM search_logs_normalized LIMIT 1) as sample_serp_type;