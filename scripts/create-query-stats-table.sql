CREATE TABLE query_monthly_stats AS
SELECT
    query_id,
    -- Truncate timestamp to the first day of the month (e.g., 2023-01-01)
    DATE_TRUNC('month', log_date)::date AS search_month,
    COUNT(*) as search_count
FROM search_logs_normalized
GROUP BY 1, 2;

-- Index it for fast filtering later
CREATE INDEX idx_qms_query_id ON query_monthly_stats(query_id);
CREATE INDEX idx_qms_month ON query_monthly_stats(search_month);



SELECT *
FROM queries
WHERE query_id = '15756873'




SELECT
    search_logs_normalized.query_id,
    COUNT(*) as search_count
FROM search_logs_normalized
JOIN public.queries q on search_logs_normalized.query_id = q.query_id
WHERE q.is_structured is FALSE AND q.is_boolean is FALSE and q.has_doi is FALSE
GROUP BY 1
ORDER BY search_count DESC;
