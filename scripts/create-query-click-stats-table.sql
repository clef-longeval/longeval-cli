-- Create click monthly stats
CREATE TABLE public.query_monthly_click_stats (
    query_id             INTEGER,
    click_month          DATE,
    total_clicks         BIGINT DEFAULT 0,
    author               BIGINT DEFAULT 0,
    data_provider        BIGINT DEFAULT 0,
    get_pdf              BIGINT DEFAULT 0,
    works                BIGINT DEFAULT 0,
    not_specified        BIGINT DEFAULT 0,
    PRIMARY KEY (query_id, click_month)
);

ALTER TABLE public.query_monthly_click_stats OWNER TO postgres;


-- -- Increase memory for sorting and hashes (Work Mem)
-- SET work_mem = '4GB';

-- -- Allow more CPU cores to participate in the scan
-- SET max_parallel_workers_per_gather = 24; -- Adjust based on your new core count
-- SET min_parallel_table_scan_size = '100MB';

-- -- Speed up the final VACUUM/REINDEX
-- SET maintenance_work_mem = '4GB';


INSERT INTO public.query_monthly_click_stats (
    query_id,
    click_month,
    total_clicks,
    author,
    data_provider,
    get_pdf,
    works,
    not_specified
)
SELECT
    sn.query_id,
    DATE_TRUNC('month', sn.log_date)::DATE AS click_month,
    COUNT(tl.id) AS total_clicks,
    COUNT(tl.id) FILTER (WHERE tl.entry_type = 'author') AS author,
    COUNT(tl.id) FILTER (WHERE tl.entry_type = 'data-provider') AS data_provider,
    COUNT(tl.id) FILTER (WHERE tl.entry_type = 'get-pdf') AS get_pdf,
    COUNT(tl.id) FILTER (WHERE tl.entry_type = 'works') AS works,
    COUNT(tl.id) FILTER (WHERE tl.entry_type NOT IN ('author', 'data-provider', 'get-pdf', 'works')) AS not_specified
FROM public.search_logs_normalized sn
INNER JOIN public.tracker_logs tl ON sn.search_id = tl.search_id
GROUP BY sn.query_id, click_month
ON CONFLICT (query_id, click_month)
DO UPDATE SET
    total_clicks = EXCLUDED.total_clicks,
    author = EXCLUDED.author,
    data_provider = EXCLUDED.data_provider,
    get_pdf = EXCLUDED.get_pdf,
    works = EXCLUDED.works,
    not_specified = EXCLUDED.not_specified;
