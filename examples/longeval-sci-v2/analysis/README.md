# Log analysis

Run `python3 examples/longeval-sci-v2/analysis/collect_monthly_activity.py`
from the repository root. SQL and fixed paths are inside the script.
The database is opened read-only; output is `output/monthly_clicks.csv`.

Each row contains `month`, `type`, `clicks`, and `valid_clicks`.
All stored interaction types are included, and repeated clicks remain separate.
Month is the originating search month in UTC, including for later clicks.
Searches without a parseable date are excluded.

Valid clicks exclude interactions with verdicts stored globally or under `snapshot-3` (the existing classification run)
from classifier `click-delay` with label `late`. This classifier defaults to
marking clicks more than 24 hours after their search as late. No other filters
are applied. Unlabelled interactions are retained, matching exclusion-filter
semantics (including interactions with missing timestamps).

For monthly queries, run `collect_monthly_queries.py` in the same directory.
It saves `output/monthly_queries.csv`: one row per month and query ID, with
search count and structured, Boolean, DOI, and word-count labels. Labels use
global verdicts where present, otherwise the existing `snapshot-3` run.
Missing labels stay blank; they are not treated as false. All ingested searches
with parseable dates are included, without benchmark filters.

For monthly category totals, count rows for distinct queries and sum `searches`
for search volume. A query may occur in multiple months, and the classifier
flags overlap. Snapshot click counts and manual selection labels are omitted.

Run `collect_snapshot_query_clicks.py` for `output/snapshot_query_clicks.csv`,
with columns `snapshot`, `query_id`, and `clicks`. This exports the stored numeric
`clicks` classifier values for snapshot-1 (Mar–May 2025), snapshot-2 (Jun–Aug),
and snapshot-3 (Sep–Nov), without recomputing counts or applying query filters.
Counts reflect the configuration used when each classifier ran; the current
snapshot configurations count only `works` clicks and exclude late clicks.
Compare `clicks >= 20` to examine the 20-click bound within each snapshot.
Queries without clicks have no stored verdict and are absent from this export;
it cannot by itself provide the denominator for percentages of all queries.
No join to `queries` is needed because classifier rows already contain query IDs.

After query collection finishes, run `plot_query_overlap.py` (requires
pandas and matplotlib). It reads the CSV, without accessing SQLite, and saves only
`output/monthly_query_overlap.pdf`. The two heatmaps show
query intersections and Jaccard similarity (intersection / union), restricted
to unstructured queries (`is_structured = 0`) observed in at least three
different months. Queries searched many
times within only one month are excluded.
The count diagonal is each month's recurring-query-set size; Jaccard's diagonal
is 100% for nonempty sets. Empty-set unions have undefined similarity.
Search frequency does not weight the overlap; Boolean/DOI flags are not filtered. Missing
months remain absent, rather than being assumed to have zero activity.

Run `plot_query_frequency.py` for `output/query_frequency.pdf`: a histogram
of total searches per query across all collected months, with logarithmic bins
and axes. All queries are included without eligibility filters. The y-axis
counts queries per bin, not a probability density.

Run `plot_query_types.py` for `output/query_types.pdf`: unstructured,
structured, Boolean, and DOI flags from stored classifiers. Paired bars compare
shares of global distinct queries (deduplicated across months) and search
volume. Unstructured means `is-structured = 0`; it can still include Boolean
operators or DOI identifiers. Flags overlap, so the bars do not form a partition
and their percentages need not sum to 100%. Missing labels match no category
for that classifier; their counts are printed separately.

Run `plot_click_activity.py` for `output/click_activity.pdf`. Three panels (works, author, data-provider) show valid clicks with late clicks stacked above in a lighter
shade. Only 2025 is shown. The three panels are arranged side by side in a wide figure. Panels use independent y-axis scales in millions. Blank types are shown as Unknown;
absent month/type pairs within the CSV's month range are filled with zero stored
events, which does not establish logging completeness. Month attribution and
validity follow the click collector's definitions.

Run `plot_query_months.py` for `output/query_months.pdf`: distinct query counts
by number of observed months, restricted to queries appearing in at least two
months. All query types are included, without weighting by searches. Months
need not be consecutive. The y-axis is logarithmic; the excluded single-month
share is annotated.

Run `plot_upset.py` for `output/query_upset.pdf` (requires pandas, matplotlib,
and upsetplot). It shows the intersections of judged query IDs in the `dctr`
datasets for snapshots 1, 2, and 3. Each query is counted once per snapshot,
and queries with no qrels in all three snapshots are omitted.

## Snapshot-3 system-ranking agreement

Run `analyze-rankings.py` with pandas, matplotlib, scipy, and repro-eval.
It reads `data/processed/results.csv` and saves `output/ranking-agreement.pdf`.
Change `MEASURE` and `RBO_P` at the top of the script if needed.

The paired heatmaps compare full assessment sets on snapshot 3 using the same
systems, with SciPy Kendall tau-b and repro_eval RBO at the number of systems.
Each source retains its own query set. repro_eval uses normalized finite-depth
RBO and breaks score ties by reverse alphabetical system ID; tau-b retains ties.
