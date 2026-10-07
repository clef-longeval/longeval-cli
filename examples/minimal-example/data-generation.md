## 2. How the data was generated

`generate_logs.py` (fixed seed; rerunning it reproduces the committed files
byte for byte, which CI checks) simulates three days, 2024-11-04 to -06:

- **12 topical queries**, searched 28 times on each of days 1 and 2 and 14
  times on day 3. Each has a pool of 15 candidate documents: 1–2 with grade 2,
  2–3 with grade 1, the rest grade 0. Every search shows a random 10 of the 15
  in random order, so each document is seen at many ranks.
- **Clicks** follow a position-based model: the document at rank `r` is
  clicked with probability `1/sqrt(r) * attractiveness`, with attractiveness
  0.03 / 0.35 / 0.85 for grades 0 / 1 / 2. A click is `works` (3 in 4) or
  `get-pdf`. In about 15% of searches a user also opens an `author` or
  `data-provider` page.
- **2 boolean queries** (CORE syntax) with clicks, and **2 queries that are
  never clicked**, so that the filter has something to drop.
- **Deliberate imperfections**, one or two of each: a click 30 hours after
  its search, a click on a document the search didn't show, two
  `"trackId": "undefined"` events, a search without `uid`, a search that
  returned nothing, and one `search_id` logged twice with different content.
- **A bot**: on days 1 and 2 it makes 4 bursts of 20 searches a few seconds
  apart, and clicks ranks 1–5 of every result. Some of its searches are for
  `free pdf download`, a query no human uses.

40 users issue the human searches.