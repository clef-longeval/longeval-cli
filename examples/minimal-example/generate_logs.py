"""Generate the minimal example's synthetic search logs (seeded, reproducible).

    uv run python examples/minimal-example/generate_logs.py

Writes ``logs/<day>.jsonl`` in the LongEval-Sci v2 format (read by
``parser.py`` next to this script) and ``ground_truth.tsv``, the graded
relevance the clicks were simulated from. The committed files are this
script's output; rerunning it reproduces them byte for byte.

Clicks follow a position-based model: a displayed document is clicked with
probability ``examination(rank) * attractiveness(grade)``. Every query has a
pool of candidate documents and each search shows a random 10 of them in a
random order, so every document is seen at several ranks. That is what lets a
click model tell a document's attractiveness apart from its position.

A handful of lines are deliberately imperfect, so each pipeline step has
something to report (see README.md, "How the data was generated").
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 2024
HERE = Path(__file__).parent
DAYS = [datetime(2024, 11, d, tzinfo=timezone.utc) for d in (4, 5, 6)]
SERP_SIZE = 10
POOL_SIZE = 15
#: grade -> probability that an examined document is clicked
ATTRACTIVENESS = {0: 0.03, 1: 0.35, 2: 0.85}
#: interaction types: these two are clicks (see filter.yaml), the others not
CLICK_TYPES = ["works"] * 3 + ["get-pdf"]
OTHER_TYPES = ["author", "data-provider"]

#: Plain topical queries, each searched many times: the click model's data.
TOPICAL = [
    "climate change adaptation in coastal cities",
    "transformer models for protein structure prediction",
    "microplastics in freshwater ecosystems",
    "effects of remote work on employee productivity",
    "antibiotic resistance in hospital wastewater",
    "deep learning for medical image segmentation",
    "urban heat island mitigation strategies",
    "social media use and adolescent mental health",
    "lithium-ion battery degradation mechanisms",
    "gender pay gap in academia",
    "soil carbon sequestration agriculture",
    "misinformation spread during the covid-19 pandemic",
]
#: Boolean / field queries (CORE syntax): dropped by `require: is-boolean: 0`.
BOOLEAN = [
    "renewable energy AND policy",
    'graph neural networks AND language:"en" AND '
    "(yearPublished>=2020 AND yearPublished<=2024)",
]
#: Searched a few times, never clicked: dropped by `min: clicks: 1`.
UNCLICKED = ["UNGER, JIM / CARTOONIST", "translate english to tagalog"]
#: Searched only by the bot: has clicks, but none once bot sessions are dropped.
BOT_ONLY = "free pdf download"
#: searches per query on days 1, 2 and 3 (day 3 is outside snapshot-1)
SEARCHES = {"topical": (28, 28, 14), "boolean": (3, 3, 1), "unclicked": (3, 2, 1)}


def examination(rank: int) -> float:
    return 1 / rank**0.5


def main() -> None:
    rng = random.Random(SEED)
    users = [hashlib.md5(f"user-{i}".encode()).hexdigest()[:12] for i in range(40)]
    serial = iter(range(10**6))

    def search_id() -> str:
        return hashlib.md5(f"search-{next(serial)}".encode()).hexdigest()

    def moment(day: datetime) -> datetime:
        return day + timedelta(seconds=rng.randrange(8 * 3600, 22 * 3600))

    # Candidate pools and their grades: 1-2 highly relevant, 2-3 partially.
    pools: dict[str, list[tuple[int, int]]] = {}
    for query in TOPICAL + BOOLEAN + UNCLICKED:
        docs = rng.sample(range(100_000, 160_000_000), POOL_SIZE)
        n2, n1 = rng.randint(1, 2), rng.randint(2, 3)
        grades = [2] * n2 + [1] * n1 + [0] * (POOL_SIZE - n2 - n1)
        rng.shuffle(grades)
        pools[query] = list(zip(docs, grades))

    events: list[tuple[datetime, dict]] = []

    def emit(when: datetime, record: dict) -> None:
        events.append((when, {"date": int(when.timestamp()), **record}))

    def search(query: str, when: datetime, *, clicked: bool) -> tuple[str, list[int]]:
        sid = search_id()
        shown = rng.sample(pools[query], SERP_SIZE)
        serp = [doc for doc, _ in shown]
        emit(
            when,
            {"search_id": sid, "serp": serp, "query": query, "uid": rng.choice(users)},
        )
        if not clicked:
            return sid, serp
        for rank, (doc, grade) in enumerate(shown, start=1):
            if rng.random() < examination(rank) * ATTRACTIVENESS[grade]:
                after = timedelta(seconds=rng.randrange(5, 300))
                emit(
                    when + after,
                    {"trackId": f"{sid}-{doc}", "type": rng.choice(CLICK_TYPES)},
                )
        if rng.random() < 0.15:  # a non-click interaction (author/provider page)
            after = timedelta(seconds=rng.randrange(5, 300))
            emit(
                when + after,
                {
                    "trackId": f"{sid}-{rng.choice(serp)}",
                    "type": rng.choice(OTHER_TYPES),
                },
            )
        return sid, serp

    for kind, queries in (
        ("topical", TOPICAL),
        ("boolean", BOOLEAN),
        ("unclicked", UNCLICKED),
    ):
        for query in queries:
            for day, count in zip(DAYS, SEARCHES[kind]):
                for _ in range(count):
                    search(query, moment(day), clicked=kind != "unclicked")

    # --- Deliberate imperfections, one or two of each ---------------------
    day1 = DAYS[0]
    # A click 30 hours after its search: an interaction like any other, but
    # ClicksClassifier(within_hours=24) does not count it.
    when = moment(day1)
    sid, serp = search(TOPICAL[0], when, clicked=False)
    emit(when + timedelta(hours=30), {"trackId": f"{sid}-{serp[0]}", "type": "works"})
    # A click on a document the search did not show: import-clicks rejects it.
    when = moment(day1)
    sid, _ = search(TOPICAL[1], when, clicked=False)
    emit(when + timedelta(seconds=40), {"trackId": f"{sid}-999", "type": "works"})
    # Frontend glitches: a click without a search id at all.
    emit(moment(day1), {"trackId": "undefined"})
    emit(moment(DAYS[1]), {"trackId": "undefined"})
    # A search without a uid: gets its own anonymous session.
    when = moment(DAYS[1])
    serp = [doc for doc, _ in pools[TOPICAL[2]][:SERP_SIZE]]
    emit(when, {"search_id": search_id(), "serp": serp, "query": TOPICAL[2]})
    # A search that returned nothing: imported, but not a click-model session.
    emit(
        moment(DAYS[1]),
        {
            "search_id": search_id(),
            "serp": [],
            "query": "doi:(10.1016/j.bone.2011.08.025)",
            "uid": users[0],
        },
    )
    # The same search_id logged twice with different content: first one wins.
    when = moment(day1)
    sid, serp = search(TOPICAL[3], when, clicked=False)
    emit(
        when + timedelta(seconds=1),
        {"search_id": sid, "serp": serp[::-1], "query": TOPICAL[3], "uid": users[1]},
    )

    # --- A bot: bursts of searches seconds apart, always clicking ranks 1-5 --
    # Appended last, so the random draws for everything above are unchanged.
    bot = hashlib.md5(b"bot").hexdigest()[:12]
    pools[BOT_ONLY] = [(doc, 0) for doc in rng.sample(range(100_000, 160_000_000), 10)]
    for day in DAYS[:2]:
        for _ in range(4):
            when = moment(day)
            for query in rng.choices(TOPICAL + [BOT_ONLY], k=20):
                when += timedelta(seconds=rng.randrange(2, 5))
                sid = search_id()
                serp = [doc for doc, _ in rng.sample(pools[query], SERP_SIZE)]
                emit(when, {"search_id": sid, "serp": serp, "query": query, "uid": bot})
                for rank, doc in enumerate(serp[:5], start=1):
                    emit(
                        when + timedelta(seconds=rank),
                        {"trackId": f"{sid}-{doc}", "type": "works"},
                    )

    write_logs(events)
    write_ground_truth(pools)


def write_logs(events: list[tuple[datetime, dict]]) -> None:
    out = HERE / "logs"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()
    by_day: dict[str, list[str]] = {}
    for when, record in sorted(events, key=lambda e: e[0]):
        by_day.setdefault(when.date().isoformat(), []).append(json.dumps(record))
    for day, lines in by_day.items():
        (out / f"{day}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ground_truth(pools: dict[str, list[tuple[int, int]]]) -> None:
    lines = ["query\tdoc_id\tgrade"]
    for query, pool in pools.items():
        lines += [f"{query}\t{doc}\t{grade}" for doc, grade in sorted(pool)]
    (HERE / "ground_truth.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
