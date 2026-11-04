"""Fit CLAX's DCTR model for one snapshot and score its query-document pairs.

    uv sync --extra clax
    uv run python train_dctr.py --filter filters/snapshot-1.yaml \
        --database-url "$DB" --output work/dctr-snapshot-1.parquet

The filter decides which query sessions and clicks are used. The output has
`query_id`, `doc_id`, `score` (click probability) and `query_sessions`, the
input of `export qrels` and `export snapshot`.
"""

from pathlib import Path

import click
import optax
from clax import Trainer
from clax.models import DocumentCTRModel
from flax import nnx

from src.clicks.clax_loaders import loaders_from_db
from src.clicks.scoring import score_from_db, write_scored_docs
from src.database import create_database_engine, make_session_factory
from src.filtering import load_snapshot_config


@click.command()
@click.option("--filter", "filter_path", required=True, type=click.Path(exists=True))
@click.option("--database-url", "url", default=None)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--epochs", default=20, show_default=True)
def main(filter_path: str, url: str | None, output: str, epochs: int) -> None:
    config = load_snapshot_config(Path(filter_path))
    engine = create_database_engine(url)
    with make_session_factory(engine)() as session:
        # The loaders read lazily, so train while the session is open.
        query_doc_ids, train, validation = loaders_from_db(
            session, config, holdout=10, batch_size=1024, max_positions=10
        )
        model = DocumentCTRModel(query_doc_pairs=len(query_doc_ids), rngs=nnx.Rngs(0))
        Trainer(optimizer=optax.adamw(3e-3), epochs=epochs).train(
            model, train, validation
        )

    scored = score_from_db(model, query_doc_ids, config, database_url=url)
    write_scored_docs(scored, output)
    click.echo(f"{len(query_doc_ids)} query-document pairs -> {output}")


if __name__ == "__main__":
    main()
