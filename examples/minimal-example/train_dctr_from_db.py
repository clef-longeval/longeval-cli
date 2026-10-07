"""Bare-minimum DCTR: fit CLAX's Document CTR model on the example database.

    uv sync --extra clax
    uv run python examples/minimal-example/train_dctr_from_db.py \
        --database-url "$DB" --output /tmp/minimal-scores.parquet

Query sessions come from the database (the snapshot and click types in
``filter.yaml``), ``loaders_from_db`` turns them into CLAX batches with every
5th query session held out for validation, and the model learns one click
probability per (query, document) pair.
"""

from pathlib import Path

import click
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
from clax import Trainer
from clax.models import DocumentCTRModel
from flax import nnx

from src.clicks.clax_loaders import loaders_from_db
from src.database import create_database_engine, make_session_factory
from src.filtering import load_snapshot_config


@click.command()
@click.option("--database-url", "url", default=None)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--epochs", default=200, show_default=True)
def main(url: str | None, output: str, epochs: int) -> None:
    config = load_snapshot_config(Path(__file__).with_name("filter.yaml"))
    engine = create_database_engine(url)
    with make_session_factory(engine)() as session:
        # The loaders read lazily, so train while the session is open.
        query_doc_ids, train, validation = loaders_from_db(
            session, config, holdout=5, batch_size=64
        )
        model = DocumentCTRModel(query_doc_pairs=len(query_doc_ids), rngs=nnx.Rngs(0))
        trainer = Trainer(optimizer=optax.adamw(0.05), epochs=epochs)
        trainer.train(model, train, validation)

    ids = jnp.arange(len(query_doc_ids))
    ctr = np.asarray(model.ctr.prob({"query_doc_ids": ids})).reshape(-1)
    queries, docs = zip(*query_doc_ids)
    pq.write_table(
        pa.table(
            {"query_id": queries, "doc_id": docs, "score": ctr.astype("float64")}
        ),
        output,
    )
    click.echo(f"{len(query_doc_ids)} query-document pairs -> {output}")


if __name__ == "__main__":
    main()
