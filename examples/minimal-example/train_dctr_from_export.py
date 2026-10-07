"""Bare-minimum DCTR from a frozen export: no database needed for training.

    uv sync --extra clax
    uv run longeval-cli export clicks /tmp/minimal-clicks \
        --filter examples/minimal-example/filter.yaml --database-url "$DB"
    uv run python examples/minimal-example/train_dctr_from_export.py \
        /tmp/minimal-clicks --output /tmp/minimal-scores-export.parquet

``export clicks`` saves the snapshot's query sessions once (sharded gzip-JSONL plus
a checksummed ``manifest.json``). Training then replays that directory with
``loaders_from_file``, so it can run later or elsewhere and always sees the
same query sessions in the same order. The model and output are the same as in
``train_dctr_from_db.py``, which streams from the database instead.
"""

import click
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
from clax import Trainer
from clax.models import DocumentCTRModel
from flax import nnx

from src.clicks.clax_loaders import loaders_from_file


@click.command()
@click.argument("export_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--epochs", default=200, show_default=True)
def main(export_dir: str, output: str, epochs: int) -> None:
    # holdout=5: every 5th query session validates, as in train_dctr_from_db.py
    query_doc_ids, train, validation = loaders_from_file(
        export_dir, holdout=5, batch_size=64
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
