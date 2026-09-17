"""Database configuration shared by the offline and live-database workflows."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DATABASE_URL = "sqlite+pysqlite:///longeval.sqlite3"


class Base(DeclarativeBase):
    """Base class for the LongEval ORM models."""

    metadata = sa.MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "pk": "pk_%(table_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        }
    )


def database_url(url: str | None = None) -> str:
    """Return an explicit URL, the configured URL, or the offline SQLite default."""
    return url or os.environ.get("LONGEVAL_DATABASE_URL", DEFAULT_DATABASE_URL)


def create_database_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create an engine suitable for SQLite development or PostgreSQL production."""
    engine = sa.create_engine(database_url(url), echo=echo, pool_pre_ping=True)

    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)

    return engine


def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
    """Enable SQLite foreign-key enforcement for every newly opened connection."""
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_schema(engine: Engine) -> None:
    """Create missing tables without altering existing live-database tables."""
    # Importing here ensures model metadata is registered without a module-level
    # circular import: models import ``Base`` from this module.
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory with predictable post-commit object behaviour."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit a unit of work, rolling it back if the operation fails."""
    with session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
