"""
SQLAlchemy engine/session setup.

The engine and session factory are created lazily (on first use, then
cached) rather than at import time. This keeps `import app.database`
cheap and driver-agnostic -- pure-logic modules (validation, spatial
analysis, models) can be imported and unit tested without psycopg2 or a
reachable Postgres instance installed.

The tutorial keeps schema management simple: `init_db()` calls
`Base.metadata.create_all`, which is enough for a local/learning
deployment. A production system would swap this for Alembic migrations
(the dependency is already in requirements.txt for that next step).
"""
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables that don't exist yet. Safe to call on every startup."""
    from app.models import document  # noqa: F401  (ensures models are registered on Base)

    Base.metadata.create_all(bind=get_engine())


@contextmanager
def get_session():
    """Context-managed session for use inside Celery tasks / scripts."""
    session: Session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency: yields a session and guarantees it's closed."""
    session: Session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
