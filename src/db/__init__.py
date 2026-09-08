# __init__.py

"""Postgres access. Read-only is enforced by the database role, not here."""

from src.db.postgres import (
    DatabaseError,
    check_postgres_connection,
    postgres_configured,
)

__all__ = [
    "DatabaseError",
    "check_postgres_connection",
    "postgres_configured",
]
