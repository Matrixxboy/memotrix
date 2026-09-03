"""Database URL helper — re-exports the explicit resolver (no credential fallbacks)."""

from memotrix.config import resolve_database_url as get_database_url

__all__ = ["get_database_url"]
