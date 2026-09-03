"""Memotrix entrypoint — delegates to the context CLI."""

from tests.context_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
