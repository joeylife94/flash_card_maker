"""Entry point for running flashcard_engine as a module.

Usage:
    python -m flashcard_engine <command> [options]
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
