"""Flashcard material-set production engine (MVP).

This package intentionally focuses on producing:
- page images + optional crops
- result.json (cards)
- review_queue.json (ambiguous cases)

Exporter formats (Anki/CSV/Quizlet) are out of scope.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
