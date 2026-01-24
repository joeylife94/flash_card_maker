from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Page:
    page_index: int  # 0-based
    page_id: str  # e.g. page_003
    source_ref: str  # e.g. book.pdf#page=3
    image_path: str  # relative path under job dir (pages/page_003.png)


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass
class PageContext:
    page: Page
    raw_ocr: dict[str, Any] | None = None
    clean_ocr: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None
    segment: dict[str, Any] | None = None
