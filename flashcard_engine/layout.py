from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .job import JobPaths
from .utils import write_json


@dataclass
class LayoutClassifier:
    paths: JobPaths
    layout_cfg: dict[str, Any]

    def classify(self, page_id: str, clean: dict[str, Any]) -> dict[str, Any]:
        out_path = self.paths.stage_layout_dir / f"{page_id}.json"

        n = int(clean.get("clean_count", 0))
        single_word_max = int(self.layout_cfg.get("single_word_max_tokens", 2))

        if n == 0:
            layout_type = "unknown"
            needs_review = True
            reasons = ["ocr_empty_or_unstable"]
        elif n <= single_word_max:
            layout_type = "single_word"
            needs_review = False
            reasons = []
        else:
            layout_type = "multi_word"
            needs_review = False
            reasons = []

        out = {
            "page_id": page_id,
            "layout_type": layout_type,
            "cleaned_words": n,
            "needs_review": needs_review,
            "reasons": reasons,
        }
        write_json(out_path, out)
        return out
