from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .job import JobPaths
from .utils import write_json


@dataclass
class LayoutClassifier:
    paths: JobPaths
    layout_cfg: dict[str, Any]

    def classify(self, page_id: str, clean: dict[str, Any], image_size: tuple[int, int] | None = None) -> dict[str, Any]:
        out_path = self.paths.stage_layout_dir / f"{page_id}.json"

        n = int(clean.get("clean_count", 0))
        # Backward-compatible fallback.
        single_word_max = int(self.layout_cfg.get("single_word_max_tokens", 2))

        # v0.2 bbox-based heuristic knobs
        dominant_ratio_min = float(self.layout_cfg.get("single_word_dominant_token_ratio", 0.70))
        max_hull_area_ratio = float(self.layout_cfg.get("single_word_max_hull_area_ratio", 0.25))
        max_tokens_fallback = int(self.layout_cfg.get("single_word_max_tokens_fallback", 6))

        if n == 0:
            layout_type = "unknown"
            needs_review = True
            reasons = ["ocr_empty_or_unstable"]
        else:
            layout_type = None
            reasons = []
            needs_review = False

            # Prefer bbox/area heuristic when possible.
            tokens = clean.get("tokens", [])
            if image_size and tokens:
                try:
                    w, h = image_size
                    page_area = float(max(1, w * h))
                    bboxes = [t.get("bbox_xyxy") for t in tokens if t.get("bbox_xyxy")]
                    if bboxes:
                        xs0 = [int(b[0]) for b in bboxes]
                        ys0 = [int(b[1]) for b in bboxes]
                        xs1 = [int(b[2]) for b in bboxes]
                        ys1 = [int(b[3]) for b in bboxes]
                        hx0, hy0, hx1, hy1 = min(xs0), min(ys0), max(xs1), max(ys1)
                        hull_area = float(max(1, (hx1 - hx0) * (hy1 - hy0)))
                        hull_area_ratio = hull_area / page_area

                        areas = [max(1, (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1]))) for b in bboxes]
                        dominant_ratio = float(max(areas)) / float(max(1.0, hull_area))

                        # If a single bbox dominates the text hull and the hull itself is compact,
                        # treat as single_word even if OCR produced a few tokens.
                        if (
                            dominant_ratio >= dominant_ratio_min
                            and hull_area_ratio <= max_hull_area_ratio
                            and len(tokens) <= max_tokens_fallback
                        ):
                            layout_type = "single_word"
                        else:
                            layout_type = "multi_word"
                except Exception:
                    # fail-soft: fall back to count heuristic
                    layout_type = None

            if layout_type is None:
                # Fallback heuristic (v0.1 behavior)
                if n <= single_word_max:
                    layout_type = "single_word"
                else:
                    layout_type = "multi_word"

        out = {
            "page_id": page_id,
            "layout_type": layout_type,
            "cleaned_words": n,
            "needs_review": needs_review,
            "reasons": reasons,
        }
        write_json(out_path, out)
        return out
