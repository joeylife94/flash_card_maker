from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import clamp


def _pick_representative_token(clean: dict[str, Any]) -> dict[str, Any] | None:
    tokens = clean.get("tokens", [])
    if not tokens:
        return None
    # highest confidence wins
    return max(tokens, key=lambda t: float(t.get("confidence", 0.0)))


@dataclass
class FlashcardBuilder:
    source_name: str
    min_confidence: float
    confidence_cfg: dict[str, Any]

    def build_cards_for_page(
        self,
        page_id: str,
        source_ref: str,
        layout: dict[str, Any],
        clean: dict[str, Any],
        segment: dict[str, Any] | None,
        page_image_path: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns (cards, review_items)."""
        layout_type = layout.get("layout_type", "unknown")
        review_items: list[dict[str, Any]] = []
        cards: list[dict[str, Any]] = []

        if layout_type == "single_word":
            token = _pick_representative_token(clean)
            if not token:
                cards.append(
                    {
                        "page_id": page_id,
                        "layout_type": layout_type,
                        "word": "UNKNOWN",
                        "front_image_path": page_image_path,
                        "source_ref": source_ref,
                        "confidence": 0.0,
                        "needs_review": True,
                        "reasons": ["ocr_empty"],
                    }
                )
                review_items.append(
                    {
                        "page_id": page_id,
                        "source_ref": source_ref,
                        "word_candidates": [],
                        "reason": "ocr_empty",
                        "suggested_action": "manual_enter_word_or_fix_ocr",
                    }
                )
                return cards, review_items

            word = token.get("text", "")
            conf = float(token.get("confidence", 0.0))

            front = page_image_path
            reasons: list[str] = []
            needs_review = False

            if segment and segment.get("status") == "success" and segment.get("crop_path"):
                front = segment["crop_path"]
            else:
                reasons.append("segment_failed_or_skipped")

            if not word:
                needs_review = True
                reasons.append("word_missing")

            if conf < self.min_confidence:
                needs_review = True
                reasons.append("low_confidence")

            cards.append(
                {
                    "page_id": page_id,
                    "layout_type": layout_type,
                    "word": word,
                    "front_image_path": front,
                    "source_ref": source_ref,
                    "confidence": conf,
                    "needs_review": needs_review,
                    "reasons": reasons,
                }
            )

            if needs_review and ("segment_failed_or_skipped" in reasons or "low_confidence" in reasons):
                review_items.append(
                    {
                        "page_id": page_id,
                        "source_ref": source_ref,
                        "word_candidates": [t.get("text") for t in clean.get("tokens", []) if t.get("text")],
                        "reason": "segment_failed" if "segment_failed_or_skipped" in reasons else "low_confidence",
                        "suggested_action": "manual_crop_or_fix_word",
                    }
                )
            return cards, review_items

        if layout_type == "multi_word":
            tokens = clean.get("tokens", [])
            c_default = float(self.confidence_cfg.get("multi_word_default", 0.5))
            cmin = float(self.confidence_cfg.get("multi_word_min", 0.4))
            cmax = float(self.confidence_cfg.get("multi_word_max", 0.6))

            for t in tokens:
                word = (t.get("text") or "").strip()
                if not word:
                    continue
                raw_conf = float(t.get("confidence", c_default))
                conf = clamp(raw_conf, cmin, cmax)

                needs_review = True  # MVP: multi_word is inherently ambiguous
                reasons = ["multi_word_page"]
                if raw_conf < self.min_confidence:
                    reasons.append("low_confidence")

                cards.append(
                    {
                        "page_id": page_id,
                        "layout_type": layout_type,
                        "word": word,
                        "front_image_path": page_image_path,
                        "source_ref": source_ref,
                        "confidence": conf,
                        "needs_review": needs_review,
                        "reasons": reasons,
                    }
                )

            if len(cards) == 0:
                review_items.append(
                    {
                        "page_id": page_id,
                        "source_ref": source_ref,
                        "word_candidates": [],
                        "reason": "ocr_empty",
                        "suggested_action": "manual_enter_word_or_fix_ocr",
                    }
                )
            return cards, review_items

        # unknown
        cards.append(
            {
                "page_id": page_id,
                "layout_type": layout_type,
                "word": "UNKNOWN",
                "front_image_path": page_image_path,
                "source_ref": source_ref,
                "confidence": 0.0,
                "needs_review": True,
                "reasons": ["layout_uncertain"],
            }
        )
        review_items.append(
            {
                "page_id": page_id,
                "source_ref": source_ref,
                "word_candidates": [t.get("text") for t in clean.get("tokens", []) if t.get("text")],
                "reason": "layout_uncertain",
                "suggested_action": "manual_review",
            }
        )
        return cards, review_items
