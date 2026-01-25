from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import clamp, stable_card_id, utc_now_iso


# Review reason enums (stable strings)
RR_OCR_EMPTY = "OCR_EMPTY"
RR_LAYOUT_UNCERTAIN = "LAYOUT_UNCERTAIN"
RR_WORD_MISSING = "WORD_MISSING"
RR_LOW_CONFIDENCE = "LOW_CONFIDENCE"
RR_SEGMENT_FAILED = "SEGMENT_FAILED"
RR_CROP_FAILED = "CROP_FAILED"
RR_CROP_GATED_SMALL = "CROP_GATED_SMALL"
RR_CROP_GATED_RATIO = "CROP_GATED_RATIO"
RR_SUSPICIOUS_BBOX = "SUSPICIOUS_BBOX"
RR_HEURISTIC_WARNING = "HEURISTIC_WARNING"


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
                card_id = stable_card_id(source_ref, page_id, "UNKNOWN", None)
                now = utc_now_iso()
                cards.append(
                    {
                        "card_id": card_id,
                        "page_id": page_id,
                        "source_page_id": page_id,
                        "token_index": 0,
                        "layout_type": layout_type,
                        "word": "UNKNOWN",
                        "bbox_xyxy": None,
                        "method": "page",
                        "front_image_path": page_image_path,
                        "source_ref": source_ref,
                        "confidence": 0.0,
                        "needs_review": True,
                        "status": "review",
                        "created_at": now,
                        "updated_at": now,
                        "reasons": [RR_OCR_EMPTY],
                    }
                )
                review_items.append(
                    {
                        "card_id": card_id,
                        "page_id": page_id,
                        "source_ref": source_ref,
                        "text": "",
                        "bbox_xyxy": None,
                        "word_candidates": [],
                        "review_reason": RR_OCR_EMPTY,
                        "reason": RR_OCR_EMPTY,
                        "suggested_action": "manual_enter_word_or_fix_ocr",
                    }
                )
                return cards, review_items

            word = token.get("text", "")
            conf = float(token.get("confidence", 0.0))
            bbox_xyxy = token.get("bbox_xyxy")

            front = page_image_path
            reasons: list[str] = []
            needs_review = False
            method = "page"

            if segment and segment.get("status") == "success" and segment.get("crop_path"):
                front = segment["crop_path"]
                method = "segmenter"
            else:
                reasons.append(RR_SEGMENT_FAILED)

            if not word:
                needs_review = True
                reasons.append(RR_WORD_MISSING)

            if conf < self.min_confidence:
                needs_review = True
                reasons.append(RR_LOW_CONFIDENCE)

            card_id = stable_card_id(source_ref, page_id, str(word), bbox_xyxy)
            now = utc_now_iso()
            status = "review" if needs_review else "active"

            cards.append(
                {
                    "card_id": card_id,
                    "page_id": page_id,
                    "source_page_id": page_id,
                    "token_index": 0,
                    "layout_type": layout_type,
                    "word": word,
                    "bbox_xyxy": bbox_xyxy,
                    "method": method,
                    "front_image_path": front,
                    "source_ref": source_ref,
                    "confidence": conf,
                    "needs_review": needs_review,
                    "status": status,
                    "created_at": now,
                    "updated_at": now,
                    "reasons": reasons,
                }
            )

            if needs_review and (RR_SEGMENT_FAILED in reasons or RR_LOW_CONFIDENCE in reasons):
                review_items.append(
                    {
                        "card_id": card_id,
                        "page_id": page_id,
                        "source_ref": source_ref,
                        "text": str(word),
                        "bbox_xyxy": bbox_xyxy,
                        "word_candidates": [t.get("text") for t in clean.get("tokens", []) if t.get("text")],
                        "review_reason": RR_SEGMENT_FAILED if RR_SEGMENT_FAILED in reasons else RR_LOW_CONFIDENCE,
                        "reason": RR_SEGMENT_FAILED if RR_SEGMENT_FAILED in reasons else RR_LOW_CONFIDENCE,
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
                token_index = int(t.get("token_index") or 0)
                raw_conf = float(t.get("confidence", c_default))
                conf = clamp(raw_conf, cmin, cmax)

                bbox_xyxy = t.get("bbox_xyxy")
                crop_path = t.get("crop_path")
                crop_status = t.get("crop_status")
                warnings = t.get("warnings") or []

                # Default: multi_word tokens do NOT go to review.
                needs_review = False
                reasons: list[str] = []

                front = page_image_path
                method = "page"
                if crop_path:
                    front = crop_path
                    method = "bbox_crop"
                elif crop_status == "failed":
                    # Distinguish gate reasons from generic failures.
                    if "CROP_GATED_SMALL" in warnings:
                        reasons.append(RR_CROP_GATED_SMALL)
                    if "CROP_GATED_RATIO" in warnings:
                        reasons.append(RR_CROP_GATED_RATIO)
                    if RR_CROP_GATED_SMALL not in reasons and RR_CROP_GATED_RATIO not in reasons:
                        reasons.append(RR_CROP_FAILED)

                if raw_conf < self.min_confidence:
                    reasons.append(RR_LOW_CONFIDENCE)

                # Map warnings to review reasons.
                if "BBOX_INVALID" in warnings:
                    reasons.append(RR_SUSPICIOUS_BBOX)
                if any(w for w in warnings if w not in ("BBOX_INVALID", "CROP_GATED_SMALL", "CROP_GATED_RATIO")):
                    reasons.append(RR_HEURISTIC_WARNING)

                card_id = stable_card_id(source_ref, page_id, str(word), bbox_xyxy)

                # v0.3: lifecycle metadata
                needs_review = bool(reasons)
                now = utc_now_iso()
                status = "review" if needs_review else "active"
                cards.append(
                    {
                        "card_id": card_id,
                        "page_id": page_id,
                        "source_page_id": page_id,
                        "token_index": token_index,
                        "layout_type": layout_type,
                        "word": word,
                        "bbox_xyxy": bbox_xyxy,
                        "method": method,
                        "front_image_path": front,
                        "source_ref": source_ref,
                        "confidence": conf,
                        "needs_review": needs_review,
                        "status": status,
                        "created_at": now,
                        "updated_at": now,
                        "reasons": reasons,
                    }
                )

                # Selective review items for multi_word
                if reasons:
                    # pick a stable, primary reason
                    primary = reasons[0]
                    review_items.append(
                        {
                            "card_id": card_id,
                            "page_id": page_id,
                            "source_ref": source_ref,
                            "text": str(word),
                            "bbox_xyxy": bbox_xyxy,
                            "review_reason": primary,
                            "reason": primary,
                            "suggested_action": "manual_review",
                            "front_image_path": front,
                            "token_index": token_index,
                        }
                    )

            if len(cards) == 0:
                card_id = stable_card_id(source_ref, page_id, "", None)
                review_items.append(
                    {
                        "card_id": card_id,
                        "page_id": page_id,
                        "source_ref": source_ref,
                        "text": "",
                        "bbox_xyxy": None,
                        "word_candidates": [],
                        "review_reason": RR_OCR_EMPTY,
                        "reason": RR_OCR_EMPTY,
                        "suggested_action": "manual_enter_word_or_fix_ocr",
                    }
                )
            return cards, review_items

        # unknown
        unknown_card_id = stable_card_id(source_ref, page_id, "UNKNOWN", None)
        now = utc_now_iso()
        cards.append(
            {
                "card_id": unknown_card_id,
                "page_id": page_id,
                "source_page_id": page_id,
                "layout_type": layout_type,
                "word": "UNKNOWN",
                "bbox_xyxy": None,
                "method": "page",
                "front_image_path": page_image_path,
                "source_ref": source_ref,
                "confidence": 0.0,
                "needs_review": True,
                "status": "review",
                "created_at": now,
                "updated_at": now,
                "reasons": [RR_LAYOUT_UNCERTAIN],
            }
        )
        review_items.append(
            {
                "card_id": unknown_card_id,
                "page_id": page_id,
                "source_ref": source_ref,
                "text": "UNKNOWN",
                "bbox_xyxy": None,
                "word_candidates": [t.get("text") for t in clean.get("tokens", []) if t.get("text")],
                "review_reason": RR_LAYOUT_UNCERTAIN,
                "reason": RR_LAYOUT_UNCERTAIN,
                "suggested_action": "manual_review",
            }
        )
        return cards, review_items
