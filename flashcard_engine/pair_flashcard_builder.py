"""Pair Mode Flashcard Builder - Convert extracted pairs to flashcards.

This module takes the output from pair extraction (picture + text crops)
and builds flashcard-ready data structures that can be exported to Anki/CSV.

Flow:
    result_pairs.json → PairFlashcardBuilder → result.json (cards format)

Features:
    - Forward cards: Picture (front) → Text (back)
    - Reverse cards: Text (front) → Picture (back) [optional]
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .utils import load_json, write_json, utc_now_iso


class CardDirection(Enum):
    """Direction of flashcard."""
    FORWARD = "forward"   # Picture → Text (default)
    REVERSE = "reverse"   # Text → Picture
    BOTH = "both"         # Generate both directions


@dataclass
class PairCard:
    """A flashcard created from a picture-text pair."""
    card_id: str
    pair_id: str
    page_id: str
    order_index: int
    direction: str  # forward or reverse
    # Front side: picture or text depending on direction
    front_image_path: str
    front_text: str  # For reverse cards
    # Back side: text or picture depending on direction  
    back_text: str
    back_image_path: str | None
    # Metadata
    source_ref: str
    confidence: float
    status: str  # active, review, rejected
    needs_review: bool
    reasons: list[str]
    created_at: str
    updated_at: str


@dataclass 
class PairFlashcardStats:
    """Statistics from pair-to-flashcard conversion."""
    pairs_total: int = 0
    cards_created: int = 0
    forward_cards: int = 0
    reverse_cards: int = 0
    cards_needing_review: int = 0
    cards_rejected: int = 0
    cards_no_text: int = 0


def _generate_card_id(pair_id: str, page_id: str, text: str, direction: str = "forward") -> str:
    """Generate stable card ID from pair data."""
    payload = f"{pair_id}|{page_id}|{text}|{direction}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class PairFlashcardBuilder:
    """Build flashcards from extracted pairs."""
    
    def __init__(
        self, 
        job_dir: str | Path, 
        source_name: str = "",
        direction: CardDirection = CardDirection.FORWARD,
    ):
        self.job_dir = Path(job_dir)
        self.source_name = source_name
        self.direction = direction
        
    def build_from_result_pairs(
        self, 
        include_reverse: bool = False,
    ) -> tuple[list[dict[str, Any]], PairFlashcardStats]:
        """Load result_pairs.json and build flashcards.
        
        Args:
            include_reverse: If True, generate reverse cards (Text → Picture)
        
        Returns:
            (cards_list, stats)
        """
        result_pairs_path = self.job_dir / "result_pairs.json"
        
        if not result_pairs_path.exists():
            return [], PairFlashcardStats()
        
        data = load_json(result_pairs_path)
        schema_version = data.get("schema_version", "1.0")
        
        # Determine effective direction
        effective_direction = self.direction
        if include_reverse:
            effective_direction = CardDirection.BOTH
        
        if schema_version.startswith("2.0"):
            # SAM-based format with pages[]
            return self._build_from_sam_format(data, effective_direction)
        else:
            # Original format with pairs[]
            return self._build_from_original_format(data, effective_direction)
    
    def _build_from_sam_format(
        self, 
        data: dict[str, Any],
        direction: CardDirection,
    ) -> tuple[list[dict[str, Any]], PairFlashcardStats]:
        """Build cards from SAM extractor output (schema 2.0)."""
        stats = PairFlashcardStats()
        cards: list[dict[str, Any]] = []
        now = utc_now_iso()
        
        pages = data.get("pages", [])
        
        for page_data in pages:
            page_id = page_data.get("page_id", "")
            page_pairs = page_data.get("pairs", [])
            
            for pair in page_pairs:
                stats.pairs_total += 1
                
                pair_id = pair.get("pair_id", "")
                order_index = pair.get("order_index", 0)
                caption_text = pair.get("caption_text", "").strip()
                picture_path = pair.get("picture_path", "")
                text_path = pair.get("text_path", "")
                needs_review = pair.get("needs_review", False)
                reasons = pair.get("reasons", [])
                confidence = pair.get("confidence", 0.0)
                
                # Determine status
                if not caption_text:
                    stats.cards_no_text += 1
                    status = "review"
                    needs_review = True
                    if "NO_TEXT" not in reasons:
                        reasons.append("NO_TEXT")
                elif needs_review:
                    status = "review"
                else:
                    status = "active"
                
                if needs_review:
                    stats.cards_needing_review += 1
                
                source_ref = f"{self.source_name}#{page_id}" if self.source_name else page_id
                
                # Generate forward card (Picture → Text) if requested
                if direction in (CardDirection.FORWARD, CardDirection.BOTH):
                    card_id = _generate_card_id(pair_id, page_id, caption_text, "forward")
                    
                    card = {
                        "card_id": card_id,
                        "pair_id": pair_id,
                        "page_id": page_id,
                        "source_page_id": page_id,
                        "token_index": order_index,
                        "direction": "forward",
                        "layout_type": "pair",
                        "word": caption_text,  # For backward compatibility
                        "front_text": "",      # Picture front, no text
                        "back_text": caption_text,
                        "bbox_xyxy": pair.get("picture_bbox"),
                        "method": "pair_sam",
                        "front_image_path": self._resolve_path(picture_path),
                        "back_image_path": self._resolve_path(text_path) if text_path else None,
                        "source_ref": source_ref,
                        "confidence": confidence,
                        "needs_review": needs_review,
                        "status": status,
                        "created_at": now,
                        "updated_at": now,
                        "reasons": reasons.copy(),
                    }
                    
                    cards.append(card)
                    stats.cards_created += 1
                    stats.forward_cards += 1
                
                # Generate reverse card (Text → Picture) if requested
                if direction in (CardDirection.REVERSE, CardDirection.BOTH) and caption_text:
                    card_id = _generate_card_id(pair_id, page_id, caption_text, "reverse")
                    
                    card = {
                        "card_id": card_id,
                        "pair_id": pair_id,
                        "page_id": page_id,
                        "source_page_id": page_id,
                        "token_index": order_index,
                        "direction": "reverse",
                        "layout_type": "pair_reverse",
                        "word": caption_text,
                        "front_text": caption_text,  # Text front
                        "back_text": "",             # Picture back
                        "bbox_xyxy": pair.get("picture_bbox"),
                        "method": "pair_sam_reverse",
                        "front_image_path": self._resolve_path(text_path) if text_path else "",
                        "back_image_path": self._resolve_path(picture_path),  # Picture is the answer
                        "source_ref": source_ref,
                        "confidence": confidence,
                        "needs_review": needs_review,
                        "status": status,
                        "created_at": now,
                        "updated_at": now,
                        "reasons": reasons.copy(),
                    }
                    
                    cards.append(card)
                    stats.cards_created += 1
                    stats.reverse_cards += 1
        
        return cards, stats
    
    def _build_from_original_format(
        self, 
        data: dict[str, Any],
        direction: CardDirection,
    ) -> tuple[list[dict[str, Any]], PairFlashcardStats]:
        """Build cards from original pair extractor output (schema 1.0)."""
        stats = PairFlashcardStats()
        cards: list[dict[str, Any]] = []
        now = utc_now_iso()
        
        pairs = data.get("pairs", [])
        
        for pair in pairs:
            stats.pairs_total += 1
            
            pair_id = pair.get("pair_id", "")
            page_id = pair.get("page_id", "")
            item_index = pair.get("item_index", 0)
            caption_text = pair.get("caption_text", "").strip()
            picture_path = pair.get("picture_path", "")
            text_path = pair.get("text_path", "")
            needs_review = pair.get("needs_review", False)
            reasons = pair.get("reasons", [])
            confidence = pair.get("confidence", 0.0)
            status = pair.get("status", "active")
            
            if not caption_text:
                stats.cards_no_text += 1
                status = "review"
                needs_review = True
            
            if needs_review:
                stats.cards_needing_review += 1
            
            if status == "rejected":
                stats.cards_rejected += 1
                continue  # Skip rejected pairs
            
            source_ref = f"{self.source_name}#{page_id}" if self.source_name else page_id
            
            # Generate forward card (Picture → Text) if requested
            if direction in (CardDirection.FORWARD, CardDirection.BOTH):
                card_id = _generate_card_id(pair_id, page_id, caption_text, "forward")
                
                card = {
                    "card_id": card_id,
                    "pair_id": pair_id,
                    "page_id": page_id,
                    "source_page_id": page_id,
                    "token_index": item_index,
                    "direction": "forward",
                    "layout_type": "pair",
                    "word": caption_text,
                    "front_text": "",
                    "back_text": caption_text,
                    "bbox_xyxy": pair.get("bbox_item_xyxy"),
                    "method": "pair_grid",
                    "front_image_path": self._resolve_path(picture_path),
                    "back_image_path": self._resolve_path(text_path) if text_path else None,
                    "source_ref": source_ref,
                    "confidence": confidence,
                    "needs_review": needs_review,
                    "status": status,
                    "created_at": now,
                    "updated_at": now,
                    "reasons": reasons.copy(),
                }
                
                cards.append(card)
                stats.cards_created += 1
                stats.forward_cards += 1
            
            # Generate reverse card (Text → Picture) if requested
            if direction in (CardDirection.REVERSE, CardDirection.BOTH) and caption_text:
                card_id = _generate_card_id(pair_id, page_id, caption_text, "reverse")
                
                card = {
                    "card_id": card_id,
                    "pair_id": pair_id,
                    "page_id": page_id,
                    "source_page_id": page_id,
                    "token_index": item_index,
                    "direction": "reverse",
                    "layout_type": "pair_reverse",
                    "word": caption_text,
                    "front_text": caption_text,
                    "back_text": "",
                    "bbox_xyxy": pair.get("bbox_item_xyxy"),
                    "method": "pair_grid_reverse",
                    "front_image_path": self._resolve_path(text_path) if text_path else "",
                    "back_image_path": self._resolve_path(picture_path),
                    "source_ref": source_ref,
                    "confidence": confidence,
                    "needs_review": needs_review,
                    "status": status,
                    "created_at": now,
                    "updated_at": now,
                    "reasons": reasons.copy(),
                }
                
                cards.append(card)
                stats.cards_created += 1
                stats.reverse_cards += 1
        
        return cards, stats
    
    def _resolve_path(self, path: str | None) -> str:
        """Resolve path to be relative to job_dir."""
        if not path:
            return ""
        
        path_obj = Path(path)
        
        # If already relative, return as-is
        if not path_obj.is_absolute():
            return path
        
        # Try to make relative to job_dir
        try:
            return str(path_obj.relative_to(self.job_dir))
        except ValueError:
            return path
    
    def write_result_json(self, cards: list[dict[str, Any]], job_meta: dict[str, Any]) -> Path:
        """Write cards to result.json in standard format."""
        result = {
            "job": job_meta,
            "cards": cards,
        }
        
        result_path = self.job_dir / "result.json"
        write_json(result_path, result)
        return result_path


def build_flashcards_from_pairs(
    job_dir: str | Path,
    source_name: str = "",
) -> tuple[list[dict[str, Any]], PairFlashcardStats]:
    """Convenience function to build flashcards from pair extraction results.
    
    Args:
        job_dir: Job directory containing result_pairs.json
        source_name: Optional source name for card references
        
    Returns:
        (cards_list, stats)
    """
    builder = PairFlashcardBuilder(job_dir, source_name)
    return builder.build_from_result_pairs()
