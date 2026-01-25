from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json, utc_now_iso, write_json


@dataclass
class ApplyReviewStats:
    feedback_items: int = 0
    applied: int = 0
    skipped_unknown_card: int = 0
    skipped_already_applied: int = 0


def _normalize_card_for_v03(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    now = utc_now_iso()
    out.setdefault("created_at", now)
    out.setdefault("updated_at", now)
    out.setdefault("source_page_id", out.get("page_id"))
    out.setdefault("status", "review" if out.get("needs_review") else "active")
    return out


def apply_review_feedback(*, job_dir: str | Path, feedback_path: str | Path) -> ApplyReviewStats:
    """Apply human review feedback to a job directory.

    Feedback JSON format:
    [
      {"card_id": "...", "action": "approve|reject|edit", "edited_text": "..."}
    ]

    Behavior:
    - approve: mark card active, needs_review=false, remove from review_queue
    - reject: mark card rejected, remove from review_queue
    - edit: update word then approve

    Idempotent:
    - Re-running on already-applied actions should not duplicate cards.
    """
    job_dir = Path(job_dir)
    result_path = job_dir / "result.json"
    review_path = job_dir / "review_queue.json"

    result = load_json(result_path)
    review = load_json(review_path)

    cards = result.get("cards", []) if isinstance(result, dict) else []
    review_items = review.get("items", []) if isinstance(review, dict) else []

    cards_by_id: dict[str, dict[str, Any]] = {}
    for c in cards:
        if isinstance(c, dict) and c.get("card_id"):
            cards_by_id[str(c["card_id"])] = _normalize_card_for_v03(c)

    review_by_id: dict[str, dict[str, Any]] = {}
    for it in review_items:
        if isinstance(it, dict) and it.get("card_id"):
            review_by_id[str(it["card_id"])] = dict(it)

    feedback = load_json(feedback_path)
    if not isinstance(feedback, list):
        raise ValueError("review_feedback.json must be a list")

    stats = ApplyReviewStats(feedback_items=len(feedback))

    for entry in feedback:
        if not isinstance(entry, dict):
            continue
        card_id = str(entry.get("card_id") or "")
        action = str(entry.get("action") or "").lower()
        edited_text = entry.get("edited_text")

        if not card_id:
            continue

        card = cards_by_id.get(card_id)
        if card is None:
            # For now, only support reviewing existing cards.
            stats.skipped_unknown_card += 1
            continue

        # Detect already-applied
        if action == "approve" and card.get("status") == "active" and not card.get("needs_review"):
            stats.skipped_already_applied += 1
            review_by_id.pop(card_id, None)
            continue
        if action == "reject" and card.get("status") == "rejected":
            stats.skipped_already_applied += 1
            review_by_id.pop(card_id, None)
            continue

        if action == "edit":
            if not isinstance(edited_text, str) or not edited_text.strip():
                # treat invalid edit as no-op
                stats.skipped_already_applied += 1
                continue
            normalized_text = edited_text.strip()
            if (
                card.get("status") == "active"
                and not card.get("needs_review")
                and str(card.get("word") or "").strip() == normalized_text
            ):
                stats.skipped_already_applied += 1
                review_by_id.pop(card_id, None)
                continue
            continue

        now = utc_now_iso()

        if action == "edit":
            # At this point normalized_text is valid and differs (or card isn't active yet).
            card["word"] = normalized_text
            action = "approve"

        if action == "approve":
            card["status"] = "active"
            card["needs_review"] = False
            card["reasons"] = []
            card["updated_at"] = now
            review_by_id.pop(card_id, None)
            stats.applied += 1
            continue

        if action == "reject":
            card["status"] = "rejected"
            card["needs_review"] = False
            card["updated_at"] = now
            review_by_id.pop(card_id, None)
            stats.applied += 1
            continue

    # Reassemble stable outputs
    result_out = dict(result) if isinstance(result, dict) else {"job": {}, "cards": []}
    result_out["cards"] = list(cards_by_id.values())

    review_out = dict(review) if isinstance(review, dict) else {"items": []}
    review_out["items"] = list(review_by_id.values())

    write_json(result_path, result_out)
    write_json(review_path, review_out)

    return stats
