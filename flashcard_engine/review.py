from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json, utc_now_iso, write_json
from .learning import capture_learning_data_from_feedback, LearningDataStats


@dataclass
class ApplyReviewStats:
    feedback_items: int = 0
    applied: int = 0
    skipped_unknown_card: int = 0
    skipped_already_applied: int = 0
    # Learning data stats
    learning_records_written: int = 0


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

    feedback_obj = load_json(feedback_path)
    if isinstance(feedback_obj, list):
        feedback_items = feedback_obj
    elif isinstance(feedback_obj, dict):
        items = feedback_obj.get("items")
        if not isinstance(items, list):
            raise ValueError("review_feedback.json object must contain list field: items")
        feedback_items = items
    else:
        raise ValueError("review_feedback.json must be a list or an object with items")

    stats = ApplyReviewStats(feedback_items=len(feedback_items))

    for entry in feedback_items:
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

        # Lifecycle hardening: do not allow rejected -> active via approve/edit.
        if card.get("status") == "rejected" and action in ("approve", "edit"):
            stats.skipped_already_applied += 1
            review_by_id.pop(card_id, None)
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

        # Handle edit as "edit then approve". Must be idempotent.
        if action == "edit":
            if not isinstance(edited_text, str) or not edited_text.strip():
                # treat invalid edit as no-op
                stats.skipped_already_applied += 1
                continue

            normalized_text = edited_text.strip()

            # If it's already active and already equals the edited text, do nothing (idempotent)
            # but ensure the review item is cleaned up.
            if (
                card.get("status") == "active"
                and not card.get("needs_review")
                and str(card.get("word") or "").strip() == normalized_text
            ):
                stats.skipped_already_applied += 1
                review_by_id.pop(card_id, None)
                continue

            # Apply edit (even if the text happens to be the same while still in review)
            card["word"] = normalized_text
            action = "approve"

        now = utc_now_iso()

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

    # Capture learning data from all feedback items (for model training)
    try:
        learning_stats = capture_learning_data_from_feedback(
            job_dir=job_dir,
            cards_by_id=cards_by_id,
            feedback_items=feedback_items,
        )
        stats.learning_records_written = learning_stats.records_written
    except Exception:
        # Fail-soft: learning data capture should not break review workflow
        pass

    # Reassemble stable outputs
    result_out = dict(result) if isinstance(result, dict) else {"job": {}, "cards": []}
    result_out["cards"] = list(cards_by_id.values())

    review_out = dict(review) if isinstance(review, dict) else {"items": []}
    # Ensure rejected cards never appear in review output.
    filtered_items: list[dict[str, Any]] = []
    for it in review_by_id.values():
        cid = str(it.get("card_id") or "")
        c = cards_by_id.get(cid)
        if c is not None and c.get("status") == "rejected":
            continue
        filtered_items.append(it)
    review_out["items"] = filtered_items

    write_json(result_path, result_out)
    write_json(review_path, review_out)

    return stats
