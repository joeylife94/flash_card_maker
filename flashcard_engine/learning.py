"""Human-in-the-loop Learning Data Pipeline.

Extends review feedback handling to capture training data for future
model improvement. When humans approve, reject, or edit cards, this
module stores machine-readable records suitable for model training.

Learning records contain:
- Original predictions (bbox, text)
- Human corrections
- Failure reasons
- Page image references
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import load_json, utc_now_iso, write_json, append_jsonl


@dataclass
class LearningRecord:
    """A single training record from human review."""
    record_id: str
    card_id: str
    page_id: str
    page_image_path: str
    
    # Original predictions
    predicted_bbox_xyxy: list[int] | None
    predicted_text: str
    predicted_confidence: float
    
    # Human corrections
    corrected_bbox_xyxy: list[int] | None
    corrected_text: str | None
    
    # Review action
    action: str  # approve, reject, edit
    
    # Context
    failure_reasons: list[str]
    layout_type: str
    method: str
    
    # Metadata
    created_at: str
    source_ref: str


@dataclass
class LearningDataStats:
    """Statistics for learning data capture."""
    records_written: int = 0
    approvals: int = 0
    rejections: int = 0
    edits: int = 0
    bbox_corrections: int = 0
    text_corrections: int = 0


def _generate_record_id(card_id: str, action: str, timestamp: str) -> str:
    """Generate a unique record ID."""
    import hashlib
    payload = f"{card_id}|{action}|{timestamp}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class LearningDataPipeline:
    """Pipeline for capturing and storing learning data from human review."""
    
    def __init__(self, job_dir: str | Path):
        self.job_dir = Path(job_dir)
        self.learning_dir = self.job_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Main learning data file
        self.records_file = self.learning_dir / "training_records.jsonl"
        # Summary statistics
        self.stats_file = self.learning_dir / "learning_stats.json"
        # Batch exports for model training
        self.exports_dir = self.learning_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_review_action(
        self,
        card: dict[str, Any],
        action: str,
        edited_text: str | None = None,
        corrected_bbox: list[int] | None = None,
    ) -> LearningRecord:
        """Capture a single review action as a training record.
        
        Args:
            card: The card being reviewed (from result.json)
            action: Review action (approve, reject, edit)
            edited_text: New text if action is edit
            corrected_bbox: Corrected bounding box if provided
            
        Returns:
            The created LearningRecord
        """
        now = utc_now_iso()
        
        record = LearningRecord(
            record_id=_generate_record_id(card.get("card_id", ""), action, now),
            card_id=str(card.get("card_id", "")),
            page_id=str(card.get("page_id", "")),
            page_image_path=str(card.get("front_image_path", "")),
            
            predicted_bbox_xyxy=card.get("bbox_xyxy"),
            predicted_text=str(card.get("word", "")),
            predicted_confidence=float(card.get("confidence", 0.0)),
            
            corrected_bbox_xyxy=corrected_bbox,
            corrected_text=edited_text if action == "edit" else None,
            
            action=action,
            
            failure_reasons=card.get("reasons", []),
            layout_type=str(card.get("layout_type", "unknown")),
            method=str(card.get("method", "unknown")),
            
            created_at=now,
            source_ref=str(card.get("source_ref", "")),
        )
        
        # Append to JSONL file
        append_jsonl(self.records_file, asdict(record))
        
        # Update stats
        self._update_stats(record)
        
        return record
    
    def _update_stats(self, record: LearningRecord) -> None:
        """Update learning statistics."""
        stats = self._load_stats()
        
        stats["records_written"] = stats.get("records_written", 0) + 1
        
        if record.action == "approve":
            stats["approvals"] = stats.get("approvals", 0) + 1
        elif record.action == "reject":
            stats["rejections"] = stats.get("rejections", 0) + 1
        elif record.action == "edit":
            stats["edits"] = stats.get("edits", 0) + 1
            if record.corrected_text:
                stats["text_corrections"] = stats.get("text_corrections", 0) + 1
        
        if record.corrected_bbox_xyxy:
            stats["bbox_corrections"] = stats.get("bbox_corrections", 0) + 1
        
        stats["updated_at"] = utc_now_iso()
        
        write_json(self.stats_file, stats)
    
    def _load_stats(self) -> dict[str, Any]:
        """Load or initialize stats."""
        if self.stats_file.exists():
            try:
                return load_json(self.stats_file)
            except Exception:
                pass
        return {
            "created_at": utc_now_iso(),
            "records_written": 0,
            "approvals": 0,
            "rejections": 0,
            "edits": 0,
            "bbox_corrections": 0,
            "text_corrections": 0,
        }
    
    def load_all_records(self) -> list[LearningRecord]:
        """Load all learning records from JSONL file."""
        records = []
        if not self.records_file.exists():
            return records
        
        with open(self.records_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(LearningRecord(**data))
                except Exception:
                    continue
        
        return records
    
    def export_training_batch(self, format: str = "jsonl") -> Path:
        """Export training records in a format suitable for model training.
        
        Args:
            format: Export format ("jsonl" or "json")
            
        Returns:
            Path to the exported file
        """
        records = self.load_all_records()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            export_path = self.exports_dir / f"training_batch_{timestamp}.json"
            export_data = {
                "exported_at": utc_now_iso(),
                "records_count": len(records),
                "records": [asdict(r) for r in records],
            }
            write_json(export_path, export_data)
        else:
            # Default: JSONL
            export_path = self.exports_dir / f"training_batch_{timestamp}.jsonl"
            with open(export_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        
        return export_path
    
    def get_stats(self) -> dict[str, Any]:
        """Get current learning statistics."""
        return self._load_stats()


def capture_learning_data_from_feedback(
    job_dir: str | Path,
    cards_by_id: dict[str, dict[str, Any]],
    feedback_items: list[dict[str, Any]],
) -> LearningDataStats:
    """Capture learning data from a batch of feedback items.
    
    This function is called during apply_review_feedback to capture
    training data for all review actions.
    
    Args:
        job_dir: Job directory path
        cards_by_id: Dictionary of card_id -> card data
        feedback_items: List of feedback entries
        
    Returns:
        Statistics about captured data
    """
    pipeline = LearningDataPipeline(job_dir)
    stats = LearningDataStats()
    
    for entry in feedback_items:
        if not isinstance(entry, dict):
            continue
        
        card_id = str(entry.get("card_id") or "")
        action = str(entry.get("action") or "").lower()
        edited_text = entry.get("edited_text")
        corrected_bbox = entry.get("corrected_bbox")
        
        if not card_id or action not in ("approve", "reject", "edit"):
            continue
        
        card = cards_by_id.get(card_id)
        if card is None:
            continue
        
        # Capture the learning record
        pipeline.capture_review_action(
            card=card,
            action=action,
            edited_text=edited_text if action == "edit" else None,
            corrected_bbox=corrected_bbox,
        )
        
        stats.records_written += 1
        if action == "approve":
            stats.approvals += 1
        elif action == "reject":
            stats.rejections += 1
        elif action == "edit":
            stats.edits += 1
            if edited_text:
                stats.text_corrections += 1
        
        if corrected_bbox:
            stats.bbox_corrections += 1
    
    return stats
