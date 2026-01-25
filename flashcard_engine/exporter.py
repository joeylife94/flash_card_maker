from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .job import JobPaths, record_error
from .utils import load_json


_TOKEN_RE = re.compile(r"token_(\d{4})_")


@dataclass
class ExportStats:
    cards_seen: int = 0
    cards_exported: int = 0
    cards_skipped_review: int = 0
    cards_skipped_missing_image: int = 0
    cards_invalid: int = 0


def _token_index_from_path(front_image_path: str | None) -> int:
    if not front_image_path:
        return 10**9
    m = _TOKEN_RE.search(front_image_path)
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except Exception:
        return 10**9


def _page_sort_key(page_id: str) -> int:
    # page_001 -> 1, else stable fallback
    try:
        if page_id.startswith("page_"):
            return int(page_id.split("_")[-1])
    except Exception:
        pass
    return 10**9


def _normalize_card(card: dict[str, Any]) -> dict[str, Any]:
    # v0.3: tolerate older jobs missing new fields.
    out = dict(card)
    out.setdefault("status", "active" if not out.get("needs_review") else "review")
    out.setdefault("source_page_id", out.get("page_id"))
    out.setdefault("created_at", None)
    out.setdefault("updated_at", None)
    return out


def export_csv(
    *,
    job_dir: str | Path,
    out_path: str | Path,
    include_review: bool = False,
) -> ExportStats:
    """Export cards to CSV.

    Rules:
    - By default exports only cards with needs_review == False AND status != rejected
    - If include_review is True, includes review cards as well (still excludes rejected)
    - Missing images => skip card + warning to errors.jsonl
    - Fail-soft: raises only if it cannot export any valid cards

    CSV columns:
    - front_text
    - back_text (empty)
    - front_image_path
    - source_ref
    - card_id
    - review_reason
    """
    job_dir = Path(job_dir)
    out_path = Path(out_path)

    paths = JobPaths(
        job_dir=job_dir,
        input_dir=job_dir / "input",
        pages_dir=job_dir / "pages",
        crops_dir=job_dir / "pages" / "crops",
        stage_ocr_dir=job_dir / "stage" / "ocr",
        stage_layout_dir=job_dir / "stage" / "layout",
        stage_segment_dir=job_dir / "stage" / "segment",
        result_json=job_dir / "result.json",
        review_json=job_dir / "review_queue.json",
        metrics_json=job_dir / "metrics.json",
        errors_jsonl=job_dir / "errors.jsonl",
    )

    stats = ExportStats()

    result = load_json(paths.result_json)
    cards = result.get("cards", []) if isinstance(result, dict) else []

    normalized: list[dict[str, Any]] = []
    for c in cards:
        if not isinstance(c, dict):
            stats.cards_invalid += 1
            continue
        normalized.append(_normalize_card(c))

    # Stable ordering: page_id then token order (best-effort)
    normalized.sort(
        key=lambda c: (
            _page_sort_key(str(c.get("page_id", ""))),
            _token_index_from_path(str(c.get("front_image_path") or "")),
        )
    )

    rows: list[dict[str, str]] = []
    for c in normalized:
        stats.cards_seen += 1

        if c.get("status") == "rejected":
            stats.cards_skipped_review += 1
            continue

        needs_review = bool(c.get("needs_review"))
        if needs_review and not include_review:
            stats.cards_skipped_review += 1
            continue

        front_image_path = str(c.get("front_image_path") or "")
        if front_image_path:
            img_abs = job_dir / front_image_path
            if not img_abs.exists():
                stats.cards_skipped_missing_image += 1
                record_error(paths, page_id=str(c.get("page_id")), stage="export", message=f"missing_image: {front_image_path}")
                continue

        reasons = c.get("reasons") or []
        review_reason = ""
        if isinstance(reasons, list) and reasons:
            review_reason = str(reasons[0])

        rows.append(
            {
                "front_text": str(c.get("word") or ""),
                "back_text": "",
                "front_image_path": front_image_path,
                "source_ref": str(c.get("source_ref") or ""),
                "card_id": str(c.get("card_id") or ""),
                "review_reason": review_reason if needs_review else "",
            }
        )

    if not rows:
        raise RuntimeError("No exportable cards (all skipped or invalid)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["front_text", "back_text", "front_image_path", "source_ref", "card_id", "review_reason"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            stats.cards_exported += 1

    return stats
