from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from .utils import load_json


def _is_sha1_hex(s: Any) -> bool:
    if not isinstance(s, str) or len(s) != 40:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False


def _validate_image_refs(job_dir: Path, obj: Any, errors: list[str]) -> int:
    cards = (obj or {}).get("cards", []) if isinstance(obj, dict) else []
    missing = 0
    for c in cards:
        if not isinstance(c, dict):
            continue
        rel = c.get("front_image_path")
        if not rel:
            continue
        p = job_dir / str(rel)
        if not p.exists():
            errors.append(f"missing card image: {rel}")
            missing += 1
    return missing


def _validate_cards_schema(obj: Any, errors: list[str]) -> int:
    invalid = 0
    cards = (obj or {}).get("cards", []) if isinstance(obj, dict) else []
    for idx, c in enumerate(cards):
        if not isinstance(c, dict):
            errors.append(f"invalid card[{idx}]: not an object")
            invalid += 1
            continue

        # Required keys must exist (may be None for bbox_xyxy).
        for k in ("card_id", "method", "bbox_xyxy", "front_image_path", "page_id", "source_ref"):
            if k not in c:
                errors.append(f"invalid card[{idx}]: missing field {k}")
                invalid += 1
                continue

        if not _is_sha1_hex(c.get("card_id")):
            errors.append(f"invalid card[{idx}]: card_id not sha1 hex")
            invalid += 1

        method = c.get("method")
        if method not in ("page", "bbox_crop", "segmenter"):
            errors.append(f"invalid card[{idx}]: method={method}")
            invalid += 1

        bbox = c.get("bbox_xyxy")
        if bbox is not None:
            ok = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
            if not ok:
                errors.append(f"invalid card[{idx}]: bbox_xyxy must be [int,int,int,int] or null")
                invalid += 1

    return invalid


def _validate_review_refs(job_dir: Path, obj: Any, errors: list[str]) -> int:
    items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
    missing = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        rel = it.get("front_image_path")
        if not rel:
            continue
        p = job_dir / str(rel)
        if not p.exists():
            errors.append(f"missing review image: {rel}")
            missing += 1
    return missing


def _validate_review_schema(obj: Any, errors: list[str]) -> int:
    invalid = 0
    items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"invalid review[{idx}]: not an object")
            invalid += 1
            continue

        for k in ("card_id", "review_reason", "page_id", "source_ref"):
            if k not in it:
                errors.append(f"invalid review[{idx}]: missing field {k}")
                invalid += 1
                continue

        if not _is_sha1_hex(it.get("card_id")):
            errors.append(f"invalid review[{idx}]: card_id not sha1 hex")
            invalid += 1

        bbox = it.get("bbox_xyxy")
        if bbox is not None:
            ok = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
            if not ok:
                errors.append(f"invalid review[{idx}]: bbox_xyxy must be [int,int,int,int] or null")
                invalid += 1

    return invalid


def validate_job_dir(job_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    job_dir = Path(job_dir)
    errors: list[str] = []

    missing_contract_files = 0
    missing_images = 0
    invalid_cards = 0
    invalid_review_items = 0

    for f in ("result.json", "review_queue.json", "metrics.json", "errors.jsonl"):
        p = job_dir / f
        if not p.exists():
            missing_contract_files += 1
            errors.append(f"missing: {p}")

    try:
        result = load_json(job_dir / "result.json")
        missing_images += _validate_image_refs(job_dir, result, errors)
        invalid_cards += _validate_cards_schema(result, errors)
    except Exception as e:
        errors.append(f"failed to read result.json: {e}")
        invalid_cards += 1

    try:
        review = load_json(job_dir / "review_queue.json")
        missing_images += _validate_review_refs(job_dir, review, errors)
        invalid_review_items += _validate_review_schema(review, errors)
    except Exception as e:
        errors.append(f"failed to read review_queue.json: {e}")
        invalid_review_items += 1

    summary: dict[str, Any] = {
        "missing_contract_files": missing_contract_files,
        "missing_images": missing_images,
        "invalid_cards": invalid_cards,
        "invalid_review_items": invalid_review_items,
        "errors": errors,
    }

    ok = not errors
    return ok, summary


def validate_apkg(job_dir: str | Path, apkg_path: str | Path) -> tuple[bool, dict[str, Any]]:
    """Validate an Anki .apkg file.

    Rules:
    - File exists and is a valid zip
    - Contains collection.anki2
    - Media file count >= number of active cards with existing images in job_dir

    This is optional and does not alter Output Contract.
    """

    job_dir = Path(job_dir)
    apkg_path = Path(apkg_path)

    errors: list[str] = []

    if not apkg_path.exists() or not apkg_path.is_file():
        errors.append(f"apkg_missing: {apkg_path}")
        return False, {"errors": errors}

    # Expected media count based on job_dir result.json
    expected_media = 0
    try:
        result = load_json(job_dir / "result.json")
        cards = result.get("cards", []) if isinstance(result, dict) else []
        for c in cards:
            if not isinstance(c, dict):
                continue
            if str(c.get("status") or "") != "active":
                continue
            rel = str(c.get("front_image_path") or "").strip()
            if not rel:
                continue
            if (job_dir / rel).exists():
                expected_media += 1
    except Exception as e:
        errors.append(f"apkg_expected_media_failed: {e}")

    try:
        with zipfile.ZipFile(apkg_path, "r") as z:
            names = set(z.namelist())
            if "collection.anki2" not in names:
                errors.append("apkg_missing_collection.anki2")
            # genanki packs media as numbered files plus a 'media' mapping file
            media_count = sum(1 for n in names if n.isdigit())
            if media_count < expected_media:
                errors.append(f"apkg_media_too_few: media_count={media_count} expected_min={expected_media}")
    except zipfile.BadZipFile:
        errors.append("apkg_invalid_zip")
    except Exception as e:
        errors.append(f"apkg_validate_failed: {e}")

    ok = not errors
    return ok, {"expected_media": expected_media, "errors": errors}
