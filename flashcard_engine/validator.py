from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import zipfile
from html import unescape
from pathlib import Path
from typing import Any

from .utils import ensure_job_relative_path, load_json


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
        card_id = str(c.get("card_id") or "")
        rel = c.get("front_image_path")
        if not rel:
            continue
        try:
            p = ensure_job_relative_path(job_dir, str(rel), field="front_image_path")
        except Exception as e:
            errors.append(f"unsafe card front_image_path: card_id={card_id} path={rel} error={e}")
            missing += 1
            continue
        if not p.exists():
            errors.append(f"missing card image: card_id={card_id} path={rel}")
            missing += 1
    return missing


def _validate_cards_schema(obj: Any, errors: list[str], *, file_name: str) -> int:
    invalid = 0
    cards = (obj or {}).get("cards", []) if isinstance(obj, dict) else []
    for idx, c in enumerate(cards):
        if not isinstance(c, dict):
            errors.append(f"{file_name}: invalid card[{idx}]: not an object")
            invalid += 1
            continue

        card_id = str(c.get("card_id") or "")

        # Required keys must exist (may be None for bbox_xyxy).
        for k in (
            "card_id",
            "method",
            "bbox_xyxy",
            "front_image_path",
            "page_id",
            "source_ref",
            "source_page_id",
            "token_index",
        ):
            if k not in c:
                errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: missing field {k}")
                invalid += 1
                continue

        if not _is_sha1_hex(c.get("card_id")):
            errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: card_id not sha1 hex")
            invalid += 1

        # Types
        spid = c.get("source_page_id")
        if not isinstance(spid, str) or not spid.strip():
            errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: source_page_id must be non-empty str")
            invalid += 1

        ti = c.get("token_index")
        if type(ti) is not int:
            errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: token_index must be int")
            invalid += 1

        fip = c.get("front_image_path")
        if not isinstance(fip, str) or not fip.strip():
            errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: front_image_path must be non-empty str")
            invalid += 1

        method = c.get("method")
        if method not in ("page", "page_fallback", "bbox_crop", "segmenter", "bbox_fallback"):
            errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: method={method}")
            invalid += 1

        # Lifecycle truth: degraded fallback methods must be in review.
        needs_review = bool(c.get("needs_review"))
        reasons = c.get("reasons") or []
        if method in ("page_fallback", "bbox_fallback") and not needs_review:
            errors.append(
                f"{file_name}: invalid card[{idx}] card_id={card_id}: method={method} requires needs_review=true"
            )
            invalid += 1
        if isinstance(reasons, list) and "SEGMENT_FAILED" in reasons and not needs_review:
            errors.append(
                f"{file_name}: invalid card[{idx}] card_id={card_id}: SEGMENT_FAILED requires needs_review=true"
            )
            invalid += 1

        bbox = c.get("bbox_xyxy")
        if bbox is not None:
            ok = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
            if not ok:
                errors.append(f"{file_name}: invalid card[{idx}] card_id={card_id}: bbox_xyxy must be [int,int,int,int] or null")
                invalid += 1

    return invalid


def _validate_review_refs(job_dir: Path, obj: Any, errors: list[str]) -> int:
    items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
    missing = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        card_id = str(it.get("card_id") or "")
        rel = it.get("front_image_path")
        if not rel:
            continue
        try:
            p = ensure_job_relative_path(job_dir, str(rel), field="front_image_path")
        except Exception as e:
            errors.append(f"unsafe review front_image_path: card_id={card_id} path={rel} error={e}")
            missing += 1
            continue
        if not p.exists():
            errors.append(f"missing review image: card_id={card_id} path={rel}")
            missing += 1
    return missing


def _validate_review_schema(obj: Any, errors: list[str], *, file_name: str) -> int:
    invalid = 0
    items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"{file_name}: invalid review[{idx}]: not an object")
            invalid += 1
            continue

        card_id = str(it.get("card_id") or "")

        for k in ("card_id", "review_reason", "page_id", "source_ref", "front_image_path"):
            if k not in it:
                errors.append(f"{file_name}: invalid review[{idx}] card_id={card_id}: missing field {k}")
                invalid += 1
                continue

        if not _is_sha1_hex(it.get("card_id")):
            errors.append(f"{file_name}: invalid review[{idx}] card_id={card_id}: card_id not sha1 hex")
            invalid += 1

        rr = it.get("review_reason")
        if not isinstance(rr, str) or not rr.strip():
            errors.append(f"{file_name}: invalid review[{idx}] card_id={card_id}: review_reason must be non-empty str")
            invalid += 1

        fip = it.get("front_image_path")
        if not isinstance(fip, str) or not fip.strip():
            errors.append(f"{file_name}: invalid review[{idx}] card_id={card_id}: front_image_path must be non-empty str")
            invalid += 1

        bbox = it.get("bbox_xyxy")
        if bbox is not None:
            ok = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
            if not ok:
                errors.append(f"{file_name}: invalid review[{idx}] card_id={card_id}: bbox_xyxy must be [int,int,int,int] or null")
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

        # Job metadata must be present and indicate completion.
        job_obj = result.get("job") if isinstance(result, dict) else None
        if not isinstance(job_obj, dict):
            errors.append("result.json: missing/invalid job object")
        else:
            for k in ("job_id", "source", "input", "created_at"):
                if k not in job_obj:
                    errors.append(f"result.json: job missing field {k}")
            if isinstance(job_obj.get("input"), dict):
                for k in ("type", "path"):
                    if k not in job_obj.get("input"):
                        errors.append(f"result.json: job.input missing field {k}")
            else:
                errors.append("result.json: job.input must be an object")

        missing_images += _validate_image_refs(job_dir, result, errors)
        invalid_cards += _validate_cards_schema(result, errors, file_name="result.json")
    except Exception as e:
        errors.append(f"failed to read result.json: {e}")
        invalid_cards += 1

    # metrics.json must be valid JSON and indicate completion.
    try:
        metrics = load_json(job_dir / "metrics.json")
        if not isinstance(metrics, dict):
            errors.append("metrics.json: must be an object")
        else:
            for k in ("created_at", "pages_total", "pages_processed", "cards_total", "review_items_total"):
                if k not in metrics:
                    errors.append(f"metrics.json: missing field {k}")
            finished = metrics.get("finished")
            completed_at = metrics.get("completed_at")
            if finished is not True:
                errors.append("metrics.json: job not finished (finished!=true)")
            if not isinstance(completed_at, str) or not completed_at.strip():
                errors.append("metrics.json: missing/invalid completed_at")

            # Basic invariants
            try:
                pt = int(metrics.get("pages_total") or 0)
                pp = int(metrics.get("pages_processed") or 0)
                if pp < 0 or pt < 0 or pp > pt:
                    errors.append(f"metrics.json: invalid pages_processed/pages_total: {pp}/{pt}")
            except Exception:
                errors.append("metrics.json: pages_total/pages_processed must be ints")
    except Exception as e:
        errors.append(f"failed to read metrics.json: {e}")

    try:
        review = load_json(job_dir / "review_queue.json")
        missing_images += _validate_review_refs(job_dir, review, errors)
        invalid_review_items += _validate_review_schema(review, errors, file_name="review_queue.json")
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
        - Contains collection.anki2 and media mapping file
        - Parse SQLite notes and extract referenced media filenames (e.g., <img src="...")
        - For each referenced filename, ensure it exists in media mapping values
            and that the corresponding numeric media entry exists in the zip.

        This is optional and does not alter Output Contract.
    """

    job_dir = Path(job_dir)
    apkg_path = Path(apkg_path)

    errors: list[str] = []
    warnings: list[str] = []

    if not apkg_path.exists() or not apkg_path.is_file():
        errors.append(f"apkg_missing: {apkg_path}")
        return False, {"errors": errors}

    try:
        with zipfile.ZipFile(apkg_path, "r") as z:
            names = set(z.namelist())
            if "collection.anki2" not in names:
                errors.append("apkg_missing_collection.anki2")

            if "media" not in names:
                errors.append("apkg_missing_media_mapping")
                ok = not errors
                return ok, {"errors": errors}

            media_raw = z.read("media")
            try:
                media_txt = media_raw.decode("utf-8")
            except Exception:
                media_txt = media_raw.decode("utf-8", errors="replace")

            try:
                media_map = json.loads(media_txt)
            except Exception as e:
                errors.append(f"apkg_media_mapping_invalid_json: {e}")
                ok = not errors
                return ok, {"errors": errors}

            if not isinstance(media_map, dict):
                errors.append("apkg_media_mapping_not_object")
                ok = not errors
                return ok, {"errors": errors}

            # Normalize mapping keys/values to strings
            media_index_to_name: dict[str, str] = {}
            for k, v in media_map.items():
                if k is None or v is None:
                    continue
                ks = str(k)
                vs = str(v)
                if not ks:
                    continue
                media_index_to_name[ks] = vs

            # Detect duplicate filenames mapping to multiple indices (safest: fail)
            name_to_indices: dict[str, list[str]] = {}
            for idx, fn in media_index_to_name.items():
                name_to_indices.setdefault(fn, []).append(idx)
            dupes = {fn: idxs for fn, idxs in name_to_indices.items() if len(idxs) > 1}
            if dupes:
                # Anki packages can legally map multiple indices to the same filename.
                # Treat as warning, but validate that at least one corresponding blob exists.
                sample = list(dupes.items())[:10]
                details = "; ".join([f"{fn}=>{idxs}" for fn, idxs in sample])
                warnings.append(f"apkg_media_mapping_duplicate_filenames: {details}")

            # Load SQLite collection to find referenced media names.
            col_bytes = z.read("collection.anki2")
            with tempfile.NamedTemporaryFile(prefix="apkg_collection_", suffix=".anki2", delete=False) as tf:
                tf.write(col_bytes)
                tmp_path = tf.name

            try:
                conn = sqlite3.connect(tmp_path)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT flds FROM notes")
                    rows = cur.fetchall()
                finally:
                    conn.close()
            except Exception as e:
                errors.append(f"apkg_sqlite_read_failed: {e}")
                ok = not errors
                return ok, {"errors": errors}
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

            img_re = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
            sound_re = re.compile(r"\[sound:([^\]]+)\]", flags=re.IGNORECASE)

            referenced: set[str] = set()
            for (flds,) in rows:
                if not isinstance(flds, str):
                    try:
                        flds = str(flds)
                    except Exception:
                        continue

                for m in img_re.finditer(flds):
                    src = unescape(m.group(1)).strip()
                    if not src:
                        continue
                    referenced.add(Path(src).name)

                for m in sound_re.finditer(flds):
                    src = unescape(m.group(1)).strip()
                    if not src:
                        continue
                    referenced.add(Path(src).name)

            # Validate each referenced filename exists in mapping and blob exists in zip.
            missing_in_mapping: list[str] = []
            missing_blob: list[str] = []
            for fn in sorted(referenced):
                idxs = name_to_indices.get(fn) or []
                if not idxs:
                    missing_in_mapping.append(fn)
                    continue
                # Allow any index for this filename to satisfy blob presence.
                if not any(str(idx) in names for idx in idxs):
                    missing_blob.append(f"{fn} (indices={idxs})")

            if missing_in_mapping:
                errors.append("apkg_missing_media_mapping_filenames: " + ", ".join(missing_in_mapping[:50]))
            if missing_blob:
                errors.append("apkg_missing_media_blobs: " + ", ".join(missing_blob[:50]))
    except zipfile.BadZipFile:
        errors.append("apkg_invalid_zip")
    except Exception as e:
        errors.append(f"apkg_validate_failed: {e}")

    ok = not errors
    return ok, {
        "referenced_filenames": sorted(list(referenced)) if 'referenced' in locals() else [],
        "warnings": warnings,
        "errors": errors,
    }
