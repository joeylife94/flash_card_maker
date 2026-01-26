from __future__ import annotations

import hashlib
import html
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..job import JobPaths, record_error
from ..utils import ensure_job_relative_path, load_json
@dataclass
class ApkgExportStats:
    cards_seen: int = 0
    cards_exported: int = 0
    cards_skipped_inactive: int = 0
    cards_skipped_missing_image: int = 0
    cards_invalid: int = 0
    deck_name: str | None = None
def _normalize_card(card: dict[str, Any]) -> dict[str, Any]:
    # v0.3+/v0.4: tolerate older jobs missing new fields.
    out = dict(card)
    out.setdefault("status", "active" if not out.get("needs_review") else "review")
    out.setdefault("source_page_id", out.get("page_id"))
    out.setdefault("token_index", 0)
    out.setdefault("created_at", None)
    out.setdefault("updated_at", None)
    return out


def _stable_int_id(s: str) -> int:
    # genanki ids must be int; keep stable across runs.
    digest = hashlib.sha1(s.encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], "big", signed=False)
    return n % (2**31 - 1)


def _parse_tags(tags_csv: str | None) -> list[str]:
    if not tags_csv:
        return []
    tags: list[str] = []
    for t in tags_csv.split(","):
        t = t.strip()
        if not t:
            continue
        # Anki tags should not contain spaces; normalize a bit.
        tags.append(t.replace(" ", "_"))
    return tags


def export_apkg(
    *,
    job_dir: str | Path,
    out_path: str | Path,
    deck_name: str | None = None,
    tags: str | None = None,
) -> ApkgExportStats:
    """Export approved cards as an Anki .apkg with embedded media.

    Rules (v0.4):
    - Export ONLY cards with status == active
    - Exclude rejected cards
    - Preserve order: page_id then token_index
    - Missing image => skip card + warning; continue
    - If 0 cards exported => error (exit non-zero at CLI)

    Model:
    - Fields: Front, Back
    - Front contains image + text
    - Back contains text (currently empty is allowed)
    """

    try:
        import genanki  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "genanki is required for apkg export. Install with: pip install genanki"
        ) from e

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

    stats = ApkgExportStats()

    result = load_json(paths.result_json)
    cards = result.get("cards", []) if isinstance(result, dict) else []

    normalized: list[dict[str, Any]] = []
    for c in cards:
        if not isinstance(c, dict):
            stats.cards_invalid += 1
            continue
        normalized.append(_normalize_card(c))

    # v0.4.1 deterministic ordering: source_page_id ASC, token_index ASC
    # (No path parsing for ordering.)
    normalized.sort(key=lambda c: (str(c.get("source_page_id") or ""), int(c.get("token_index") or 0)))

    # Determine default deck name from first card's source_ref
    if deck_name is None:
        for c in normalized:
            sr = str(c.get("source_ref") or "").strip()
            if sr:
                deck_name = sr
                break
        if deck_name is None:
            deck_name = job_dir.name

    stats.deck_name = deck_name

    export_tags = _parse_tags(tags)

    model = genanki.Model(
        _stable_int_id(f"flashcard_engine:model:{deck_name}"),
        "flashcard_engine_basic_image",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
            }
        ],
    )

    deck = genanki.Deck(_stable_int_id(f"flashcard_engine:deck:{deck_name}"), deck_name)

    media_tmp = job_dir / "export_media_tmp"
    if media_tmp.exists():
        shutil.rmtree(media_tmp)
    media_tmp.mkdir(parents=True, exist_ok=True)

    used_media_names: dict[str, Path] = {}
    media_files: list[str] = []

    unsafe_paths: list[str] = []

    def add_media(*, card_id: str, rel_path: str) -> str:
        src_abs = ensure_job_relative_path(job_dir, rel_path, field="front_image_path")
        base = Path(rel_path).name
        media_name = base

        if media_name in used_media_names and used_media_names[media_name] != src_abs:
            media_name = f"{card_id}_{base}"

        # If this exact media_name already points to the same source file, reuse it
        # and do not add duplicate media entries.
        if media_name in used_media_names and used_media_names[media_name] == src_abs:
            return media_name

        dst = media_tmp / media_name
        shutil.copyfile(src_abs, dst)
        used_media_names[media_name] = src_abs
        media_files.append(str(dst))
        return media_name

    for c in normalized:
        stats.cards_seen += 1

        status = str(c.get("status") or "")
        if status != "active":
            stats.cards_skipped_inactive += 1
            continue

        front_image_path = str(c.get("front_image_path") or "").strip()
        if not front_image_path:
            stats.cards_skipped_missing_image += 1
            record_error(paths, page_id=str(c.get("page_id")), stage="export", message="missing_image_path")
            continue

        try:
            img_abs = ensure_job_relative_path(job_dir, front_image_path, field="front_image_path")
        except Exception as e:
            unsafe_paths.append(
                f"unsafe_front_image_path card_id={str(c.get('card_id') or '')} path={front_image_path} error={e}"
            )
            continue

        if not img_abs.exists():
            stats.cards_skipped_missing_image += 1
            record_error(
                paths,
                page_id=str(c.get("page_id")),
                stage="export",
                message=f"missing_image: {front_image_path}",
            )
            continue

        card_id = str(c.get("card_id") or "")
        try:
            media_name = add_media(card_id=card_id or "unknown", rel_path=front_image_path)
        except Exception as e:
            unsafe_paths.append(
                f"unsafe_front_image_path card_id={card_id} path={front_image_path} error={e}"
            )
            continue

        front_text = str(c.get("word") or "")
        safe_text = html.escape(front_text)

        front_html = f'<img src="{html.escape(media_name)}"><br>{safe_text}' if safe_text else f'<img src="{html.escape(media_name)}">'

        note = genanki.Note(model=model, fields=[front_html, ""])  # back text empty for now
        if export_tags:
            note.tags = export_tags
        deck.add_note(note)
        stats.cards_exported += 1

    if unsafe_paths:
        details = "\n".join(unsafe_paths[:20])
        more = "" if len(unsafe_paths) <= 20 else f"\n...and {len(unsafe_paths) - 20} more"
        raise RuntimeError(f"Unsafe front_image_path detected; refusing to export.\n{details}{more}")

    if stats.cards_exported <= 0:
        raise RuntimeError("No active cards exported (status=active). Did you apply-review approve/edit?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pkg = genanki.Package(deck)
    pkg.media_files = media_files
    pkg.write_to_file(str(out_path))

    return stats
