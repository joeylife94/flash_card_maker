from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .builder import FlashcardBuilder
from .cleaner import TextCleaner
from .config import EngineConfig
from .cropper import crop_multiword_tokens_for_page
from .job import JobPaths, record_error
from .layout import LayoutClassifier
from .ocr import OCRExtractor
from .page_provider import PageProvider
from .segmenter import Segmenter
from .utils import utc_now_iso
from .writer import JobWriter
from .utils import load_json


def _try_load_mocked_clean_ocr(
    mocked_dir: str | None,
    *,
    paths: JobPaths,
    page_id: str,
    page_index: int,
) -> dict[str, Any] | None:
    """Try to load a mocked *cleaned* OCR JSON for a page.

    Supported filenames (to keep the smoke fixture simple and stable):
    - <page_id>_clean.json (v0.1 internal cleaner output name)
    - <page_id>.cleaned.json
    - page_<idx:03d>.cleaned.json (0-based index)
    - page_<idx>.cleaned.json (0-based index, no padding)
    """
    if not mocked_dir:
        return None

    base = Path(mocked_dir)
    candidates = [
        base / f"{page_id}_clean.json",
        base / f"{page_id}.cleaned.json",
        base / f"page_{page_index:03d}.cleaned.json",
        base / f"page_{page_index}.cleaned.json",
    ]

    any_exists = False
    for p in candidates:
        if p.exists() and p.is_file():
            any_exists = True
            try:
                data = load_json(p)
                if isinstance(data, dict) and "tokens" in data:
                    return data
            except Exception as e:
                # warn and fall back to real OCR
                record_error(paths, page_id=page_id, stage="mocked_ocr", message=f"failed_to_parse: {p.name}: {e}")
                return None

    if not any_exists:
        # warn and fall back to real OCR
        record_error(paths, page_id=page_id, stage="mocked_ocr", message="mocked_clean_missing")
    return None


@dataclass
class RunOptions:
    input_path: str
    input_type: str
    lang: str
    source: str
    dpi: int
    min_confidence: float
    segmenter: str
    segmenter_device: str
    mocked_ocr_dir: str | None = None


class EnginePipeline:
    def __init__(self, paths: JobPaths, cfg: EngineConfig, opts: RunOptions):
        self.paths = paths
        self.cfg = cfg
        self.opts = opts

        self.page_provider = PageProvider(
            input_path=opts.input_path,
            input_type=opts.input_type,
            dpi=opts.dpi,
            paths=paths,
        )
        self.ocr = OCRExtractor(lang=opts.lang, paths=paths)
        self.cleaner = TextCleaner(paths=paths, cleanup_cfg=cfg.cleanup)
        self.layout = LayoutClassifier(paths=paths, layout_cfg=cfg.layout)
        self.segmenter = Segmenter(
            mode=opts.segmenter,
            device=opts.segmenter_device,
            paths=paths,
            segment_cfg=cfg.segment,
        )
        self.builder = FlashcardBuilder(
            source_name=opts.source,
            min_confidence=opts.min_confidence,
            confidence_cfg=cfg.confidence,
        )
        self.writer = JobWriter(paths=paths)

    def run(self, job_id: str) -> None:
        cards: list[dict[str, Any]] = []
        review_items: list[dict[str, Any]] = []

        metrics: dict[str, Any] = {
            "created_at": utc_now_iso(),
            "pages_total": 0,
            "pages_processed": 0,
            "cards_total": 0,
            "review_items_total": 0,
            "segment_success_count": 0,
            "segment_fail_count": 0,
            "ocr_empty_count": 0,
            # v0.2
            "pages_single_word": 0,
            "pages_multi_word": 0,
            "multiword_tokens": 0,
            "multiword_crops_written": 0,
            "multiword_crop_failures": 0,
            "multiword_crops_gated_small": 0,
            "multiword_crops_gated_ratio": 0,
            "deduped_tokens": 0,
            "review_items": 0,
        }

        job_meta = {
            "job_id": job_id,
            "source": self.opts.source,
            "input": {"type": self.opts.input_type, "path": self.opts.input_path},
            "created_at": metrics["created_at"],
        }

        for page, pil_img in self.page_provider.iter_pages():
            metrics["pages_total"] += 1
            try:
                # Optional: bypass OCR+cleaner using a deterministic mocked cleaned OCR JSON.
                mocked_clean = _try_load_mocked_clean_ocr(
                    self.opts.mocked_ocr_dir,
                    paths=self.paths,
                    page_id=page.page_id,
                    page_index=page.page_index,
                )
                if mocked_clean is not None:
                    raw = {"tokens": []}
                    clean = mocked_clean
                    # Ensure required keys exist.
                    clean.setdefault("page_id", page.page_id)
                    clean.setdefault("raw_count", 0)
                    clean.setdefault("clean_count", len(clean.get("tokens", []) or []))
                    clean.setdefault("deduped_count", 0)
                else:
                    # Real OCR path (fail-soft): if OCR is unavailable/fails, continue with empty tokens.
                    try:
                        raw = self.ocr.extract(page.page_id, pil_img)
                    except Exception as e:
                        record_error(self.paths, page_id=page.page_id, stage="ocr", message=str(e))
                        raw = {"tokens": []}

                    try:
                        clean = self.cleaner.clean(page.page_id, raw)
                    except Exception as e:
                        record_error(self.paths, page_id=page.page_id, stage="cleaner", message=str(e))
                        clean = {
                            "page_id": page.page_id,
                            "tokens": [],
                            "raw_count": len(raw.get("tokens", []) or []),
                            "clean_count": 0,
                            "deduped_count": 0,
                        }
                layout = self.layout.classify(page.page_id, clean, image_size=pil_img.size)

                metrics["deduped_tokens"] += int(clean.get("deduped_count", 0) or 0)

                if int(clean.get("clean_count", 0)) == 0:
                    metrics["ocr_empty_count"] += 1

                if layout.get("layout_type") == "single_word":
                    metrics["pages_single_word"] += 1
                elif layout.get("layout_type") == "multi_word":
                    metrics["pages_multi_word"] += 1

                # v0.2: for multi_word pages, create token-level crops from OCR bboxes.
                if layout.get("layout_type") == "multi_word":
                    tokens = list(clean.get("tokens", []))
                    # v0.4.1: set deterministic per-page ordering index on tokens.
                    for i, t in enumerate(tokens):
                        if isinstance(t, dict) and "token_index" not in t:
                            t["token_index"] = i
                    new_tokens, crop_stats = crop_multiword_tokens_for_page(
                        paths=self.paths,
                        page_index=page.page_index,
                        page_id=page.page_id,
                        image=pil_img,
                        tokens=tokens,
                        crop_cfg=self.cfg.crop,
                    )
                    clean["tokens"] = new_tokens
                    metrics["multiword_tokens"] += int(crop_stats.tokens_seen)
                    metrics["multiword_crops_written"] += int(crop_stats.crops_written)
                    metrics["multiword_crop_failures"] += int(crop_stats.crop_failures)
                    metrics["multiword_crops_gated_small"] += int(crop_stats.crops_gated_small)
                    metrics["multiword_crops_gated_ratio"] += int(crop_stats.crops_gated_ratio)

                segment: dict[str, Any] | None = None
                if layout.get("layout_type") == "single_word":
                    if self.opts.segmenter == "off":
                        segment = {"page_id": page.page_id, "status": "skipped", "reason": "segmenter_off"}
                    else:
                        token = None
                        tokens = clean.get("tokens", [])
                        if tokens:
                            token = max(tokens, key=lambda t: float(t.get("confidence", 0.0)))

                        if token and token.get("bbox_xyxy") and token.get("text"):
                            segment = self.segmenter.run_single_word(
                                page_id=page.page_id,
                                image=pil_img,
                                word=str(token.get("text")),
                                token_bbox=token.get("bbox_xyxy"),
                            )
                            if segment.get("status") == "success":
                                metrics["segment_success_count"] += 1
                            elif segment.get("status") == "failed":
                                metrics["segment_fail_count"] += 1
                        else:
                            # hard rule: representative word + bbox required; otherwise skip and review.
                            segment = {"page_id": page.page_id, "status": "skipped", "reason": "missing_word_or_bbox"}

                page_cards, page_reviews = self.builder.build_cards_for_page(
                    page_id=page.page_id,
                    source_ref=page.source_ref,
                    layout=layout,
                    clean=clean,
                    segment=segment,
                    page_image_path=page.image_path,
                )

                cards.extend(page_cards)
                review_items.extend(page_reviews)
                metrics["pages_processed"] += 1
            except Exception as e:
                record_error(self.paths, page_id=page.page_id, stage="page", message=str(e))
                # continue

        metrics["cards_total"] = len(cards)
        metrics["review_items_total"] = len(review_items)
        metrics["review_items"] = len(review_items)

        self.writer.write_final(job_meta=job_meta, cards=cards, review_items=review_items, metrics=metrics)
