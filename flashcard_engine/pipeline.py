from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .builder import FlashcardBuilder
from .cleaner import TextCleaner
from .config import EngineConfig
from .job import JobPaths, record_error
from .layout import LayoutClassifier
from .ocr import OCRExtractor
from .page_provider import PageProvider
from .segmenter import Segmenter
from .utils import utc_now_iso
from .writer import JobWriter


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
                raw = self.ocr.extract(page.page_id, pil_img)
                clean = self.cleaner.clean(page.page_id, raw)
                layout = self.layout.classify(page.page_id, clean)

                if int(clean.get("clean_count", 0)) == 0:
                    metrics["ocr_empty_count"] += 1

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

        self.writer.write_final(job_meta=job_meta, cards=cards, review_items=review_items, metrics=metrics)
