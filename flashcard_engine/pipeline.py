from __future__ import annotations

from dataclasses import dataclass, asdict
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
from .panel_debug import PanelDebugPack, PanelDebugConfig
from .pair_extractor import PairExtractor, PairConfig, LearningCache
from .pdf_converter import pdf_to_images, iter_pdf_pages
from .sam_pair_extractor import SAMPairExtractor
from .segmenter import Segmenter
from .utils import utc_now_iso, write_json
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
    debug_panels: bool = False
    mode: str = "flashcard"  # "flashcard" or "pair"
    learning_enabled: bool = True
    use_sam: bool = False  # Use SAM-based picture detection (pair mode only)


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
        
        # Panel debug pack (optional)
        self.panel_debug: PanelDebugPack | None = None
        if opts.debug_panels:
            self.panel_debug = PanelDebugPack(paths=paths, config=PanelDebugConfig())
        
        # Pair extractor with learning cache (for pair mode)
        self.pair_extractor: PairExtractor | None = None
        self.sam_pair_extractor: SAMPairExtractor | None = None
        self.learning_cache: LearningCache | None = None
        if opts.mode == "pair":
            # Initialize learning cache from workspace root
            workspace_root = paths.job_dir.parent.parent  # workspace/jobs/<job_id> -> workspace
            if opts.learning_enabled:
                self.learning_cache = LearningCache(workspace_root)
            
            if opts.use_sam:
                # Use SAM-based extraction
                self.sam_pair_extractor = SAMPairExtractor(
                    workspace=workspace_root,
                    device=opts.segmenter_device,
                )
            else:
                # Use original grid-based extraction
                self.pair_extractor = PairExtractor(
                    paths=paths,
                    config=PairConfig(),
                    learning_cache=self.learning_cache,
                )

    def run(self, job_id: str) -> None:
        """Run pipeline in the configured mode."""
        if self.opts.mode == "pair":
            self._run_pair_mode(job_id)
        else:
            self._run_flashcard_mode(job_id)

    def _run_pair_mode(self, job_id: str) -> None:
        """Run pair extraction mode - separates picture/text crops."""
        from .pair_extractor import ItemPair, PagePairDiagnostics
        
        # Branch based on SAM vs grid-based extraction
        if self.sam_pair_extractor is not None:
            self._run_sam_pair_mode(job_id)
            return
        
        # Original grid-based pair extraction
        all_pairs: list[dict[str, Any]] = []
        all_diagnostics: list[dict[str, Any]] = []
        
        metrics: dict[str, Any] = {
            "mode": "pair",
            "created_at": utc_now_iso(),
            "pages_total": 0,
            "pages_processed": 0,
            "pairs_total": 0,
            "pairs_needing_review": 0,
            "pairs_blank_picture": 0,
            "pairs_blank_text": 0,
            "pairs_ocr_empty": 0,
            "items_detected": 0,
            "learning_enabled": self.opts.learning_enabled,
        }
        
        # Add learning cache stats if available
        if self.learning_cache:
            learning_stats = self.learning_cache.get_stats()
            metrics["learning_records_total"] = learning_stats.get("total_records", 0)
            metrics["learning_caption_corrections"] = learning_stats.get("caption_corrections", 0)
            metrics["learning_cached_parameters"] = learning_stats.get("cached_parameters", 0)
        
        job_meta = {
            "job_id": job_id,
            "mode": "pair",
            "source": self.opts.source,
            "input": {"type": self.opts.input_type, "path": self.opts.input_path},
            "created_at": metrics["created_at"],
        }
        
        assert self.pair_extractor is not None, "pair_extractor should be initialized in pair mode"
        
        for page, pil_img in self.page_provider.iter_pages():
            metrics["pages_total"] += 1
            try:
                # Optional: get OCR tokens for caption extraction
                ocr_tokens = None
                try:
                    mocked_clean = _try_load_mocked_clean_ocr(
                        self.opts.mocked_ocr_dir,
                        paths=self.paths,
                        page_id=page.page_id,
                        page_index=page.page_index,
                    )
                    if mocked_clean is not None:
                        ocr_tokens = mocked_clean.get("tokens", [])
                    else:
                        raw = self.ocr.extract(page.page_id, pil_img)
                        clean = self.cleaner.clean(page.page_id, raw)
                        ocr_tokens = clean.get("tokens", [])
                except Exception as e:
                    record_error(self.paths, page_id=page.page_id, stage="pair_ocr", message=str(e))
                    ocr_tokens = []
                
                # Extract pairs
                pairs, diagnostics = self.pair_extractor.extract_pairs_from_page(
                    page_id=page.page_id,
                    page_index=page.page_index,
                    page_image=pil_img,
                    ocr_tokens=ocr_tokens,
                )
                
                # Accumulate metrics
                metrics["items_detected"] += diagnostics.items_detected
                metrics["pairs_total"] += len(pairs)
                metrics["pairs_needing_review"] += sum(1 for p in pairs if p.needs_review)
                metrics["pairs_blank_picture"] += sum(
                    1 for p in pairs if "BLANK_PICTURE" in p.reasons
                )
                metrics["pairs_blank_text"] += sum(
                    1 for p in pairs if "BLANK_TEXT" in p.reasons
                )
                metrics["pairs_ocr_empty"] += sum(
                    1 for p in pairs if "OCR_EMPTY" in p.reasons
                )
                
                # Convert to dicts for JSON serialization
                all_pairs.extend(asdict(p) for p in pairs)
                all_diagnostics.append(asdict(diagnostics))
                
                metrics["pages_processed"] += 1
                
            except Exception as e:
                record_error(self.paths, page_id=page.page_id, stage="pair_page", message=str(e))
        
        # Write result_pairs.json
        result_pairs = {
            "schema_version": "1.0",
            "job_id": job_id,
            "mode": "pair",
            "created_at": metrics["created_at"],
            "pairs": all_pairs,
            "diagnostics": all_diagnostics,
        }
        write_json(self.paths.result_pairs_json, result_pairs)
        
        # Write review queue (for pairs needing review)
        review_items = [
            {
                "pair_id": p["pair_id"],
                "page_id": p["page_id"],
                "picture_path": p["picture_path"],
                "text_path": p["text_path"],
                "caption_text": p["caption_text"],
                "reasons": p["reasons"],
                "confidence": p["confidence"],
                "status": p["status"],
            }
            for p in all_pairs if p.get("needs_review")
        ]
        
        # Write standard outputs
        self.writer.write_final(
            job_meta=job_meta,
            cards=[],  # No cards in pair mode
            review_items=review_items,
            metrics=metrics,
        )

    def _run_sam_pair_mode(self, job_id: str) -> None:
        """Run SAM-based pair extraction mode."""
        assert self.sam_pair_extractor is not None, "sam_pair_extractor must be initialized"
        
        from .sam_pair_extractor import PageSummary
        
        all_summaries: list[dict[str, Any]] = []
        
        metrics: dict[str, Any] = {
            "mode": "pair_sam",
            "created_at": utc_now_iso(),
            "pages_total": 0,
            "pages_processed": 0,
            "pairs_total": 0,
            "pairs_needing_review": 0,
            "pictures_detected": 0,
            "text_blocks_detected": 0,
            "sam_version": "SAM",
        }
        
        job_meta = {
            "job_id": job_id,
            "mode": "pair_sam",
            "source": self.opts.source,
            "input": {"type": self.opts.input_type, "path": self.opts.input_path},
            "created_at": metrics["created_at"],
        }
        
        output_dir = self.paths.job_dir / "sam_pairs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for page, pil_img in self.page_provider.iter_pages():
            metrics["pages_total"] += 1
            try:
                summary = self.sam_pair_extractor.extract_page(
                    page_id=page.page_id,
                    page_index=page.page_index,
                    image=pil_img,
                    output_dir=output_dir,
                )
                
                # Accumulate metrics
                metrics["pairs_total"] += summary.pairs_extracted
                metrics["pairs_needing_review"] += summary.pairs_needing_review
                metrics["pictures_detected"] += summary.pictures_detected
                metrics["text_blocks_detected"] += summary.text_blocks_detected
                
                all_summaries.append(asdict(summary))
                metrics["pages_processed"] += 1
                
            except Exception as e:
                record_error(self.paths, page_id=page.page_id, stage="sam_pair_page", message=str(e))
        
        # Write result_pairs.json with SAM format
        result_pairs = {
            "schema_version": "2.0-sam",
            "job_id": job_id,
            "mode": "pair_sam",
            "created_at": metrics["created_at"],
            "pages": all_summaries,
        }
        write_json(self.paths.result_pairs_json, result_pairs)
        
        # Build review items from summaries
        review_items = []
        for page_summary in all_summaries:
            page_dir = output_dir / f"page_{page_summary['page_index'] + 1:02d}"
            for pair_data in page_summary.get("pairs", []):
                if pair_data.get("needs_review"):
                    pair_index = pair_data.get("order_index", 0) + 1
                    pair_dir = page_dir / f"pair_{pair_index:03d}"
                    review_items.append({
                        "pair_id": pair_data["pair_id"],
                        "page_id": page_summary["page_id"],
                        "picture_path": str(pair_dir / "image.png"),
                        "text_path": str(pair_dir / "text.png"),
                        "caption_text": pair_data.get("caption_text", ""),
                        "reasons": pair_data.get("reasons", []),
                        "confidence": pair_data.get("confidence", 0.0),
                        "status": "active" if pair_data.get("needs_review") else "inactive",
                    })
        
        # Write standard outputs
        self.writer.write_final(
            job_meta=job_meta,
            cards=[],  # No cards in pair mode
            review_items=review_items,
            metrics=metrics,
        )

    def _run_flashcard_mode(self, job_id: str) -> None:
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
            # Panel debug metrics
            "debug_panels_enabled": self.opts.debug_panels,
            "panels_total": 0,
            "panels_needing_review": 0,
            "panels_blank": 0,
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
                    def _canonical_token_sort_key(t: Any) -> tuple:
                        if not isinstance(t, dict):
                            return (2, 10**9, 10**9, "")
                        bbox = t.get("bbox_xyxy")
                        text = str(t.get("text") or "")
                        try:
                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                                return (0, y0, x0, y1, x1, text)
                        except Exception:
                            pass
                        return (1, 10**9, 10**9, 10**9, 10**9, text)

                    tokens = list(clean.get("tokens", []))
                    # Canonical ordering before assigning token_index (determinism).
                    try:
                        tokens.sort(key=_canonical_token_sort_key)
                    except Exception:
                        pass
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

                # Panel debug pack generation (if enabled)
                if self.panel_debug is not None:
                    try:
                        panel_diags = self.panel_debug.process_page(
                            page_id=page.page_id,
                            page_index=page.page_index,
                            page_image=pil_img,
                            tokens=clean.get("tokens", []),
                        )
                        metrics["panels_total"] += len(panel_diags)
                        metrics["panels_needing_review"] += sum(1 for d in panel_diags if d.needs_review)
                        metrics["panels_blank"] += sum(
                            1 for d in panel_diags 
                            if d.blank_score > self.panel_debug.config.blank_score_threshold
                        )
                    except Exception as e:
                        record_error(self.paths, page_id=page.page_id, stage="panel_debug", message=str(e))

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
