"""SAM-based Pair Extractor - Dynamic picture/text pairing with no grid assumptions.

Architecture:
1. Text Detection (Tesseract OCR primary, EasyOCR fallback) → Find text regions first
2. Picture Detection (SAM) → Find picture masks, filter against text regions
3. Pairing Engine → Match pictures to nearest text blocks deterministically
4. Cropping → Output image.png, text.png, meta.json per pair

Key invariants:
- Picture-first: P pictures → P pairs (even if text missing)
- Deterministic: Same input + config = same output
- Decoupled: SAM never detects text, OCR never detects pictures
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from .utils import write_json, load_json, utc_now_iso, append_jsonl


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairConfig:
    """Configuration for SAM-based pair extraction."""
    # SAM filtering thresholds
    min_mask_area_ratio: float = 0.003  # Reject masks < 0.3% of page
    max_mask_area_ratio: float = 0.15   # Reject masks > 15% of page
    text_iou_threshold: float = 0.15    # Reject masks with IoU > 0.15 with text
    
    # Picture validation thresholds (실제 그림 판별)
    min_color_variance: float = 500.0   # 색상 분산 (적당히)
    min_edge_density: float = 0.05      # 엣지 밀도 (핵심: 실제 그림은 0.1 이상)
    min_aspect_ratio: float = 0.4       # 최소 종횡비
    max_aspect_ratio: float = 2.5       # 최대 종횡비
    background_color_threshold: int = 240  # 배경색 판별
    min_non_background_ratio: float = 0.20  # 비배경 픽셀 비율
    
    # Pairing thresholds
    max_pairing_distance_px: int = 500  # Max distance between picture and text
    search_direction: Literal["below", "above", "right", "left", "nearest"] = "below"
    margin_tolerance_px: int = 50
    
    # Text merging
    line_merge_factor: float = 1.5  # Merge lines if gap < factor * line_height
    min_text_area_px: int = 300     # Discard tiny text boxes
    min_text_length: int = 2        # Minimum text length (chars) to be valid
    min_text_confidence: float = 0.4  # Minimum OCR confidence to keep (Tesseract: 0–1)
    
    # NMS
    nms_iou_threshold: float = 0.5  # NMS threshold


@dataclass
class BBox:
    """Bounding box in XYXY format."""
    x0: int
    y0: int
    x1: int
    y1: int
    
    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)
    
    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def to_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]
    
    @classmethod
    def from_list(cls, lst: list[int]) -> "BBox":
        return cls(int(lst[0]), int(lst[1]), int(lst[2]), int(lst[3]))


@dataclass
class TextBlock:
    """Detected text region with optional OCR result."""
    bbox: BBox
    text: str = ""
    confidence: float = 0.0
    line_count: int = 1
    assigned: bool = False  # Track if already matched to a picture


@dataclass
class PictureCandidate:
    """SAM-detected picture region."""
    bbox: BBox
    mask_area: int = 0
    confidence: float = 1.0
    order_index: int = 0  # Canonical ordering index


@dataclass
class ExtractedPair:
    """A single picture-text pair."""
    pair_id: str
    order_index: int
    picture_bbox: list[int]
    text_bbox: list[int] | None
    caption_text: str
    has_text: bool
    needs_review: bool
    reasons: list[str]
    confidence: float
    picture_path: str
    text_path: str
    meta_path: str


@dataclass
class PageSummary:
    """Summary of extraction results for a page."""
    page_id: str
    page_index: int
    image_size: list[int]
    pictures_detected: int
    text_blocks_detected: int
    pairs_extracted: int
    pairs_needing_review: int
    unmatched_text_blocks: int
    created_at: str
    pairs: list[dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY UTILITIES (Deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou(box1: BBox, box2: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    # Intersection coordinates
    ix0 = max(box1.x0, box2.x0)
    iy0 = max(box1.y0, box2.y0)
    ix1 = min(box1.x1, box2.x1)
    iy1 = min(box1.y1, box2.y1)
    
    # Intersection area
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    intersection = iw * ih
    
    if intersection == 0:
        return 0.0
    
    # Union area
    union = box1.area + box2.area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_centroid_distance(box1: BBox, box2: BBox) -> float:
    """Compute Euclidean distance between box centroids."""
    c1 = box1.center
    c2 = box2.center
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def canonical_sort_key(bbox: BBox) -> tuple[int, int]:
    """Sort key for top-to-bottom, left-to-right ordering."""
    cx, cy = bbox.center
    # Quantize Y to rows (every 50px) for stable row ordering
    row = int(cy) // 50
    return (row, int(cx))


def non_max_suppression(boxes: list[PictureCandidate], iou_threshold: float) -> list[PictureCandidate]:
    """Apply NMS to picture candidates, keeping highest confidence."""
    if not boxes:
        return []
    
    # Sort by confidence descending
    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep = []
    
    for box in sorted_boxes:
        should_keep = True
        for kept in keep:
            if compute_iou(box.bbox, kept.bbox) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(box)
    
    return keep


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT DETECTION (Tesseract → EasyOCR fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class TextDetector:
    """Text detection using Tesseract OCR (primary) with EasyOCR fallback."""
    
    # Default Tesseract path on Windows (UB-Mannheim installer)
    _TESS_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    def __init__(self, lang: str = "en"):
        """Initialize text detector.
        
        Args:
            lang: Language code. Supports 'en', 'de', 'ko', etc.
                  Multiple languages can be specified as comma-separated string.
                  For Tesseract mapping: en→eng, de→deu, ko→kor, etc.
        """
        self.lang = lang
        self._ocr_engine: str | None = None  # 'tesseract' | 'easyocr' | None
        self._ocr = None  # EasyOCR reader (fallback)
        self._initialized = False
    
    # ── Language code mapping ──────────────────────────────────────────────
    _TESS_LANG_MAP: dict[str, str] = {
        "en": "eng", "de": "deu", "ko": "kor", "zh": "chi_sim",
        "ja": "jpn", "fr": "fra", "es": "spa", "it": "ita",
        "pt": "por", "ru": "rus", "ar": "ara",
    }
    
    def _tess_lang(self) -> str:
        """Convert lang codes to Tesseract format (e.g. 'de,en' → 'deu+eng')."""
        parts = [l.strip() for l in self.lang.split(",")]
        tess_parts = [self._TESS_LANG_MAP.get(p, p) for p in parts]
        return "+".join(tess_parts)
    
    def _init_ocr(self) -> bool:
        """Lazy initialization — try Tesseract first, then EasyOCR."""
        if self._initialized:
            return self._ocr_engine is not None
        self._initialized = True
        
        # ── Try Tesseract first ────────────────────────────────────────────
        try:
            import pytesseract
            import os
            if os.path.exists(self._TESS_WIN_PATH):
                pytesseract.pytesseract.tesseract_cmd = self._TESS_WIN_PATH
            # Quick sanity check
            pytesseract.get_languages()
            self._ocr_engine = "tesseract"
            print(f"[TextDetector] Using Tesseract OCR (lang={self._tess_lang()})")
            return True
        except Exception as e:
            print(f"[TextDetector] Tesseract unavailable: {e}")
        
        # ── Fallback: EasyOCR ──────────────────────────────────────────────
        try:
            import easyocr
            langs = [l.strip() for l in self.lang.split(",")]
            self._ocr = easyocr.Reader(langs, gpu=False)
            self._ocr_engine = "easyocr"
            print(f"[TextDetector] Using EasyOCR (lang={langs})")
            return True
        except Exception as e:
            print(f"[TextDetector] EasyOCR initialization failed: {e}")
        
        return False
    
    # ── Main detection entry point ────────────────────────────────────────
    def detect(self, image: Image.Image) -> list[TextBlock]:
        """Detect text regions in image, return merged text blocks."""
        if not self._init_ocr():
            return self._fallback_detect(image)
        
        if self._ocr_engine == "tesseract":
            return self._detect_tesseract(image)
        else:
            return self._detect_easyocr(image)
    
    # ── Tesseract detection ───────────────────────────────────────────────
    def _detect_tesseract(self, image: Image.Image) -> list[TextBlock]:
        """Detect text using Tesseract OCR — returns word-level blocks."""
        try:
            import pytesseract
            pil_img = image.convert("RGB")
            
            data = pytesseract.image_to_data(
                pil_img, lang=self._tess_lang(),
                output_type=pytesseract.Output.DICT,
            )
            
            raw_blocks: list[TextBlock] = []
            n = len(data["text"])
            
            for i in range(n):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if conf < 0 or not text:
                    continue
                
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                
                raw_blocks.append(TextBlock(
                    bbox=BBox(x, y, x + w, y + h),
                    text=text,
                    confidence=conf / 100.0,  # Normalize to 0-1
                ))
            
            # Pre-filter: remove noise
            filtered = [
                blk for blk in raw_blocks
                if blk.confidence >= 0.30
                and len(blk.text) >= 2
                and blk.bbox.area >= 200
            ]
            
            return self._merge_text_blocks(filtered)
            
        except Exception as e:
            print(f"[TextDetector] Tesseract detection failed: {e}")
            return self._detect_easyocr(image) if self._ocr else self._fallback_detect(image)
    
    # ── EasyOCR detection (fallback) ──────────────────────────────────────
    def _detect_easyocr(self, image: Image.Image) -> list[TextBlock]:
        """Detect text using EasyOCR — fallback engine."""
        try:
            img_array = np.array(image.convert("RGB"))
            result = self._ocr.readtext(img_array)
            
            if not result:
                return []
            
            raw_blocks: list[TextBlock] = []
            
            for item in result:
                if not item or len(item) < 3:
                    continue
                
                bbox_points = item[0]
                text = item[1]
                conf = item[2]
                
                if not bbox_points or len(bbox_points) < 4:
                    continue
                
                xs = [int(p[0]) for p in bbox_points]
                ys = [int(p[1]) for p in bbox_points]
                bbox = BBox(min(xs), min(ys), max(xs), max(ys))
                
                raw_blocks.append(TextBlock(
                    bbox=bbox, 
                    text=str(text) if text else "", 
                    confidence=float(conf) if conf else 0.0
                ))
            
            # Pre-filter garbage
            filtered = [
                blk for blk in raw_blocks
                if blk.confidence >= 0.15
                and not (len(blk.text.strip()) <= 1 and not blk.text.strip().isalpha())
                and blk.bbox.area >= 200
            ]
            
            return self._merge_text_blocks(filtered)
            
        except Exception as e:
            print(f"[TextDetector] OCR detection failed: {e}")
            return self._fallback_detect(image)
    
    def _merge_text_blocks(self, blocks: list[TextBlock], factor: float = 1.5) -> list[TextBlock]:
        """Merge vertically adjacent text lines into caption blocks."""
        if not blocks:
            return []
        
        # Sort by Y position
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox.y0)
        
        merged: list[TextBlock] = []
        current_group: list[TextBlock] = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            last = current_group[-1]
            last_height = last.bbox.height
            gap = block.bbox.y0 - last.bbox.y1
            
            # Check if same column (X overlap) and close vertically
            x_overlap = min(last.bbox.x1, block.bbox.x1) - max(last.bbox.x0, block.bbox.x0)
            same_column = x_overlap > 0.3 * min(last.bbox.width, block.bbox.width)
            
            if same_column and gap < factor * last_height and gap >= -10:
                current_group.append(block)
            else:
                merged.append(self._merge_group(current_group))
                current_group = [block]
        
        if current_group:
            merged.append(self._merge_group(current_group))
        
        return merged
    
    def _merge_group(self, group: list[TextBlock]) -> TextBlock:
        """Merge a group of text blocks into one."""
        if len(group) == 1:
            return group[0]
        
        x0 = min(b.bbox.x0 for b in group)
        y0 = min(b.bbox.y0 for b in group)
        x1 = max(b.bbox.x1 for b in group)
        y1 = max(b.bbox.y1 for b in group)
        
        merged_text = " ".join(b.text for b in group if b.text).strip()
        avg_conf = sum(b.confidence for b in group) / len(group)
        
        return TextBlock(
            bbox=BBox(x0, y0, x1, y1),
            text=merged_text,
            confidence=avg_conf,
            line_count=len(group),
        )
    
    def _fallback_detect(self, image: Image.Image) -> list[TextBlock]:
        """Fallback text detection - creates placeholder text blocks based on grid cells.
        
        This is used when PaddleOCR is not available. It creates placeholder text
        blocks in typical caption positions (bottom 30% of each grid cell).
        """
        w, h = image.size
        
        # Estimate grid based on typical vocabulary card layouts
        estimated_cols = max(1, min(4, w // 300))
        estimated_rows = max(1, min(6, h // 300))
        
        cell_w = w // estimated_cols
        cell_h = h // estimated_rows
        
        blocks = []
        for row in range(estimated_rows):
            for col in range(estimated_cols):
                # Assume text is at the bottom 30% of each cell
                x0 = col * cell_w + 10
                y0 = row * cell_h + int(cell_h * 0.7)  # Bottom 30%
                x1 = (col + 1) * cell_w - 10
                y1 = (row + 1) * cell_h - 10
                
                if x1 - x0 < 50 or y1 - y0 < 20:
                    continue
                
                bbox = BBox(x0, y0, x1, y1)
                
                # Create a placeholder text block
                # The actual text will need to be filled in via review
                blocks.append(TextBlock(
                    bbox=bbox,
                    text=f"[cell_{row}_{col}]",  # Placeholder
                    confidence=0.3,  # Low confidence for fallback
                ))
        
        return blocks


# ═══════════════════════════════════════════════════════════════════════════════
# PICTURE DETECTION (SAM-based)
# ═══════════════════════════════════════════════════════════════════════════════

class PictureDetector:
    """Picture detection using SAM or fallback methods."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._sam = None
        self._sam_type: str | None = None
        self._initialized = False
    
    def _init_sam(self) -> bool:
        """Lazy initialization of SAM model."""
        if self._initialized:
            return self._sam is not None
        
        self._initialized = True
        
        # Try Meta's official SAM first
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Look for checkpoint in common locations
            checkpoint_paths = [
                Path("models/sam_vit_b_01ec64.pth"),
                Path.home() / ".cache/sam/sam_vit_b_01ec64.pth",
                Path("sam_vit_b_01ec64.pth"),
            ]
            
            checkpoint = None
            for cp in checkpoint_paths:
                if cp.exists():
                    checkpoint = str(cp)
                    break
            
            if checkpoint:
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
                sam.to(self.device)
                self._sam = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=16,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.8,
                    min_mask_region_area=1000,
                )
                self._sam_type = "meta_sam"
                return True
        except Exception:
            pass
        
        # Try ultralytics FastSAM
        try:
            from ultralytics import FastSAM
            model = FastSAM("FastSAM-s.pt")
            self._sam = model
            self._sam_type = "fastsam"
            return True
        except Exception:
            pass
        
        # Try ultralytics SAM
        try:
            from ultralytics import SAM
            model = SAM("sam_b.pt")
            self._sam = model
            self._sam_type = "ultralytics_sam"
            return True
        except Exception:
            pass
        
        return False
    
    def detect(
        self,
        image: Image.Image,
        text_blocks: list[TextBlock],
        config: PairConfig,
    ) -> list[PictureCandidate]:
        """Detect picture regions, filtering against text blocks.
        
        실제 "그림"만 추출하기 위한 엄격한 필터링:
        1. 면적 필터 (너무 작거나 큰 것 제외)
        2. 텍스트 영역 중복 제외
        3. 종횡비 필터 (너무 길쭉한 것 제외)
        4. 색상 다양성 체크 (단색/배경 제외)
        5. 엣지 밀도 체크 (빈 영역 제외)
        """
        img_w, img_h = image.size
        page_area = img_w * img_h
        img_array = np.array(image.convert("RGB"))
        
        # Get raw masks from SAM
        raw_candidates = self._get_sam_masks(image)
        
        if not raw_candidates:
            # Fallback: use contour-based detection
            raw_candidates = self._fallback_detect(image)
        
        print(f"[PictureDetector] Raw candidates: {len(raw_candidates)}")
        
        # Apply strict filtering
        filtered: list[PictureCandidate] = []
        
        for idx, candidate in enumerate(raw_candidates):
            bbox = candidate.bbox
            area_ratio = bbox.area / page_area if page_area > 0 else 0
            
            # Filter 1: Area ratio (너무 작거나 큰 것)
            if area_ratio < config.min_mask_area_ratio:
                continue
            if area_ratio > config.max_mask_area_ratio:
                continue
            
            # Filter 2: Aspect ratio (종횡비)
            aspect = bbox.width / bbox.height if bbox.height > 0 else 0
            if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
                continue
            
            # Filter 3: Text overlap (텍스트와 겹치는 영역)
            is_text_overlap = False
            for text_block in text_blocks:
                iou = compute_iou(bbox, text_block.bbox)
                if iou > config.text_iou_threshold:
                    is_text_overlap = True
                    break
                # 텍스트 영역 내부에 완전히 포함되는 경우도 제외
                if self._is_contained(bbox, text_block.bbox, threshold=0.8):
                    is_text_overlap = True
                    break
            if is_text_overlap:
                continue
            
            # Filter 4: 실제 그림인지 검증 (색상/엣지 분석)
            if not self._is_real_picture(img_array, bbox, config):
                continue
            
            filtered.append(candidate)
        
        print(f"[PictureDetector] After filtering: {len(filtered)}")
        
        # Apply NMS to remove overlapping detections
        nms_filtered = non_max_suppression(filtered, config.nms_iou_threshold)
        
        print(f"[PictureDetector] After NMS: {len(nms_filtered)}")
        
        # Sort deterministically (top-to-bottom, left-to-right)
        sorted_candidates = sorted(nms_filtered, key=lambda c: canonical_sort_key(c.bbox))
        
        # Assign canonical order indices
        for i, candidate in enumerate(sorted_candidates):
            candidate.order_index = i
        
        return sorted_candidates
    
    def _is_contained(self, inner: BBox, outer: BBox, threshold: float = 0.8) -> bool:
        """Check if inner bbox is mostly contained within outer bbox."""
        # Calculate intersection
        ix0 = max(inner.x0, outer.x0)
        iy0 = max(inner.y0, outer.y0)
        ix1 = min(inner.x1, outer.x1)
        iy1 = min(inner.y1, outer.y1)
        
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        intersection = iw * ih
        
        if inner.area == 0:
            return False
        
        return intersection / inner.area > threshold
    
    def _is_real_picture(self, img_array: np.ndarray, bbox: BBox, config: PairConfig) -> bool:
        """실제 그림인지 검증 (단색 배경, 빈 영역 제외).
        
        검증 기준:
        1. 색상 분산 - 단색이 아닌지
        2. 비배경 픽셀 비율 - 대부분 흰색/배경이 아닌지
        3. 엣지 밀도 - 내부에 디테일이 있는지
        """
        try:
            # Crop region
            region = img_array[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
            if region.size == 0:
                return False
            
            # Check 1: 색상 분산 (Color variance)
            # 실제 그림은 다양한 색상을 포함
            gray = np.mean(region, axis=2)  # Convert to grayscale
            variance = np.var(gray)
            if variance < config.min_color_variance:
                return False
            
            # Check 2: 비배경 픽셀 비율
            # 대부분 흰색이면 그림이 아님
            bg_threshold = config.background_color_threshold
            is_background = np.all(region > bg_threshold, axis=2)  # Nearly white pixels
            non_bg_ratio = 1.0 - (np.sum(is_background) / is_background.size)
            if non_bg_ratio < config.min_non_background_ratio:
                return False
            
            # Check 3: 엣지 밀도 (Edge density)
            # 실제 그림은 내부에 선/형태가 있음
            try:
                import cv2
                gray_uint8 = gray.astype(np.uint8)
                edges = cv2.Canny(gray_uint8, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                if edge_density < config.min_edge_density:
                    return False
            except ImportError:
                # cv2 없으면 간단한 gradient 체크
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                edge_density = (np.sum(grad_x > 30) + np.sum(grad_y > 30)) / gray.size
                if edge_density < config.min_edge_density:
                    return False
            
            return True
            
        except Exception as e:
            print(f"[PictureDetector] _is_real_picture error: {e}")
            return False
    
    def _get_sam_masks(self, image: Image.Image) -> list[PictureCandidate]:
        """Get masks from SAM model."""
        if not self._init_sam():
            return []
        
        try:
            img_array = np.array(image.convert("RGB"))
            
            if self._sam_type == "meta_sam":
                masks = self._sam.generate(img_array)
                candidates = []
                for mask_data in masks:
                    seg = mask_data["segmentation"]
                    bbox_xyxy = mask_data["bbox"]  # x, y, w, h format
                    x, y, w, h = bbox_xyxy
                    bbox = BBox(int(x), int(y), int(x + w), int(y + h))
                    area = int(np.sum(seg))
                    candidates.append(PictureCandidate(
                        bbox=bbox,
                        mask_area=area,
                        confidence=float(mask_data.get("predicted_iou", 0.9)),
                    ))
                return candidates
            
            elif self._sam_type in ("fastsam", "ultralytics_sam"):
                results = self._sam(img_array, device=self.device, verbose=False)
                candidates = []
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, "masks") and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
                        
                        for i, mask in enumerate(masks):
                            if i < len(boxes):
                                x0, y0, x1, y1 = boxes[i]
                                bbox = BBox(int(x0), int(y0), int(x1), int(y1))
                            else:
                                # Compute bbox from mask
                                ys, xs = np.where(mask > 0.5)
                                if len(xs) == 0:
                                    continue
                                bbox = BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                            
                            area = int(np.sum(mask > 0.5))
                            candidates.append(PictureCandidate(
                                bbox=bbox,
                                mask_area=area,
                                confidence=0.9,
                            ))
                return candidates
            
        except Exception:
            pass
        
        return []
    
    def _fallback_detect(self, image: Image.Image) -> list[PictureCandidate]:
        """Fallback detection using contour analysis or grid-based approach."""
        try:
            import cv2
            
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect components
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter small contours
                if area < 1000:
                    continue
                
                bbox = BBox(x, y, x + w, y + h)
                candidates.append(PictureCandidate(
                    bbox=bbox,
                    mask_area=int(area),
                    confidence=0.7,
                ))
            
            return candidates
            
        except ImportError:
            # Fallback to PIL-based simple grid detection
            return self._fallback_grid_detect(image)
        except Exception:
            return self._fallback_grid_detect(image)
    
    def _fallback_grid_detect(self, image: Image.Image) -> list[PictureCandidate]:
        """Simple grid-based picture detection using PIL only."""
        w, h = image.size
        
        # Assume vocabulary pages have rectangular item blocks
        # Typical layouts: 2x3, 2x4, 3x3 grids
        
        # Analyze image to estimate grid
        gray = np.array(image.convert("L"), dtype=np.float32)
        
        # Find potential horizontal/vertical lines by checking edge density
        # Simple approach: divide image into potential cells
        
        # Estimate based on typical vocabulary card sizes (150-400px per cell)
        estimated_cols = max(1, min(4, w // 300))
        estimated_rows = max(1, min(6, h // 300))
        
        cell_w = w // estimated_cols
        cell_h = h // estimated_rows
        
        candidates = []
        for row in range(estimated_rows):
            for col in range(estimated_cols):
                x0 = col * cell_w + 10
                y0 = row * cell_h + 10
                x1 = min((col + 1) * cell_w - 10, w - 10)
                y1 = min((row + 1) * cell_h - 10, h - 10)
                
                if x1 - x0 < 100 or y1 - y0 < 100:
                    continue
                
                # Check if this region has content (not blank)
                region = gray[y0:y1, x0:x1]
                std_dev = np.std(region)
                
                # Skip mostly blank regions (low variance)
                if std_dev < 20:
                    continue
                
                bbox = BBox(x0, y0, x1, y1)
                area = (x1 - x0) * (y1 - y0)
                
                candidates.append(PictureCandidate(
                    bbox=bbox,
                    mask_area=area,
                    confidence=0.5,  # Lower confidence for grid-based detection
                ))
        
        return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# PAIRING ENGINE (Deterministic matching)
# ═══════════════════════════════════════════════════════════════════════════════

class PairingEngine:
    """Match pictures to text blocks using deterministic heuristics."""
    
    def __init__(self, config: PairConfig):
        self.config = config
    
    def compute_pairing_score(
        self,
        picture: PictureCandidate,
        text: TextBlock,
    ) -> float:
        """Compute matching score between picture and text block.
        
        Lower score = better match.
        Returns float("inf") if not a valid match.
        """
        pic_bbox = picture.bbox
        txt_bbox = text.bbox
        
        # Base: centroid distance
        distance = compute_centroid_distance(pic_bbox, txt_bbox)
        
        if distance > self.config.max_pairing_distance_px:
            return float("inf")
        
        # Direction preference penalty
        direction_penalty = 0.0
        pic_cx, pic_cy = pic_bbox.center
        txt_cx, txt_cy = txt_bbox.center
        
        if self.config.search_direction == "below":
            # Prefer text below picture
            if txt_cy < pic_cy:  # Text is above picture
                direction_penalty = 200.0
            # Bonus if text is directly below (aligned X)
            x_offset = abs(txt_cx - pic_cx)
            if x_offset < pic_bbox.width * 0.5:
                direction_penalty -= 50.0
                
        elif self.config.search_direction == "above":
            if txt_cy > pic_cy:  # Text is below picture
                direction_penalty = 200.0
                
        elif self.config.search_direction == "right":
            if txt_cx < pic_cx:  # Text is left of picture
                direction_penalty = 200.0
                
        elif self.config.search_direction == "left":
            if txt_cx > pic_cx:  # Text is right of picture
                direction_penalty = 200.0
        
        # Nearest: no direction penalty
        
        # Column alignment bonus
        x_overlap = min(pic_bbox.x1, txt_bbox.x1) - max(pic_bbox.x0, txt_bbox.x0)
        if x_overlap > 0:
            alignment_bonus = -30.0 * (x_overlap / min(pic_bbox.width, txt_bbox.width))
        else:
            alignment_bonus = 0.0
        
        score = distance + direction_penalty + alignment_bonus
        return max(0.0, score)
    
    def match_pairs(
        self,
        pictures: list[PictureCandidate],
        text_blocks: list[TextBlock],
    ) -> list[tuple[PictureCandidate, TextBlock | None, list[str]]]:
        """Match each picture to at most one text block.
        
        Returns: List of (picture, matched_text_or_none, reasons)
        """
        # Reset assignment flags
        for tb in text_blocks:
            tb.assigned = False
        
        results: list[tuple[PictureCandidate, TextBlock | None, list[str]]] = []
        
        # Process pictures in canonical order
        for picture in sorted(pictures, key=lambda p: p.order_index):
            best_text: TextBlock | None = None
            best_score = float("inf")
            
            # Find best unassigned text block
            for text in text_blocks:
                if text.assigned:
                    continue
                
                score = self.compute_pairing_score(picture, text)
                if score < best_score:
                    best_score = score
                    best_text = text
            
            reasons: list[str] = []
            
            if best_text is None or best_score == float("inf"):
                reasons.append("NO_TEXT_MATCH")
                results.append((picture, None, reasons))
            else:
                best_text.assigned = True
                if not best_text.text.strip():
                    reasons.append("OCR_EMPTY")
                results.append((picture, best_text, reasons))
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING CONFIG (Rule-based self-correction)
# ═══════════════════════════════════════════════════════════════════════════════

class LearningManager:
    """Manage learning rules and feedback processing."""
    
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.learning_dir = self.workspace / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        self.rules_config_path = self.learning_dir / "rules_config.json"
        self.records_dir = self.learning_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        
        self._rules: dict[str, Any] = {}
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules configuration."""
        if self.rules_config_path.exists():
            try:
                self._rules = load_json(self.rules_config_path)
            except Exception:
                self._rules = {}
    
    def _save_rules(self) -> None:
        """Save rules configuration."""
        write_json(self.rules_config_path, self._rules)
    
    def apply_to_config(self, config: PairConfig) -> PairConfig:
        """Apply learned rules to config."""
        if "search_direction" in self._rules:
            direction = self._rules["search_direction"]
            if direction in ("below", "above", "right", "left", "nearest"):
                config.search_direction = direction
        
        if "max_pairing_distance_px" in self._rules:
            config.max_pairing_distance_px = int(self._rules["max_pairing_distance_px"])
        
        if "margin_tolerance_px" in self._rules:
            config.margin_tolerance_px = int(self._rules["margin_tolerance_px"])
        
        if "text_iou_threshold" in self._rules:
            config.text_iou_threshold = float(self._rules["text_iou_threshold"])
        
        return config
    
    def get_caption_correction(self, pair_id: str) -> str | None:
        """Get learned caption correction for a pair."""
        corrections = self._rules.get("caption_corrections", {})
        return corrections.get(pair_id)
    
    def process_feedback(self, feedback_path: Path, job_id: str) -> dict[str, int]:
        """Process feedback and update rules."""
        if not feedback_path.exists():
            return {"processed": 0}
        
        try:
            feedback_data = load_json(feedback_path)
        except Exception:
            return {"error": "invalid_feedback_json"}
        
        # Handle both formats: {"items": [...]} or [...]
        if isinstance(feedback_data, dict):
            items = feedback_data.get("items", [])
        else:
            items = feedback_data if isinstance(feedback_data, list) else []
        
        stats = {"processed": 0, "direction_changes": 0, "caption_corrections": 0}
        
        caption_corrections = self._rules.get("caption_corrections", {})
        direction_votes: dict[str, int] = self._rules.get("direction_votes", {})
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            stats["processed"] += 1
            
            # Handle caption corrections
            if "corrected_caption" in item or "edited_caption" in item:
                pair_id = item.get("pair_id", "")
                new_caption = item.get("corrected_caption") or item.get("edited_caption", "")
                if pair_id and new_caption:
                    caption_corrections[pair_id] = new_caption
                    stats["caption_corrections"] += 1
            
            # Handle direction preference
            if "preferred_search_direction" in item:
                direction = item["preferred_search_direction"]
                if direction in ("below", "above", "right", "left", "nearest"):
                    direction_votes[direction] = direction_votes.get(direction, 0) + 1
                    stats["direction_changes"] += 1
        
        # Update rules
        self._rules["caption_corrections"] = caption_corrections
        self._rules["direction_votes"] = direction_votes
        
        # Set search_direction to most voted
        if direction_votes:
            best_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
            self._rules["search_direction"] = best_direction
        
        self._rules["last_feedback_applied"] = utc_now_iso()
        self._save_rules()
        
        # Record feedback processing
        timestamp = utc_now_iso().replace(":", "-")
        record_file = self.records_dir / f"{timestamp}__{job_id[:8]}.jsonl"
        for item in items:
            append_jsonl(record_file, {
                "job_id": job_id,
                "timestamp": utc_now_iso(),
                "feedback_item": item,
            })
        
        return stats
    
    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        records_count = sum(1 for _ in self.records_dir.glob("*.jsonl"))
        return {
            "rules_configured": len(self._rules),
            "search_direction": self._rules.get("search_direction", "below"),
            "caption_corrections": len(self._rules.get("caption_corrections", {})),
            "feedback_records": records_count,
            "last_feedback_applied": self._rules.get("last_feedback_applied"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SAMPairExtractor:
    """Complete SAM-based pair extraction pipeline."""
    
    def __init__(
        self,
        workspace: str | Path,
        config: PairConfig | None = None,
        lang: str = "en",
        device: str = "cpu",
    ):
        self.workspace = Path(workspace)
        self.config = config or PairConfig()
        
        # Initialize components
        self.text_detector = TextDetector(lang=lang)
        self.picture_detector = PictureDetector(device=device)
        self.learning_manager = LearningManager(workspace)
        
        # Apply learned rules to config
        self.config = self.learning_manager.apply_to_config(self.config)
        
        self.pairing_engine = PairingEngine(self.config)
    
    def _generate_pair_id(self, page_id: str, order_index: int, bbox: list[int]) -> str:
        """Generate stable pair ID."""
        payload = f"{page_id}|{order_index}|{','.join(map(str, bbox))}"
        return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]
    
    def extract_page(
        self,
        page_id: str,
        page_index: int,
        image: Image.Image,
        output_dir: Path,
    ) -> PageSummary:
        """Extract pairs from a single page image.
        
        Args:
            page_id: Unique page identifier
            page_index: 0-based page index
            image: PIL Image of the page
            output_dir: Directory to write outputs
            
        Returns:
            PageSummary with extraction results
        """
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        img_w, img_h = image.size
        page_dir = output_dir / f"page_{page_index + 1:02d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Text Detection (MUST run first for anti-noise filtering)
        text_blocks = self.text_detector.detect(image)
        
        # Filter: area, length, confidence
        quality_text_blocks: list[TextBlock] = []
        for tb in text_blocks:
            if tb.bbox.area < self.config.min_text_area_px:
                continue
            txt = tb.text.strip()
            if len(txt) < self.config.min_text_length:
                continue
            if tb.confidence < self.config.min_text_confidence:
                continue
            quality_text_blocks.append(tb)
        
        text_blocks = quality_text_blocks
        
        print(f"[ExtractPage] {page_id}: {len(text_blocks)} quality text blocks")
        
        # Step 2: Picture Detection (filtered against text regions)
        pictures = self.picture_detector.detect(image, text_blocks, self.config)
        
        # Step 3: Pairing
        matches = self.pairing_engine.match_pairs(pictures, text_blocks)
        
        # Step 4: Generate outputs
        pairs: list[ExtractedPair] = []
        
        for i, (picture, text, reasons) in enumerate(matches):
            pair_id = self._generate_pair_id(page_id, i, picture.bbox.to_list())
            pair_dir = page_dir / f"pair_{i + 1:03d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            # Check for learned caption correction
            corrected_caption = self.learning_manager.get_caption_correction(pair_id)
            
            # Crop and save picture
            pic_crop = image.crop((
                picture.bbox.x0, picture.bbox.y0,
                picture.bbox.x1, picture.bbox.y1
            ))
            pic_path = pair_dir / "image.png"
            pic_crop.save(pic_path, format="PNG")
            
            # Crop and save text (or placeholder)
            if text:
                txt_crop = image.crop((
                    text.bbox.x0, text.bbox.y0,
                    text.bbox.x1, text.bbox.y1
                ))
                caption_text = corrected_caption or text.text
            else:
                # Create placeholder
                txt_crop = Image.new("RGB", (100, 30), color=(255, 255, 255))
                draw = ImageDraw.Draw(txt_crop)
                draw.text((5, 5), "[no text]", fill=(200, 200, 200))
                caption_text = corrected_caption or ""
            
            txt_path = pair_dir / "text.png"
            txt_crop.save(txt_path, format="PNG")
            
            # Determine review status
            has_text = text is not None and bool(caption_text.strip())
            needs_review = not has_text or len(reasons) > 0
            
            if corrected_caption:
                reasons = [r for r in reasons if r != "OCR_EMPTY"]
                has_text = True
                needs_review = False
            
            # Compute confidence
            confidence = picture.confidence
            if not has_text:
                confidence *= 0.5
            if reasons:
                confidence *= 0.8
            
            # Create meta.json
            meta = {
                "pair_id": pair_id,
                "order_index": i,
                "picture_bbox": picture.bbox.to_list(),
                "text_bbox": text.bbox.to_list() if text else None,
                "caption_text": caption_text,
                "has_text": has_text,
                "needs_review": needs_review,
                "reasons": reasons,
                "confidence": round(confidence, 4),
            }
            
            meta_path = pair_dir / "meta.json"
            write_json(meta_path, meta)
            
            # Create ExtractedPair
            pairs.append(ExtractedPair(
                pair_id=pair_id,
                order_index=i,
                picture_bbox=picture.bbox.to_list(),
                text_bbox=text.bbox.to_list() if text else None,
                caption_text=caption_text,
                has_text=has_text,
                needs_review=needs_review,
                reasons=reasons,
                confidence=round(confidence, 4),
                picture_path=str(pic_path.relative_to(output_dir)),
                text_path=str(txt_path.relative_to(output_dir)),
                meta_path=str(meta_path.relative_to(output_dir)),
            ))
        
        # Count unmatched text blocks
        unmatched_text = sum(1 for tb in text_blocks if not tb.assigned)
        
        # Create page summary
        summary = PageSummary(
            page_id=page_id,
            page_index=page_index,
            image_size=[img_w, img_h],
            pictures_detected=len(pictures),
            text_blocks_detected=len(text_blocks),
            pairs_extracted=len(pairs),
            pairs_needing_review=sum(1 for p in pairs if p.needs_review),
            unmatched_text_blocks=unmatched_text,
            created_at=utc_now_iso(),
            pairs=[asdict(p) for p in pairs],
        )
        
        # Write summary.json
        summary_path = page_dir / "summary.json"
        write_json(summary_path, asdict(summary))
        
        return summary
    
    def extract_job(
        self,
        job_id: str,
        images: list[tuple[str, Image.Image]],  # List of (page_id, image)
    ) -> dict[str, Any]:
        """Extract pairs from multiple pages.
        
        Args:
            job_id: Unique job identifier
            images: List of (page_id, PIL Image) tuples
            
        Returns:
            Job summary dict
        """
        output_dir = self.workspace / "output" / f"job_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_summaries: list[PageSummary] = []
        
        for page_index, (page_id, image) in enumerate(images):
            summary = self.extract_page(
                page_id=page_id,
                page_index=page_index,
                image=image,
                output_dir=output_dir,
            )
            all_summaries.append(summary)
        
        # Aggregate stats
        job_summary = {
            "job_id": job_id,
            "created_at": utc_now_iso(),
            "pages_processed": len(all_summaries),
            "total_pairs": sum(s.pairs_extracted for s in all_summaries),
            "total_needing_review": sum(s.pairs_needing_review for s in all_summaries),
            "total_pictures_detected": sum(s.pictures_detected for s in all_summaries),
            "total_text_blocks_detected": sum(s.text_blocks_detected for s in all_summaries),
            "total_unmatched_text": sum(s.unmatched_text_blocks for s in all_summaries),
            "pages": [asdict(s) for s in all_summaries],
            "config": asdict(self.config),
            "learning_stats": self.learning_manager.get_stats(),
        }
        
        # Write job summary
        job_summary_path = output_dir / "job_summary.json"
        write_json(job_summary_path, job_summary)
        
        return job_summary
    
    def apply_feedback(self, feedback_path: str | Path, job_id: str) -> dict[str, int]:
        """Apply feedback and update learning rules."""
        return self.learning_manager.process_feedback(Path(feedback_path), job_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pairs_from_image(
    image_path: str | Path,
    workspace: str | Path = "./workspace",
    job_id: str | None = None,
    lang: str = "en",
    device: str = "cpu",
    config: PairConfig | None = None,
) -> dict[str, Any]:
    """Convenience function to extract pairs from a single image."""
    image_path = Path(image_path)
    workspace = Path(workspace)
    
    if job_id is None:
        job_id = hashlib.sha1(
            str(image_path).encode(), usedforsecurity=False
        ).hexdigest()[:12]
    
    image = Image.open(image_path).convert("RGB")
    page_id = image_path.stem
    
    extractor = SAMPairExtractor(
        workspace=workspace,
        lang=lang,
        device=device,
        config=config,
    )
    
    return extractor.extract_job(job_id, [(page_id, image)])


def extract_pairs_from_folder(
    folder_path: str | Path,
    workspace: str | Path = "./workspace",
    job_id: str | None = None,
    lang: str = "en",
    device: str = "cpu",
    config: PairConfig | None = None,
) -> dict[str, Any]:
    """Convenience function to extract pairs from a folder of images."""
    folder_path = Path(folder_path)
    workspace = Path(workspace)
    
    if job_id is None:
        job_id = hashlib.sha1(
            str(folder_path).encode(), usedforsecurity=False
        ).hexdigest()[:12]
    
    # Collect images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    images: list[tuple[str, Image.Image]] = []
    for img_file in image_files:
        try:
            image = Image.open(img_file).convert("RGB")
            images.append((img_file.stem, image))
        except Exception:
            continue
    
    extractor = SAMPairExtractor(
        workspace=workspace,
        lang=lang,
        device=device,
        config=config,
    )
    
    return extractor.extract_job(job_id, images)
