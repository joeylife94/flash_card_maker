"""Pair Mode Extractor - Extract picture/text crop pairs from workbook pages.

This module detects item blocks (vocabulary cells) in workbook pages and
separates them into:
1. picture-only crops
2. text-only crops

Each pair is linked via explicit mapping in result_pairs.json.

The module supports a learning loop where human corrections improve future runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .job import JobPaths, record_error
from .panel_debug import compute_blank_score, BlankScoreResult
from .utils import clamp_bbox_xyxy, utc_now_iso, write_json, load_json, append_jsonl


# Pair extraction reason codes
PAIR_REASON_BLANK_PICTURE = "BLANK_PICTURE"
PAIR_REASON_BLANK_TEXT = "BLANK_TEXT"
PAIR_REASON_NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
PAIR_REASON_SMALL_ITEM = "SMALL_ITEM"
PAIR_REASON_UNCERTAIN_SPLIT = "UNCERTAIN_SPLIT"
PAIR_REASON_OCR_EMPTY = "OCR_EMPTY"
PAIR_REASON_LEARNED_CORRECTION = "LEARNED_CORRECTION"


@dataclass
class PairConfig:
    """Configuration for pair extraction."""
    # Item detection
    min_item_area_px: int = 5000
    min_item_width: int = 50
    min_item_height: int = 50
    
    # Picture/text split heuristics
    default_text_ratio: float = 0.3  # Bottom 30% is text by default
    text_region_position: str = "bottom"  # bottom, top, right, left
    
    # Quality thresholds
    blank_score_threshold: float = 0.7
    edge_density_min: float = 0.01
    
    # Grid detection
    grid_detection_enabled: bool = True
    grid_line_threshold: int = 8
    
    # Learning
    learning_enabled: bool = True


@dataclass
class ItemPair:
    """A single picture-text pair."""
    pair_id: str
    page_id: str
    item_index: int
    picture_path: str | None
    text_path: str | None
    caption_text: str
    bbox_item_xyxy: list[int]
    bbox_picture_xyxy: list[int] | None
    bbox_text_xyxy: list[int] | None
    status: str  # active, review, rejected
    needs_review: bool
    reasons: list[str]
    blank_score_picture: float
    blank_score_text: float
    confidence: float


@dataclass
class PagePairDiagnostics:
    """Diagnostics for pair extraction on a single page."""
    page_id: str
    page_index: int
    image_size: list[int]
    items_detected: int
    pairs_extracted: int
    pairs_needing_review: int
    grid_detected: bool
    grid_rows: int
    grid_cols: int
    text_ratio_used: float
    text_position_used: str
    learned_adjustments: list[str]
    created_at: str


def _generate_pair_id(page_id: str, item_index: int, bbox: list[int]) -> str:
    """Generate stable pair ID."""
    bbox_str = ",".join(str(int(x)) for x in bbox)
    payload = f"{page_id}|{item_index}|{bbox_str}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _compute_page_hash(image: Image.Image) -> str:
    """Compute a hash of the page image for learning cache lookup."""
    # Resize to small thumbnail for fast hashing
    thumb = image.copy()
    thumb.thumbnail((64, 64))
    arr = np.array(thumb.convert("L"))
    return hashlib.md5(arr.tobytes(), usedforsecurity=False).hexdigest()


def _detect_grid_lines(image: Image.Image, threshold: int = 8) -> tuple[list[int], list[int]]:
    """Detect horizontal and vertical grid lines using edge detection.
    
    Uses adaptive thresholding: tries strict threshold first, then relaxes
    if too few lines are found.
    
    Returns:
        (horizontal_lines, vertical_lines) - lists of y and x coordinates
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    
    def _find_lines_at_threshold(thresh: int) -> tuple[list[int], list[int]]:
        # Detect horizontal lines (rows with strong horizontal edges)
        horizontal_lines = []
        for y in range(1, h - 1):
            row_diff = np.abs(gray[y, :] - gray[y - 1, :])
            if np.mean(row_diff) > thresh:
                horizontal_lines.append(y)
        
        # Detect vertical lines
        vertical_lines = []
        for x in range(1, w - 1):
            col_diff = np.abs(gray[:, x] - gray[:, x - 1])
            if np.mean(col_diff) > thresh:
                vertical_lines.append(x)
        
        return horizontal_lines, vertical_lines
    
    def cluster_lines(lines: list[int], min_gap: int = 30) -> list[int]:
        if not lines:
            return []
        lines = sorted(lines)
        clusters = [[lines[0]]]
        for line in lines[1:]:
            if line - clusters[-1][-1] < min_gap:
                clusters[-1].append(line)
            else:
                clusters.append([line])
        return [int(np.mean(c)) for c in clusters]
    
    # Try multiple thresholds: strict → relaxed
    for thresh in [threshold, max(5, threshold - 3), max(3, threshold // 2)]:
        h_lines, v_lines = _find_lines_at_threshold(thresh)
        
        clustered_h = cluster_lines(h_lines)
        clustered_v = cluster_lines(v_lines)
        
        # If we found at least 2 horizontal + 2 vertical lines, we have a grid
        if len(clustered_h) >= 2 and len(clustered_v) >= 2:
            return clustered_h, clustered_v
    
    # Final fallback: use projection-based detection
    return _detect_grid_lines_projection(gray)


def _detect_grid_lines_projection(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Detect grid lines using projection profiles (more robust for faint lines).
    
    Looks at the variance along each row/column — grid lines cause
    sudden variance drops (uniform color across the line).
    """
    h, w = gray.shape
    
    # Horizontal line detection: compute row-wise standard deviation
    row_std = np.array([np.std(gray[y, :]) for y in range(h)])
    
    # Grid lines have LOW std (they're uniform across the row)
    # Find local minima in std that are below median
    median_std = np.median(row_std)
    h_candidates = []
    for y in range(5, h - 5):
        if row_std[y] < median_std * 0.5:
            # Check it's a local minimum
            if row_std[y] <= row_std[y - 2] and row_std[y] <= row_std[y + 2]:
                h_candidates.append(y)
    
    # Vertical line detection
    col_std = np.array([np.std(gray[:, x]) for x in range(w)])
    median_col_std = np.median(col_std)
    v_candidates = []
    for x in range(5, w - 5):
        if col_std[x] < median_col_std * 0.5:
            if col_std[x] <= col_std[x - 2] and col_std[x] <= col_std[x + 2]:
                v_candidates.append(x)
    
    def cluster_lines(lines: list[int], min_gap: int = 30) -> list[int]:
        if not lines:
            return []
        lines = sorted(lines)
        clusters = [[lines[0]]]
        for line in lines[1:]:
            if line - clusters[-1][-1] < min_gap:
                clusters[-1].append(line)
            else:
                clusters.append([line])
        return [int(np.mean(c)) for c in clusters]
    
    return cluster_lines(h_candidates), cluster_lines(v_candidates)


def _detect_item_blocks_grid(
    image: Image.Image,
    config: PairConfig,
) -> list[tuple[int, int, int, int]]:
    """Detect item blocks using grid line detection."""
    w, h = image.size
    
    h_lines, v_lines = _detect_grid_lines(image, config.grid_line_threshold)
    
    # Add image boundaries
    if not h_lines or h_lines[0] > 50:
        h_lines = [0] + h_lines
    if not h_lines or h_lines[-1] < h - 50:
        h_lines = h_lines + [h]
    
    if not v_lines or v_lines[0] > 50:
        v_lines = [0] + v_lines
    if not v_lines or v_lines[-1] < w - 50:
        v_lines = v_lines + [w]
    
    # Generate grid cells
    blocks = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y0, y1 = h_lines[i], h_lines[i + 1]
            x0, x1 = v_lines[j], v_lines[j + 1]
            
            cell_w = x1 - x0
            cell_h = y1 - y0
            area = cell_w * cell_h
            
            if (cell_w >= config.min_item_width and 
                cell_h >= config.min_item_height and
                area >= config.min_item_area_px):
                blocks.append((x0, y0, x1, y1))
    
    return blocks


def _detect_item_blocks_contour(
    image: Image.Image,
    config: PairConfig,
) -> list[tuple[int, int, int, int]]:
    """Detect item blocks using contour/connected component analysis.

    Improved: uses OpenCV contour detection first, then adaptive edge
    projection, then common grid layouts as last resort.
    """
    w, h = image.size
    gray = np.array(image.convert("L"), dtype=np.float32)

    # --- Strategy 1: Try OpenCV contour detection ---
    try:
        import cv2

        gray_u8 = gray.astype(np.uint8)

        # Adaptive threshold to find cell boundaries
        binary = cv2.adaptiveThreshold(
            gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Find horizontal and vertical lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 10), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 10)))

        h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Combine lines
        grid_mask = cv2.add(h_lines_img, v_lines_img)

        # Find contours of cells (inverse of grid lines)
        grid_inv = cv2.bitwise_not(grid_mask)
        contours, _ = cv2.findContours(grid_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks: list[tuple[int, int, int, int]] = []
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            if (cw >= config.min_item_width
                    and ch >= config.min_item_height
                    and area >= config.min_item_area_px
                    and area < w * h * 0.5):  # Not the whole page
                blocks.append((cx, cy, cx + cw, cy + ch))

        if len(blocks) >= 2:
            return sorted(blocks, key=lambda b: (b[1], b[0]))

    except ImportError:
        pass

    # --- Strategy 2: Adaptive edge-projection grid estimation ---
    edges_h = np.abs(gray[1:, :] - gray[:-1, :])
    edges_v = np.abs(gray[:, 1:] - gray[:, :-1])

    h_profile = np.mean(edges_h, axis=1)
    v_profile = np.mean(edges_v, axis=0)

    h_threshold = np.percentile(h_profile, 90)
    v_threshold = np.percentile(v_profile, 90)

    h_peaks = [i for i in range(len(h_profile)) if h_profile[i] > h_threshold]
    v_peaks = [i for i in range(len(v_profile)) if v_profile[i] > v_threshold]

    def cluster(positions: list[int], min_gap: int = 50) -> list[int]:
        if not positions:
            return []
        positions = sorted(positions)
        groups: list[list[int]] = [[positions[0]]]
        for p in positions[1:]:
            if p - groups[-1][-1] < min_gap:
                groups[-1].append(p)
            else:
                groups.append([p])
        return [int(np.mean(g)) for g in groups]

    h_lines_clustered = cluster(h_peaks)
    v_lines_clustered = cluster(v_peaks)

    # Add boundaries
    if not h_lines_clustered or h_lines_clustered[0] > 50:
        h_lines_clustered = [0] + h_lines_clustered
    if not h_lines_clustered or h_lines_clustered[-1] < h - 50:
        h_lines_clustered.append(h)
    if not v_lines_clustered or v_lines_clustered[0] > 50:
        v_lines_clustered = [0] + v_lines_clustered
    if not v_lines_clustered or v_lines_clustered[-1] < w - 50:
        v_lines_clustered.append(w)

    blocks = []
    for i in range(len(h_lines_clustered) - 1):
        for j in range(len(v_lines_clustered) - 1):
            y0 = h_lines_clustered[i]
            y1 = h_lines_clustered[i + 1]
            x0 = v_lines_clustered[j]
            x1 = v_lines_clustered[j + 1]
            cell_w = x1 - x0
            cell_h = y1 - y0
            area = cell_w * cell_h
            if (cell_w >= config.min_item_width
                    and cell_h >= config.min_item_height
                    and area >= config.min_item_area_px):
                blocks.append((x0, y0, x1, y1))

    if len(blocks) >= 2:
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    # --- Strategy 3: Try common vocabulary page grid layouts ---
    best_blocks: list[tuple[int, int, int, int]] = []
    for cols, rows in [(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (2, 5), (2, 6)]:
        cell_w = w // cols
        cell_h = h // rows
        if cell_w < config.min_item_width or cell_h < config.min_item_height:
            continue
        trial_blocks: list[tuple[int, int, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                x0 = c * cell_w
                y0 = r * cell_h
                x1 = min((c + 1) * cell_w, w)
                y1 = min((r + 1) * cell_h, h)
                trial_blocks.append((x0, y0, x1, y1))
        if len(trial_blocks) > len(best_blocks):
            best_blocks = trial_blocks

    if best_blocks:
        return sorted(best_blocks, key=lambda b: (b[1], b[0]))

    return [(0, 0, w, h)]


def _split_item_into_picture_text(
    image: Image.Image,
    item_bbox: tuple[int, int, int, int],
    text_ratio: float,
    text_position: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Split an item block into picture and text regions.
    
    Args:
        image: Full page image
        item_bbox: Item bounding box (x0, y0, x1, y1)
        text_ratio: Fraction of item that is text (0.0-1.0)
        text_position: Where text is located (bottom, top, right, left)
    
    Returns:
        (picture_bbox, text_bbox)
    """
    x0, y0, x1, y1 = item_bbox
    w = x1 - x0
    h = y1 - y0
    
    if text_position == "bottom":
        # Text at bottom
        split_y = y0 + int(h * (1 - text_ratio))
        picture_bbox = (x0, y0, x1, split_y)
        text_bbox = (x0, split_y, x1, y1)
    
    elif text_position == "top":
        # Text at top
        split_y = y0 + int(h * text_ratio)
        text_bbox = (x0, y0, x1, split_y)
        picture_bbox = (x0, split_y, x1, y1)
    
    elif text_position == "right":
        # Text on right
        split_x = x0 + int(w * (1 - text_ratio))
        picture_bbox = (x0, y0, split_x, y1)
        text_bbox = (split_x, y0, x1, y1)
    
    elif text_position == "left":
        # Text on left
        split_x = x0 + int(w * text_ratio)
        text_bbox = (x0, y0, split_x, y1)
        picture_bbox = (split_x, y0, x1, y1)
    
    else:
        # Default: bottom
        split_y = y0 + int(h * (1 - text_ratio))
        picture_bbox = (x0, y0, x1, split_y)
        text_bbox = (x0, split_y, x1, y1)
    
    return picture_bbox, text_bbox


class LearningCache:
    """Cache for learned corrections and adaptive parameters."""
    
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.learning_dir = self.workspace / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        self.records_dir = self.learning_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.learning_dir / "adaptive_cache.json"
        self.corrections_file = self.learning_dir / "caption_corrections.json"
        
        self._cache: dict[str, Any] = {}
        self._corrections: dict[str, str] = {}
        self._load()
    
    def _load(self) -> None:
        """Load cached data."""
        if self.cache_file.exists():
            try:
                self._cache = load_json(self.cache_file)
            except Exception:
                self._cache = {}
        
        if self.corrections_file.exists():
            try:
                self._corrections = load_json(self.corrections_file)
            except Exception:
                self._corrections = {}
    
    def _save(self) -> None:
        """Save cached data."""
        write_json(self.cache_file, self._cache)
        write_json(self.corrections_file, self._corrections)
    
    def get_text_ratio(self, page_hash: str, default: float = 0.3) -> float:
        """Get learned text ratio for a page."""
        key = f"text_ratio_{page_hash}"
        return float(self._cache.get(key, default))
    
    def set_text_ratio(self, page_hash: str, ratio: float) -> None:
        """Set learned text ratio."""
        key = f"text_ratio_{page_hash}"
        self._cache[key] = ratio
        self._save()
    
    def get_text_position(self, page_hash: str, default: str = "bottom") -> str:
        """Get learned text position for a page."""
        key = f"text_position_{page_hash}"
        return str(self._cache.get(key, default))
    
    def set_text_position(self, page_hash: str, position: str) -> None:
        """Set learned text position."""
        key = f"text_position_{page_hash}"
        self._cache[key] = position
        self._save()
    
    def get_caption_correction(self, pair_signature: str) -> str | None:
        """Get cached caption correction for a pair signature."""
        return self._corrections.get(pair_signature)
    
    def set_caption_correction(self, pair_signature: str, corrected_text: str) -> None:
        """Cache a caption correction."""
        self._corrections[pair_signature] = corrected_text
        self._save()
    
    def get_blank_threshold(self, default: float = 0.7) -> float:
        """Get adaptive blank score threshold."""
        return float(self._cache.get("blank_threshold", default))
    
    def adjust_blank_threshold(self, delta: float) -> None:
        """Adjust blank threshold based on feedback."""
        current = self.get_blank_threshold()
        new_val = max(0.3, min(0.95, current + delta))
        self._cache["blank_threshold"] = new_val
        self._save()
    
    def record_feedback(self, job_id: str, feedback: dict[str, Any]) -> None:
        """Record a feedback entry for future learning."""
        timestamp = utc_now_iso().replace(":", "-")
        record_file = self.records_dir / f"{timestamp}__{job_id[:8]}.jsonl"
        append_jsonl(record_file, feedback)
    
    def get_all_records(self) -> list[dict[str, Any]]:
        """Load all learning records."""
        records = []
        for record_file in self.records_dir.glob("*.jsonl"):
            try:
                with open(record_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception:
                continue
        return records
    
    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        records = self.get_all_records()
        return {
            "total_records": len(records),
            "caption_corrections": len(self._corrections),
            "cached_parameters": len(self._cache),
            "blank_threshold": self.get_blank_threshold(),
        }


class PairExtractor:
    """Extracts picture/text pairs from workbook pages."""
    
    def __init__(
        self,
        paths: JobPaths,
        config: PairConfig | None = None,
        learning_cache: LearningCache | None = None,
    ):
        self.paths = paths
        self.config = config or PairConfig()
        self.learning_cache = learning_cache
    
    def extract_pairs_from_page(
        self,
        page_id: str,
        page_index: int,
        page_image: Image.Image,
        ocr_tokens: list[dict[str, Any]] | None = None,
    ) -> tuple[list[ItemPair], PagePairDiagnostics]:
        """Extract picture/text pairs from a page.
        
        Args:
            page_id: Page identifier
            page_index: 0-based page index
            page_image: PIL image of the page
            ocr_tokens: Optional OCR tokens for text extraction
        
        Returns:
            (pairs, diagnostics)
        """
        w, h = page_image.size
        page_num = page_index + 1
        page_num_str = f"{page_num:03d}"
        
        # Compute page hash for learning cache lookup
        page_hash = _compute_page_hash(page_image)
        
        # Get learned parameters
        text_ratio = self.config.default_text_ratio
        text_position = self.config.text_region_position
        learned_adjustments = []
        
        if self.learning_cache:
            learned_ratio = self.learning_cache.get_text_ratio(page_hash, default=text_ratio)
            if learned_ratio != text_ratio:
                text_ratio = learned_ratio
                learned_adjustments.append(f"text_ratio={text_ratio:.2f}")
            
            learned_pos = self.learning_cache.get_text_position(page_hash, default=text_position)
            if learned_pos != text_position:
                text_position = learned_pos
                learned_adjustments.append(f"text_position={text_position}")
        
        # Detect item blocks
        grid_detected = False
        grid_rows = 0
        grid_cols = 0
        
        if self.config.grid_detection_enabled:
            try:
                blocks = _detect_item_blocks_grid(page_image, self.config)
                if len(blocks) > 1:
                    grid_detected = True
                    # Estimate grid dimensions
                    ys = sorted(set(b[1] for b in blocks))
                    xs = sorted(set(b[0] for b in blocks))
                    grid_rows = len(ys)
                    grid_cols = len(xs)
            except Exception as e:
                record_error(self.paths, page_id, "pair_grid", str(e))
                blocks = []
        
        if not blocks:
            blocks = _detect_item_blocks_contour(page_image, self.config)
        
        if not blocks:
            # Fallback: treat entire page as single item
            blocks = [(0, 0, w, h)]
        
        # Sort blocks in reading order (top-to-bottom, left-to-right)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        
        # Create output directory for this page
        page_items_dir = self.paths.items_dir / f"page_{page_num_str}"
        page_items_dir.mkdir(parents=True, exist_ok=True)
        
        pairs: list[ItemPair] = []
        
        for item_idx, item_bbox in enumerate(blocks):
            try:
                pair = self._extract_single_pair(
                    page_id=page_id,
                    page_index=page_index,
                    page_image=page_image,
                    item_bbox=item_bbox,
                    item_index=item_idx,
                    page_items_dir=page_items_dir,
                    page_num_str=page_num_str,
                    text_ratio=text_ratio,
                    text_position=text_position,
                    ocr_tokens=ocr_tokens,
                    page_hash=page_hash,
                )
                pairs.append(pair)
            except Exception as e:
                record_error(self.paths, page_id, "pair_extract", f"item_{item_idx}: {e}")
                # Create failed pair entry
                pairs.append(ItemPair(
                    pair_id=_generate_pair_id(page_id, item_idx, list(item_bbox)),
                    page_id=page_id,
                    item_index=item_idx,
                    picture_path=None,
                    text_path=None,
                    caption_text="",
                    bbox_item_xyxy=list(item_bbox),
                    bbox_picture_xyxy=None,
                    bbox_text_xyxy=None,
                    status="review",
                    needs_review=True,
                    reasons=["EXTRACTION_FAILED"],
                    blank_score_picture=1.0,
                    blank_score_text=1.0,
                    confidence=0.0,
                ))
        
        # Create diagnostics
        diagnostics = PagePairDiagnostics(
            page_id=page_id,
            page_index=page_index,
            image_size=[w, h],
            items_detected=len(blocks),
            pairs_extracted=len([p for p in pairs if p.picture_path]),
            pairs_needing_review=sum(1 for p in pairs if p.needs_review),
            grid_detected=grid_detected,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            text_ratio_used=text_ratio,
            text_position_used=text_position,
            learned_adjustments=learned_adjustments,
            created_at=utc_now_iso(),
        )
        
        # Write diagnostics JSON
        diag_path = self.paths.stage_pair_dir / f"page_{page_num_str}.json"
        write_json(diag_path, asdict(diagnostics))
        
        return pairs, diagnostics
    
    def _extract_single_pair(
        self,
        page_id: str,
        page_index: int,
        page_image: Image.Image,
        item_bbox: tuple[int, int, int, int],
        item_index: int,
        page_items_dir: Path,
        page_num_str: str,
        text_ratio: float,
        text_position: str,
        ocr_tokens: list[dict[str, Any]] | None,
        page_hash: str,
    ) -> ItemPair:
        """Extract a single picture/text pair from an item block."""
        w, h = page_image.size
        
        # Clamp item bbox
        clamped = clamp_bbox_xyxy(item_bbox, w=w, h=h)
        if not clamped:
            raise ValueError(f"Invalid item bbox: {item_bbox}")
        
        x0, y0, x1, y1 = clamped
        item_bbox_list = [x0, y0, x1, y1]
        
        # Split into picture and text regions
        picture_bbox, text_bbox = _split_item_into_picture_text(
            page_image, clamped, text_ratio, text_position
        )
        
        # Create item output directory
        item_dir = page_items_dir / f"item_{item_index:03d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        
        reasons: list[str] = []
        needs_review = False
        
        # Crop and save picture
        picture_crop = page_image.crop(picture_bbox)
        picture_path_abs = item_dir / "picture.png"
        picture_crop.save(picture_path_abs, format="PNG")
        picture_path_rel = f"pages/items/page_{page_num_str}/item_{item_index:03d}/picture.png"
        
        # Crop and save text
        text_crop = page_image.crop(text_bbox)
        text_path_abs = item_dir / "text.png"
        text_crop.save(text_path_abs, format="PNG")
        text_path_rel = f"pages/items/page_{page_num_str}/item_{item_index:03d}/text.png"
        
        # Compute blank scores
        blank_picture = compute_blank_score(picture_crop)
        blank_text = compute_blank_score(text_crop)
        
        blank_threshold = self.config.blank_score_threshold
        if self.learning_cache:
            blank_threshold = self.learning_cache.get_blank_threshold(blank_threshold)
        
        if blank_picture.score > blank_threshold:
            reasons.append(PAIR_REASON_BLANK_PICTURE)
            needs_review = True
        
        if blank_text.score > blank_threshold:
            reasons.append(PAIR_REASON_BLANK_TEXT)
            needs_review = True
        
        # Extract caption text
        caption_text = ""
        
        # Check learning cache for corrections first
        pair_signature = f"{page_hash}_{item_index}_{text_bbox}"
        if self.learning_cache:
            cached_caption = self.learning_cache.get_caption_correction(pair_signature)
            if cached_caption:
                caption_text = cached_caption
                reasons.append(PAIR_REASON_LEARNED_CORRECTION)
        
        # If no cached correction, try OCR tokens
        if not caption_text and ocr_tokens:
            # Find tokens that overlap with text region
            tx0, ty0, tx1, ty1 = text_bbox
            matching_tokens = []
            for token in ocr_tokens:
                tok_bbox = token.get("bbox_xyxy")
                if not tok_bbox:
                    continue
                bx0, by0, bx1, by1 = tok_bbox
                # Check overlap
                if (bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0):
                    matching_tokens.append(token)
            
            if matching_tokens:
                # Sort by position and concatenate
                matching_tokens.sort(key=lambda t: (t.get("bbox_xyxy", [0])[1], t.get("bbox_xyxy", [0])[0]))
                caption_text = " ".join(str(t.get("text", "")) for t in matching_tokens).strip()
        
        if not caption_text:
            reasons.append(PAIR_REASON_OCR_EMPTY)
            needs_review = True
        
        # Check item size
        item_area = (x1 - x0) * (y1 - y0)
        if item_area < self.config.min_item_area_px:
            reasons.append(PAIR_REASON_SMALL_ITEM)
            needs_review = True
        
        # Generate pair ID
        pair_id = _generate_pair_id(page_id, item_index, item_bbox_list)
        
        # Compute confidence
        confidence = 1.0
        if blank_picture.score > 0.5:
            confidence -= 0.2
        if blank_text.score > 0.5:
            confidence -= 0.2
        if not caption_text:
            confidence -= 0.3
        if reasons:
            confidence -= 0.1 * len(reasons)
        confidence = max(0.0, min(1.0, confidence))
        
        status = "review" if needs_review else "active"
        
        return ItemPair(
            pair_id=pair_id,
            page_id=page_id,
            item_index=item_index,
            picture_path=picture_path_rel,
            text_path=text_path_rel,
            caption_text=caption_text,
            bbox_item_xyxy=item_bbox_list,
            bbox_picture_xyxy=list(picture_bbox),
            bbox_text_xyxy=list(text_bbox),
            status=status,
            needs_review=needs_review,
            reasons=reasons,
            blank_score_picture=blank_picture.score,
            blank_score_text=blank_text.score,
            confidence=confidence,
        )


def apply_pair_feedback(
    job_dir: str | Path,
    feedback_path: str | Path,
    learning_cache: LearningCache | None = None,
) -> dict[str, int]:
    """Apply human feedback to pair extraction results.
    
    Feedback JSON format:
    {
        "items": [
            {
                "pair_id": "...",
                "action": "approve|reject|edit",
                "edited_caption": "...",
                "corrected_text_bbox": [x0, y0, x1, y1],
                "corrected_text_ratio": 0.35,
                "corrected_text_position": "bottom"
            }
        ]
    }
    
    Returns:
        Statistics dict
    """
    job_dir = Path(job_dir)
    result_pairs_path = job_dir / "result_pairs.json"
    
    if not result_pairs_path.exists():
        return {"error": "result_pairs.json not found"}
    
    result = load_json(result_pairs_path)
    pairs = result.get("pairs", [])
    
    pairs_by_id = {p["pair_id"]: p for p in pairs}
    
    feedback = load_json(feedback_path)
    items = feedback.get("items", []) if isinstance(feedback, dict) else feedback
    
    stats = {
        "processed": 0,
        "approved": 0,
        "rejected": 0,
        "edited": 0,
        "learning_records": 0,
    }
    
    for item in items:
        pair_id = item.get("pair_id")
        action = item.get("action", "").lower()
        
        if not pair_id or pair_id not in pairs_by_id:
            continue
        
        pair = pairs_by_id[pair_id]
        stats["processed"] += 1
        
        # Record for learning
        if learning_cache:
            learning_record = {
                "pair_id": pair_id,
                "page_id": pair.get("page_id"),
                "action": action,
                "original_caption": pair.get("caption_text"),
                "original_status": pair.get("status"),
                "feedback_item": item,
                "timestamp": utc_now_iso(),
            }
            learning_cache.record_feedback(
                job_id=job_dir.name,
                feedback=learning_record,
            )
            stats["learning_records"] += 1
        
        if action == "approve":
            pair["status"] = "active"
            pair["needs_review"] = False
            stats["approved"] += 1
        
        elif action == "reject":
            pair["status"] = "rejected"
            pair["needs_review"] = False
            stats["rejected"] += 1
        
        elif action == "edit":
            edited_caption = item.get("edited_caption")
            if edited_caption is not None:
                pair["caption_text"] = edited_caption
                pair["reasons"] = [r for r in pair.get("reasons", []) if r != PAIR_REASON_OCR_EMPTY]
                
                # Store correction in learning cache
                if learning_cache:
                    # Create signature from page hash + item index + bbox
                    bbox_str = ",".join(str(int(x)) for x in pair.get("bbox_text_xyxy", [0, 0, 0, 0]))
                    signature = f"{pair.get('page_id')}_{pair.get('item_index')}_{bbox_str}"
                    learning_cache.set_caption_correction(signature, edited_caption)
            
            # Handle text ratio correction
            corrected_ratio = item.get("corrected_text_ratio")
            if corrected_ratio is not None and learning_cache:
                # We need page hash - for now, use page_id as proxy
                page_hash = pair.get("page_id", "")
                learning_cache.set_text_ratio(page_hash, float(corrected_ratio))
            
            # Handle text position correction
            corrected_pos = item.get("corrected_text_position")
            if corrected_pos and learning_cache:
                page_hash = pair.get("page_id", "")
                learning_cache.set_text_position(page_hash, corrected_pos)
            
            pair["status"] = "active"
            pair["needs_review"] = False
            stats["edited"] += 1
    
    # Save updated result
    result["pairs"] = list(pairs_by_id.values())
    result["feedback_applied_at"] = utc_now_iso()
    write_json(result_pairs_path, result)
    
    return stats
