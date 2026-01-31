"""Panel Debug Pack Generator.

Provides visibility into panel extraction failures through:
1. Annotated page images with detected bounding boxes
2. Panel crop images for each detected block
3. Per-page diagnostics JSON with detailed metrics
4. Blank/failure detection via blank_score computation

This module is designed to support human-in-the-loop review and
future model training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .job import JobPaths, record_error
from .utils import clamp_bbox_xyxy, slugify, utc_now_iso, write_json


# Panel diagnostic reason codes
PR_BLANK_CROP = "BLANK_CROP"
PR_OCR_EMPTY = "OCR_EMPTY"
PR_PARTIAL_CROP = "PARTIAL_CROP"
PR_OVERLAP = "OVERLAP"
PR_FALLBACK_USED = "FALLBACK_USED"
PR_BBOX_INVALID = "BBOX_INVALID"
PR_SMALL_AREA = "SMALL_AREA"
PR_LOW_EDGE_DENSITY = "LOW_EDGE_DENSITY"
PR_LOW_ENTROPY = "LOW_ENTROPY"


@dataclass
class BlankScoreResult:
    """Result of blank_score computation."""
    score: float  # 0.0 = content-rich, 1.0 = blank
    entropy: float  # grayscale entropy (higher = more content)
    edge_density: float  # edge pixel ratio (higher = more edges)
    is_blank: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class PanelDiagnostic:
    """Diagnostic information for a single detected panel."""
    block_index: int
    bbox_xyxy: list[int]
    blank_score: float
    entropy: float
    edge_density: float
    ocr_text_raw: str
    caption_text: str
    reasons: list[str]
    needs_review: bool
    status: str  # active / review / rejected
    crop_path: str | None
    

@dataclass
class PanelDebugConfig:
    """Configuration for panel debug pack generation."""
    blank_score_threshold: float = 0.7  # above this = blank
    entropy_threshold: float = 3.5  # below this = low content
    edge_density_threshold: float = 0.02  # below this = few edges
    min_area_px: int = 1000
    annotation_font_size: int = 16
    bbox_line_width: int = 3
    colors: dict[str, str] = field(default_factory=lambda: {
        "active": "#00FF00",  # green
        "review": "#FFAA00",  # orange
        "rejected": "#FF0000",  # red
        "blank": "#888888",  # gray
    })


def compute_blank_score(
    image: Image.Image,
    *,
    entropy_threshold: float = 3.5,
    edge_density_threshold: float = 0.02,
) -> BlankScoreResult:
    """Compute a blank/meaningless score for an image crop.
    
    Uses two complementary metrics:
    1. Grayscale entropy: measures information content (histogram-based)
    2. Edge density: measures structural content (Sobel-like gradient)
    
    Returns:
        BlankScoreResult with score in [0, 1] where 1 = blank
    """
    reasons: list[str] = []
    
    # Convert to grayscale numpy array
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)
    
    # 1. Compute entropy from histogram
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)  # normalize
    hist = hist[hist > 0]  # only non-zero bins
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # 2. Compute edge density using simple gradient
    # Sobel-like kernels via numpy operations
    if arr.shape[0] > 2 and arr.shape[1] > 2:
        # Simple gradient magnitude
        gx = np.abs(arr[:, 1:] - arr[:, :-1])
        gy = np.abs(arr[1:, :] - arr[:-1, :])
        
        # Threshold to get edge pixels
        edge_threshold = 30.0  # intensity threshold for edge detection
        edge_pixels_x = np.sum(gx > edge_threshold)
        edge_pixels_y = np.sum(gy > edge_threshold)
        total_pixels = arr.size
        edge_density = (edge_pixels_x + edge_pixels_y) / (2 * total_pixels + 1e-10)
    else:
        edge_density = 0.0
    
    # Determine if blank
    is_blank = False
    
    if entropy < entropy_threshold:
        is_blank = True
        reasons.append(PR_LOW_ENTROPY)
    
    if edge_density < edge_density_threshold:
        is_blank = True
        reasons.append(PR_LOW_EDGE_DENSITY)
    
    # Compute combined score (weighted average, inverted so 1 = blank)
    # Normalize entropy to [0, 1] assuming max entropy ~ 8 bits
    norm_entropy = min(max(entropy / 8.0, 0.0), 1.0)
    # Edge density is already in [0, 1] range (typically much lower)
    norm_edge = min(max(edge_density * 10, 0.0), 1.0)  # scale up since typical values are small
    
    # Combined score: lower content = higher blank score
    content_score = 0.6 * norm_entropy + 0.4 * norm_edge
    blank_score = 1.0 - content_score
    
    # Override if individual metrics indicate blank
    if is_blank:
        blank_score = max(blank_score, 0.7)
    
    # Clamp to [0, 1] to handle floating point precision issues
    blank_score = min(max(blank_score, 0.0), 1.0)
    
    return BlankScoreResult(
        score=float(blank_score),
        entropy=float(entropy),
        edge_density=float(edge_density),
        is_blank=is_blank,
        reasons=reasons,
    )


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font for annotation, with fallback."""
    try:
        # Try common system fonts
        for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except (OSError, IOError):
                continue
        # Fallback to default
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def generate_annotated_page(
    page_image: Image.Image,
    panels: list[PanelDiagnostic],
    config: PanelDebugConfig,
) -> Image.Image:
    """Generate an annotated page image with panel bounding boxes.
    
    Args:
        page_image: Original page image
        panels: List of panel diagnostics
        config: Debug configuration
        
    Returns:
        Annotated image with bboxes and labels
    """
    # Create a copy to draw on
    annotated = page_image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = _get_font(config.annotation_font_size)
    
    for panel in panels:
        # Determine color based on status
        if panel.blank_score > config.blank_score_threshold:
            color = config.colors.get("blank", "#888888")
        else:
            color = config.colors.get(panel.status, "#FFFFFF")
        
        bbox = panel.bbox_xyxy
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Draw bounding box
        draw.rectangle(
            [(x0, y0), (x1, y1)],
            outline=color,
            width=config.bbox_line_width,
        )
        
        # Draw label with block index
        label = f"#{panel.block_index}"
        if panel.needs_review:
            label += " [R]"
        if panel.blank_score > config.blank_score_threshold:
            label += " [B]"
        
        # Position label at top-left of bbox
        label_pos = (x0 + 2, y0 + 2)
        
        # Draw label background
        try:
            text_bbox = draw.textbbox(label_pos, label, font=font)
            draw.rectangle(text_bbox, fill="black")
        except Exception:
            pass
        
        draw.text(label_pos, label, fill=color, font=font)
        
        # Draw reading order number at center
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        order_label = str(panel.block_index)
        try:
            order_bbox = draw.textbbox((cx, cy), order_label, font=font)
            ow = order_bbox[2] - order_bbox[0]
            oh = order_bbox[3] - order_bbox[1]
            draw.rectangle(
                [(cx - ow//2 - 2, cy - oh//2 - 2), (cx + ow//2 + 2, cy + oh//2 + 2)],
                fill="black",
            )
            draw.text((cx - ow//2, cy - oh//2), order_label, fill="white", font=font)
        except Exception:
            draw.text((cx, cy), order_label, fill="white", font=font)
    
    return annotated


def sort_panels_reading_order(
    panels: list[dict[str, Any]],
    image_height: int,
) -> list[dict[str, Any]]:
    """Sort panels in reading order: top-to-bottom, left-to-right.
    
    Uses a row-based grouping approach:
    1. Group panels by approximate y-position (within row_threshold)
    2. Sort groups by y-position
    3. Within each group, sort by x-position
    """
    if not panels:
        return []
    
    # Row threshold: panels within this y-distance are considered same row
    row_threshold = image_height * 0.05  # 5% of image height
    
    # Extract panel data with y-center for sorting
    panel_data = []
    for i, p in enumerate(panels):
        bbox = p.get("bbox_xyxy", [0, 0, 0, 0])
        y_center = (bbox[1] + bbox[3]) / 2
        x_center = (bbox[0] + bbox[2]) / 2
        panel_data.append({
            "original_index": i,
            "panel": p,
            "y_center": y_center,
            "x_center": x_center,
        })
    
    # Sort by y_center first
    panel_data.sort(key=lambda x: x["y_center"])
    
    # Group into rows
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_y = None
    
    for pd in panel_data:
        if current_y is None or pd["y_center"] - current_y <= row_threshold:
            current_row.append(pd)
            if current_y is None:
                current_y = pd["y_center"]
        else:
            if current_row:
                rows.append(current_row)
            current_row = [pd]
            current_y = pd["y_center"]
    
    if current_row:
        rows.append(current_row)
    
    # Sort each row by x_center
    for row in rows:
        row.sort(key=lambda x: x["x_center"])
    
    # Flatten and assign reading order indices
    sorted_panels = []
    for row in rows:
        for pd in row:
            sorted_panels.append(pd["panel"])
    
    return sorted_panels


class PanelDebugPack:
    """Generator for panel debug artifacts."""
    
    def __init__(
        self,
        paths: JobPaths,
        config: PanelDebugConfig | None = None,
    ):
        self.paths = paths
        self.config = config or PanelDebugConfig()
        
        # Ensure debug directories exist
        self.annotated_dir = paths.pages_dir / "annotated"
        self.panels_dir = paths.pages_dir / "panels"
        self.stage_panel_dir = paths.job_dir / "stage" / "panel"
        
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.stage_panel_dir.mkdir(parents=True, exist_ok=True)
    
    def process_page(
        self,
        page_id: str,
        page_index: int,
        page_image: Image.Image,
        tokens: list[dict[str, Any]],
    ) -> list[PanelDiagnostic]:
        """Process a page and generate debug artifacts.
        
        Args:
            page_id: Page identifier (e.g., "page_001")
            page_index: 0-based page index
            page_image: PIL image of the page
            tokens: List of OCR tokens with bbox_xyxy and text
            
        Returns:
            List of PanelDiagnostic objects
        """
        w, h = page_image.size
        page_num = page_index + 1
        page_num_str = f"{page_num:03d}"
        
        # Sort tokens in reading order
        sorted_tokens = sort_panels_reading_order(tokens, h)
        
        # Process each token as a panel
        diagnostics: list[PanelDiagnostic] = []
        
        # Create panel crops directory for this page
        page_panels_dir = self.panels_dir / f"page_{page_num_str}"
        page_panels_dir.mkdir(parents=True, exist_ok=True)
        
        for block_idx, token in enumerate(sorted_tokens):
            try:
                diag = self._process_single_panel(
                    block_index=block_idx,
                    token=token,
                    page_image=page_image,
                    page_num_str=page_num_str,
                    page_panels_dir=page_panels_dir,
                    page_id=page_id,
                )
                diagnostics.append(diag)
            except Exception as e:
                record_error(
                    self.paths,
                    page_id=page_id,
                    stage="panel_debug",
                    message=f"block_{block_idx}: {e}",
                )
                # Create minimal diagnostic for failed panel
                diagnostics.append(PanelDiagnostic(
                    block_index=block_idx,
                    bbox_xyxy=[0, 0, 0, 0],
                    blank_score=1.0,
                    entropy=0.0,
                    edge_density=0.0,
                    ocr_text_raw="",
                    caption_text="",
                    reasons=[PR_BBOX_INVALID],
                    needs_review=True,
                    status="review",
                    crop_path=None,
                ))
        
        # Generate annotated page image
        annotated_path = self.annotated_dir / f"page_{page_num_str}_panels.png"
        try:
            annotated = generate_annotated_page(page_image, diagnostics, self.config)
            annotated.save(annotated_path, format="PNG")
        except Exception as e:
            record_error(
                self.paths,
                page_id=page_id,
                stage="panel_annotate",
                message=str(e),
            )
        
        # Write diagnostics JSON
        diag_path = self.stage_panel_dir / f"page_{page_num_str}.json"
        diag_data = {
            "page_id": page_id,
            "page_index": page_index,
            "image_size": [w, h],
            "panels_count": len(diagnostics),
            "panels_needing_review": sum(1 for d in diagnostics if d.needs_review),
            "panels_blank": sum(1 for d in diagnostics if d.blank_score > self.config.blank_score_threshold),
            "created_at": utc_now_iso(),
            "panels": [
                {
                    "block_index": d.block_index,
                    "bbox_xyxy": d.bbox_xyxy,
                    "blank_score": round(d.blank_score, 4),
                    "entropy": round(d.entropy, 4),
                    "edge_density": round(d.edge_density, 6),
                    "ocr_text_raw": d.ocr_text_raw,
                    "caption_text": d.caption_text,
                    "reasons": d.reasons,
                    "needs_review": d.needs_review,
                    "status": d.status,
                    "crop_path": d.crop_path,
                }
                for d in diagnostics
            ],
        }
        write_json(diag_path, diag_data)
        
        return diagnostics
    
    def _process_single_panel(
        self,
        block_index: int,
        token: dict[str, Any],
        page_image: Image.Image,
        page_num_str: str,
        page_panels_dir: Path,
        page_id: str,
    ) -> PanelDiagnostic:
        """Process a single panel/token."""
        w, h = page_image.size
        
        bbox = token.get("bbox_xyxy")
        text_raw = str(token.get("text") or "").strip()
        caption = text_raw  # For now, caption = cleaned text
        
        reasons: list[str] = []
        needs_review = False
        status = "active"
        crop_path: str | None = None
        
        # Validate bbox
        if not bbox or len(bbox) != 4:
            return PanelDiagnostic(
                block_index=block_index,
                bbox_xyxy=[0, 0, 0, 0],
                blank_score=1.0,
                entropy=0.0,
                edge_density=0.0,
                ocr_text_raw=text_raw,
                caption_text=caption,
                reasons=[PR_BBOX_INVALID],
                needs_review=True,
                status="review",
                crop_path=None,
            )
        
        clamped = clamp_bbox_xyxy(bbox, w=w, h=h)
        if not clamped:
            return PanelDiagnostic(
                block_index=block_index,
                bbox_xyxy=[int(x) for x in bbox],
                blank_score=1.0,
                entropy=0.0,
                edge_density=0.0,
                ocr_text_raw=text_raw,
                caption_text=caption,
                reasons=[PR_BBOX_INVALID],
                needs_review=True,
                status="review",
                crop_path=None,
            )
        
        x0, y0, x1, y1 = clamped
        area = (x1 - x0) * (y1 - y0)
        
        # Check minimum area
        if area < self.config.min_area_px:
            reasons.append(PR_SMALL_AREA)
            needs_review = True
        
        # Crop the panel
        try:
            crop = page_image.crop(clamped)
            
            # Compute blank score
            blank_result = compute_blank_score(
                crop,
                entropy_threshold=self.config.entropy_threshold,
                edge_density_threshold=self.config.edge_density_threshold,
            )
            
            if blank_result.is_blank:
                reasons.append(PR_BLANK_CROP)
                reasons.extend(blank_result.reasons)
                needs_review = True
            
            # Save crop
            crop_filename = f"panel_{block_index:03d}.png"
            crop_abs_path = page_panels_dir / crop_filename
            crop.save(crop_abs_path, format="PNG")
            
            # Relative path from job directory
            crop_path = f"pages/panels/page_{page_num_str}/{crop_filename}"
            
        except Exception as e:
            record_error(
                self.paths,
                page_id=page_id,
                stage="panel_crop",
                message=f"block_{block_index}: {e}",
            )
            blank_result = BlankScoreResult(
                score=1.0,
                entropy=0.0,
                edge_density=0.0,
                is_blank=True,
                reasons=[PR_BLANK_CROP],
            )
            reasons.append(PR_BLANK_CROP)
            needs_review = True
        
        # Check OCR
        if not text_raw:
            reasons.append(PR_OCR_EMPTY)
            needs_review = True
        
        # Determine final status
        if needs_review:
            status = "review"
        
        return PanelDiagnostic(
            block_index=block_index,
            bbox_xyxy=[int(x0), int(y0), int(x1), int(y1)],
            blank_score=blank_result.score,
            entropy=blank_result.entropy,
            edge_density=blank_result.edge_density,
            ocr_text_raw=text_raw,
            caption_text=caption,
            reasons=list(set(reasons)),  # dedupe
            needs_review=needs_review,
            status=status,
            crop_path=crop_path,
        )
    
    def get_artifact_paths(self, page_index: int) -> dict[str, str]:
        """Get relative paths for debug artifacts of a page."""
        page_num_str = f"{page_index + 1:03d}"
        return {
            "annotated_image": f"pages/annotated/page_{page_num_str}_panels.png",
            "panels_dir": f"pages/panels/page_{page_num_str}",
            "diagnostics_json": f"stage/panel/page_{page_num_str}.json",
        }
