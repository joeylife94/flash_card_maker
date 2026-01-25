from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .job import JobPaths, record_error
from .utils import clamp_bbox_xyxy, clamp_int, slugify


@dataclass
class CropStats:
    tokens_seen: int = 0
    crops_written: int = 0
    crop_failures: int = 0
    crops_gated_small: int = 0
    crops_gated_ratio: int = 0


def _pad_bbox(b: tuple[int, int, int, int], *, pad_px: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = clamp_int(x0 - pad_px, 0, max(0, w - 1))
    y0 = clamp_int(y0 - pad_px, 0, max(0, h - 1))
    x1 = clamp_int(x1 + pad_px, 1, max(1, w))
    y1 = clamp_int(y1 + pad_px, 1, max(1, h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def crop_multiword_tokens_for_page(
    *,
    paths: JobPaths,
    page_index: int,
    page_id: str,
    image: Image.Image,
    tokens: list[dict[str, Any]],
    crop_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], CropStats]:
    """Create bbox-based crops for multi_word tokens.

    Side effects:
    - Writes PNGs under pages/crops/page_<n>/token_<i>_<slug>.png
    - Annotates each token dict with:
      - crop_path (relative under job dir) on success
      - crop_status: success|skipped|failed
      - crop_bbox_xyxy: [x0,y0,x1,y1] for the crop bbox
      - warnings: list[str]

    Fail-soft:
    - Never raises; errors are recorded to errors.jsonl.
    """
    w, h = image.size

    pad_px = int(crop_cfg.get("bbox_crop_padding_px", 8))
    min_area_px = int(crop_cfg.get("bbox_crop_min_area_px", 64 * 64))
    min_area_ratio = float(crop_cfg.get("bbox_crop_min_area_ratio", 0.0))
    max_area_ratio = float(crop_cfg.get("bbox_crop_max_area_ratio", 0.80))

    page_num = int(page_index) + 1
    out_dir = paths.crops_dir / f"page_{page_num:03d}"

    stats = CropStats()

    out_tokens: list[dict[str, Any]] = []
    for i, t in enumerate(tokens):
        stats.tokens_seen += 1

        text = (t.get("text") or "").strip()
        bbox = t.get("bbox_xyxy")

        token_out = dict(t)
        token_out.setdefault("warnings", [])

        if not text or not bbox:
            token_out["crop_status"] = "skipped"
            token_out["crop_path"] = None
            out_tokens.append(token_out)
            continue

        try:
            clamped = clamp_bbox_xyxy(bbox, w=w, h=h)
            if not clamped:
                token_out["crop_status"] = "failed"
                token_out["crop_path"] = None
                token_out["warnings"].append("BBOX_INVALID")
                out_tokens.append(token_out)
                continue

            crop_bbox = _pad_bbox(clamped, pad_px=pad_px, w=w, h=h)
            cw = max(1, crop_bbox[2] - crop_bbox[0])
            ch = max(1, crop_bbox[3] - crop_bbox[1])
            area = int(cw * ch)
            area_ratio = area / float(max(1, w * h))

            token_out["crop_bbox_xyxy"] = [crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]]

            # Quality gates (do not crash; mark warning and optionally skip)
            gated = False
            if area < min_area_px:
                token_out["warnings"].append("CROP_GATED_SMALL")
                stats.crops_gated_small += 1
                gated = True

            if area_ratio < min_area_ratio:
                token_out["warnings"].append("CROP_GATED_RATIO")
                stats.crops_gated_ratio += 1
                gated = True

            if gated:
                token_out["crop_status"] = "failed"
                token_out["crop_path"] = None
                out_tokens.append(token_out)
                continue

            if area_ratio > max_area_ratio:
                token_out["warnings"].append("BBOX_TOO_LARGE")

            slug = slugify(text, max_len=60, add_hash=True)
            rel = f"pages/crops/page_{page_num:03d}/token_{i:04d}_{slug}.png"
            abs_path = paths.job_dir / rel

            crop = image.crop(crop_bbox)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(abs_path, format="PNG")

            token_out["crop_status"] = "success"
            token_out["crop_path"] = rel
            stats.crops_written += 1
            out_tokens.append(token_out)
        except Exception as e:
            stats.crop_failures += 1
            record_error(paths, page_id=page_id, stage="bbox_crop", message=f"token_{i}: {e}")
            token_out["crop_status"] = "failed"
            token_out["crop_path"] = None
            token_out.setdefault("warnings", []).append("CROP_EXCEPTION")
            out_tokens.append(token_out)

    return out_tokens, stats
