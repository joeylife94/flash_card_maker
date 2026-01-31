from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .job import JobPaths, record_error
from .utils import clamp, safe_filename_token, write_json

# Suppress warnings
warnings.filterwarnings('ignore')


def _expand_bbox_xyxy(b: list[int] | tuple[int, int, int, int], scale: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale
    nx0 = int(clamp(cx - bw / 2.0, 0, w - 1))
    ny0 = int(clamp(cy - bh / 2.0, 0, h - 1))
    nx1 = int(clamp(cx + bw / 2.0, 1, w))
    ny1 = int(clamp(cy + bh / 2.0, 1, h))
    if nx1 <= nx0:
        nx1 = min(w, nx0 + 1)
    if ny1 <= ny0:
        ny1 = min(h, ny0 + 1)
    return nx0, ny0, nx1, ny1


@dataclass
class Segmenter:
    mode: str  # off|mobilesam|fastsam
    device: str  # cpu|cuda|mps
    paths: JobPaths
    segment_cfg: dict[str, Any]
    
    def _enhance_crop(self, crop: Image.Image) -> Image.Image:
        """크롭된 이미지 품질 향상."""
        try:
            # 엣지 강조
            enhanced = crop.filter(ImageFilter.EDGE_ENHANCE)
            # 선명도 향상
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
            return enhanced
        except Exception:
            return crop
    
    def _smart_crop_with_padding(self, image: Image.Image, bbox: tuple[int, int, int, int], padding: int = 5) -> Image.Image:
        """패딩을 추가한 스마트 크롭."""
        w, h = image.size
        x0, y0, x1, y1 = bbox
        
        # 패딩 추가
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(w, x1 + padding)
        y1 = min(h, y1 + padding)
        
        return image.crop((x0, y0, x1, y1))

    def run_single_word(self, page_id: str, image: Image.Image, word: str, token_bbox: list[int] | None) -> dict[str, Any]:
        out_path = self.paths.stage_segment_dir / f"{page_id}.json"

        if self.mode == "off":
            out = {"page_id": page_id, "status": "skipped", "reason": "segmenter_off"}
            write_json(out_path, out)
            return out

        if not token_bbox:
            out = {"page_id": page_id, "status": "skipped", "reason": "missing_bbox"}
            write_json(out_path, out)
            return out

        w, h = image.size
        scale = float(self.segment_cfg.get("expand_scale", 4.0))
        min_area_ratio = float(self.segment_cfg.get("min_area_ratio", 0.02))

        seg_bbox = _expand_bbox_xyxy(token_bbox, scale=scale, w=w, h=h)
        seg_w = max(1, seg_bbox[2] - seg_bbox[0])
        seg_h = max(1, seg_bbox[3] - seg_bbox[1])
        area_ratio = (seg_w * seg_h) / float(w * h)

        safe_word = safe_filename_token(word)
        crop_rel = f"pages/crops/{page_id}_{safe_word}.png"
        crop_abs = self.paths.job_dir / crop_rel

        # Hard rule: reject too-small regions.
        if area_ratio < min_area_ratio:
            out = {
                "page_id": page_id,
                "status": "failed",
                "method": self.mode,
                "model_status": None,
                "seg_bbox": [seg_bbox[0], seg_bbox[1], seg_bbox[2], seg_bbox[3]],
                "area_ratio": float(area_ratio),
                "crop_path": None,
                "reason": "area_ratio_too_small",
            }
            write_json(out_path, out)
            return out

        model_status: str | None = None
        status = "failed"
        method = self.mode

        try:
            model_status, model_method = self._run_model_segmenter(
                image=image, seg_bbox=seg_bbox, crop_abs=crop_abs
            )
            if model_status == "success":
                status = "success"
                method = model_method
            else:
                # Fail-soft fallback: produce enhanced crop from bbox
                crop = self._smart_crop_with_padding(image, seg_bbox, padding=5)
                crop = self._enhance_crop(crop)
                crop_abs.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_abs, format="PNG", optimize=True, quality=95)
                status = "success"
                method = "bbox_fallback_enhanced"
        except Exception as e:
            record_error(self.paths, page_id=page_id, stage="segment", message=str(e))
            # 최후 폴백: 기본 bbox 크롭
            try:
                crop = image.crop(seg_bbox)
                crop_abs.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_abs, format="PNG")
                status = "success"
                method = "bbox_fallback"
            except Exception:
                status = "failed"
                method = self.mode

        out = {
            "page_id": page_id,
            "status": status,
            "method": method,
            "model_status": model_status,
            "seg_bbox": [seg_bbox[0], seg_bbox[1], seg_bbox[2], seg_bbox[3]],
            "area_ratio": float(area_ratio),
            "crop_path": crop_rel if status == "success" else None,
        }
        write_json(out_path, out)
        return out

    def _run_model_segmenter(self, image: Image.Image, seg_bbox: tuple[int, int, int, int], crop_abs) -> tuple[str, str]:
        """Run MobileSAM/FastSAM as box-prompted segmenter.

        MVP note: This is intentionally best-effort. If libraries/models are not installed,
        we return failed and let pipeline fallback to the full page image.
        """
        mode = self.mode
        if mode == "fastsam":
            try:
                from ultralytics import FastSAM
                from ultralytics.models.fastsam import FastSAMPrompt
                
                # 모델 로드 (캐싱)
                model = FastSAM("FastSAM-s.pt")
                img = np.array(image)
                
                # 추론 실행
                results = model(img, device=self.device, verbose=False, conf=0.4, iou=0.9)
                prompt = FastSAMPrompt(img, results, device=self.device)
                
                # Box prompt
                masks = prompt.box_prompt(bbox=list(seg_bbox))
                
                if masks is None or len(masks) == 0:
                    return "failed", "fastsam"

                # 가장 큰 마스크 선택
                areas = [int(np.sum(m)) for m in masks]
                idx = int(max(range(len(areas)), key=lambda i: areas[i]))
                mask = masks[idx]
                
                # 마스크 후처리: 모폴로지 연산
                from scipy import ndimage
                mask = ndimage.binary_closing(mask, iterations=2)
                mask = ndimage.binary_opening(mask, iterations=1)

                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    return "failed", "fastsam"
                    
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                
                # 상하 좌우 5픽셀 패딩
                h, w = image.size[1], image.size[0]
                x0 = max(0, x0 - 5)
                y0 = max(0, y0 - 5)
                x1 = min(w, x1 + 5)
                y1 = min(h, y1 + 5)

                # Crop and apply alpha
                crop = image.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGBA")
                mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
                alpha = (mask_crop.astype("uint8") * 255)
                import PIL.Image

                alpha_img = PIL.Image.fromarray(alpha, mode="L")
                crop.putalpha(alpha_img)

                crop_abs.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_abs, format="PNG")
                return "success", "fastsam"
            except Exception:
                return "failed", "fastsam"

        if mode == "mobilesam":
            # Placeholder for MobileSAM integration (best-effort).
            # If user installs a MobileSAM-compatible package, wire it here.
            return "failed", "mobilesam"

        return "failed", mode
