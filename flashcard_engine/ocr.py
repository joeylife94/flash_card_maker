from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from .job import JobPaths, record_error
from .utils import write_json


def _poly_to_xyxy(poly: list[list[float]] | list[tuple[float, float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return int(x0), int(y0), int(x1), int(y1)


@dataclass
class OCRExtractor:
    lang: str
    paths: JobPaths
    _ocr: Any | None = None
    _ocr_engine: str = "auto"  # auto, paddleocr, easyocr
    _use_preprocessing: bool = True
    _max_retries: int = 2

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """이미지 전처리로 OCR 정확도 향상."""
        if not self._use_preprocessing:
            return image
        
        try:
            # PIL to OpenCV
            img_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 1. 적응형 이진화 (조명 변화에 강함)
            binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )
            
            # 2. 노이즈 제거
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # 3. 샤프닝
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # OpenCV to PIL
            processed = Image.fromarray(sharpened)
            
            # 4. 대비 향상
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.5)
            
            return processed.convert('RGB')
        except Exception:
            return image
    
    def extract(self, page_id: str, image: Image.Image) -> dict[str, Any]:
        """Return raw OCR result in a stable schema.

        Always fail-soft: if OCR fails, returns empty result.
        Tries multiple engines and preprocessing strategies.
        """
        raw_path = self.paths.stage_ocr_dir / f"{page_id}_raw.json"

        # 재시도 로직
        for attempt in range(self._max_retries):
            try:
                # 전처리 적용 (첫 시도 실패시 두 번째는 전처리 강화)
                if attempt == 0:
                    processed_image = image
                else:
                    processed_image = self._preprocess_image(image)
                
                tokens = self._extract_with_engine(page_id, processed_image)
                
                if tokens:  # 성공
                    raw = {"page_id": page_id, "tokens": tokens, "attempt": attempt + 1}
                    write_json(raw_path, raw)
                    return raw
                
                # 토큰이 없으면 다시 시도
                if attempt < self._max_retries - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                if attempt == self._max_retries - 1:
                    record_error(self.paths, page_id=page_id, stage="ocr", message=str(e))
                    raw = {"page_id": page_id, "tokens": [], "error": str(e)}
                    write_json(raw_path, raw)
                    return raw
        
        # 모든 시도 실패
        raw = {"page_id": page_id, "tokens": []}
        write_json(raw_path, raw)
        return raw
    
    def _extract_with_engine(self, page_id: str, image: Image.Image) -> list[dict[str, Any]]:
        """실제 OCR 엔진으로 텍스트 추출."""
        # 엔진 자동 선택
        if self._ocr_engine == "auto":
            # EasyOCR 먼저 시도 (numpy 2.x 호환)
            try:
                return self._extract_easyocr(image)
            except Exception:
                # PaddleOCR 폴백
                try:
                    return self._extract_paddleocr(image)
                except Exception:
                    return []
        elif self._ocr_engine == "easyocr":
            return self._extract_easyocr(image)
        elif self._ocr_engine == "paddleocr":
            return self._extract_paddleocr(image)
        
        return []
    
    def _extract_easyocr(self, image: Image.Image) -> list[dict[str, Any]]:
        """EasyOCR로 텍스트 추출."""
        try:
            import easyocr
            
            if self._ocr is None or not isinstance(self._ocr, easyocr.Reader):
                # 언어 코드 변환 (en, ko 등)
                langs = self.lang.split(',')
                self._ocr = easyocr.Reader(langs, gpu=False)
            
            arr = np.array(image)
            results = self._ocr.readtext(arr)
            
            tokens = []
            for (bbox, text, confidence) in results:
                # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                xyxy = _poly_to_xyxy(bbox)
                tokens.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox_xyxy": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                })
            
            return tokens
        except Exception:
            raise
    
    def _extract_paddleocr(self, image: Image.Image) -> list[dict[str, Any]]:
        """PaddleOCR로 텍스트 추출."""
        from paddleocr import PaddleOCR
        
        if self._ocr is None or not hasattr(self._ocr, 'ocr'):
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        
        arr = np.array(image)
        try:
            result = self._ocr.ocr(arr, cls=True)
        except TypeError:
            result = self._ocr.ocr(arr)
        
        tokens = []
        for line in result or []:
            for item in line or []:
                poly, (text, score) = item
                xyxy = _poly_to_xyxy(poly)
                tokens.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox_xyxy": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                })
        
        return tokens
