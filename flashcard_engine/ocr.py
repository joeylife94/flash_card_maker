from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

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

    def extract(self, page_id: str, image: Image.Image) -> dict[str, Any]:
        """Return raw OCR result in a stable schema.

        Always fail-soft: if PaddleOCR missing or fails, returns empty result.
        """
        raw_path = self.paths.stage_ocr_dir / f"{page_id}_raw.json"

        try:
            from paddleocr import PaddleOCR  # type: ignore

            # Lazily initialize once per job run.
            if self._ocr is None:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            # PaddleOCR expects numpy array
            import numpy as np

            arr = np.array(image)
            try:
                # Older PaddleOCR versions accepted cls kwarg; newer versions may not.
                result = self._ocr.ocr(arr, cls=True)
            except TypeError:
                result = self._ocr.ocr(arr)

            tokens: list[dict[str, Any]] = []
            # result: list[ [ [poly], (text, score) ], ... ] nested by lines
            for line in result or []:
                for item in line or []:
                    poly, (text, score) = item
                    xyxy = _poly_to_xyxy(poly)
                    tokens.append(
                        {
                            "text": text,
                            "confidence": float(score),
                            "bbox_xyxy": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                        }
                    )

            raw = {"page_id": page_id, "tokens": tokens}
            write_json(raw_path, raw)
            return raw
        except Exception as e:
            record_error(self.paths, page_id=page_id, stage="ocr", message=str(e))
            raw = {"page_id": page_id, "tokens": [], "error": str(e)}
            write_json(raw_path, raw)
            return raw
