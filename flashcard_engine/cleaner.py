from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .job import JobPaths
from .utils import compile_patterns, is_numeric_only, write_json


@dataclass
class TextCleaner:
    paths: JobPaths
    cleanup_cfg: dict[str, Any]

    def clean(self, page_id: str, raw: dict[str, Any]) -> dict[str, Any]:
        clean_path = self.paths.stage_ocr_dir / f"{page_id}_clean.json"

        lowercase = bool(self.cleanup_cfg.get("lowercase", True))
        remove_numeric_only = bool(self.cleanup_cfg.get("remove_numeric_only", True))
        min_len = int(self.cleanup_cfg.get("min_token_length", 3))
        allow_short = set([t.lower() for t in self.cleanup_cfg.get("allow_short_tokens", [])])
        drop_patterns = self.cleanup_cfg.get("drop_patterns", [])
        max_tokens = int(self.cleanup_cfg.get("max_tokens_per_page", 200))

        patterns = compile_patterns(list(drop_patterns))

        cleaned: list[dict[str, Any]] = []
        for t in raw.get("tokens", []):
            text = (t.get("text") or "").strip()
            if lowercase:
                text = text.lower()

            if not text:
                continue
            if remove_numeric_only and is_numeric_only(text):
                continue
            if any(p.fullmatch(text) for p in patterns):
                continue
            if len(text) < min_len and text not in allow_short:
                continue

            cleaned.append({"text": text, "confidence": float(t.get("confidence", 0.0)), "bbox_xyxy": t.get("bbox_xyxy")})
            if len(cleaned) >= max_tokens:
                break

        out = {"page_id": page_id, "tokens": cleaned, "raw_count": len(raw.get("tokens", [])), "clean_count": len(cleaned)}
        write_json(clean_path, out)
        return out
