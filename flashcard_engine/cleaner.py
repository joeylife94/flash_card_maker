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

        # v0.2 knobs
        enable_dedupe = bool(self.cleanup_cfg.get("dedupe_enabled", False))
        drop_punct_only = bool(self.cleanup_cfg.get("drop_punctuation_only", True))
        drop_suspicious = bool(self.cleanup_cfg.get("drop_suspicious_tokens", False))
        suspicious_patterns = compile_patterns(list(self.cleanup_cfg.get("suspicious_token_patterns", [])))

        patterns = compile_patterns(list(drop_patterns))

        cleaned: list[dict[str, Any]] = []
        for t in raw.get("tokens", []):
            text = (t.get("text") or "").strip()
            if lowercase:
                text = text.lower()

            if not text:
                continue

            # Drop pure punctuation / non-alnum tokens (configurable).
            if drop_punct_only and not any(ch.isalnum() for ch in text):
                continue

            if remove_numeric_only and is_numeric_only(text):
                continue
            if any(p.fullmatch(text) for p in patterns):
                continue

            if drop_suspicious and any(p.search(text) for p in suspicious_patterns):
                continue

            if len(text) < min_len and text not in allow_short:
                continue

            cleaned.append({"text": text, "confidence": float(t.get("confidence", 0.0)), "bbox_xyxy": t.get("bbox_xyxy")})
            if len(cleaned) >= max_tokens:
                break

        deduped_count = 0
        if enable_dedupe and cleaned:
            # Dedupe by exact normalized text; keep highest-confidence instance.
            best_by_text: dict[str, dict[str, Any]] = {}
            for t in cleaned:
                k = str(t.get("text") or "")
                prev = best_by_text.get(k)
                if prev is None or float(t.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    best_by_text[k] = t
            deduped = list(best_by_text.values())
            deduped_count = max(0, len(cleaned) - len(deduped))
            cleaned = deduped

        out = {
            "page_id": page_id,
            "tokens": cleaned,
            "raw_count": len(raw.get("tokens", [])),
            "clean_count": len(cleaned),
            "deduped_count": int(deduped_count),
        }
        write_json(clean_path, out)
        return out
