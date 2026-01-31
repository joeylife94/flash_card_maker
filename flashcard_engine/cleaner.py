from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .job import JobPaths
from .utils import compile_patterns, is_numeric_only, write_json


@dataclass
class TextCleaner:
    paths: JobPaths
    cleanup_cfg: dict[str, Any]
    
    def __post_init__(self):
        """Compile patterns once on initialization."""
        self._drop_patterns = compile_patterns(list(self.cleanup_cfg.get("drop_patterns", [])))
        self._suspicious_patterns = compile_patterns(list(self.cleanup_cfg.get("suspicious_token_patterns", [])))

    def _is_likely_garbage(self, text: str) -> bool:
        """무의미한 텍스트 감지."""
        # 반복되는 문자 (aaa, 111 등)
        if len(set(text)) == 1 and len(text) > 2:
            return True
        
        # 특수문자가 50% 이상
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0 and special_count / len(text) > 0.5:
            return True
        
        # 연속된 특수문자 (!!!, ???)
        if re.search(r'[^\w\s]{3,}', text):
            return True
        
        return False
    
    def _normalize_text(self, text: str, lowercase: bool = True) -> str:
        """텍스트 정규화."""
        # 공백 정리
        text = ' '.join(text.split())
        
        # 일반적인 OCR 오류 수정
        replacements = {
            '‘': "'",  # 좌측 따옴표
            '’': "'",  # 우측 따옴표
            '“': '"',  # 좌측 큰따옴표
            '”': '"',  # 우측 큰따옴표
            '–': '-',  # en dash
            '—': '-',  # em dash
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        if lowercase:
            text = text.lower()
        
        return text
    
    def _calculate_quality_score(self, token: dict[str, Any]) -> float:
        """토큰 품질 점수 계산."""
        confidence = float(token.get("confidence", 0.0))
        text = str(token.get("text", ""))
        
        # 기본 점수는 confidence
        score = confidence * 0.6
        
        # 길이에 따른 보너스 (3-15자가 최적)
        length = len(text)
        if 3 <= length <= 15:
            score += 0.2
        elif length < 3:
            score += 0.1
        
        # 알파벳 비율
        alpha_count = sum(1 for c in text if c.isalpha())
        if length > 0:
            alpha_ratio = alpha_count / length
            score += alpha_ratio * 0.2
        
        return min(1.0, score)
    
    def clean(self, page_id: str, raw: dict[str, Any]) -> dict[str, Any]:
        clean_path = self.paths.stage_ocr_dir / f"{page_id}_clean.json"

        lowercase = bool(self.cleanup_cfg.get("lowercase", True))
        remove_numeric_only = bool(self.cleanup_cfg.get("remove_numeric_only", True))
        min_len = int(self.cleanup_cfg.get("min_token_length", 3))
        allow_short = set([t.lower() for t in self.cleanup_cfg.get("allow_short_tokens", [])])
        max_tokens = int(self.cleanup_cfg.get("max_tokens_per_page", 200))
        enable_dedupe = bool(self.cleanup_cfg.get("dedupe_enabled", False))
        drop_punct_only = bool(self.cleanup_cfg.get("drop_punctuation_only", True))
        drop_suspicious = bool(self.cleanup_cfg.get("drop_suspicious_tokens", False))
        min_confidence = float(self.cleanup_cfg.get("min_confidence", 0.3))
        
        # 품질 기반 정렬 활성화
        enable_quality_sort = bool(self.cleanup_cfg.get("quality_sort_enabled", False))

        def _token_sort_key(t: dict[str, Any]) -> tuple:
            bbox = t.get("bbox_xyxy")
            text = str(t.get("text") or "")
            try:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    return (0, y0, x0, y1, x1, text)
            except Exception:
                pass
            return (1, 10**9, 10**9, 10**9, 10**9, text)

        cleaned: list[dict[str, Any]] = []
        for t in raw.get("tokens", []):
            # 기본 정규화
            text = self._normalize_text(str(t.get("text") or "").strip(), lowercase)
            
            if not text:
                continue
            
            # 신뢰도 필터
            confidence = float(t.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            # 쓰레기 필터
            if self._is_likely_garbage(text):
                continue
            
            # 기존 필터들
            if drop_punct_only and not any(ch.isalnum() for ch in text):
                continue

            if remove_numeric_only and is_numeric_only(text):
                continue
                
            if any(p.fullmatch(text) for p in self._drop_patterns):
                continue

            if drop_suspicious and any(p.search(text) for p in self._suspicious_patterns):
                continue

            if len(text) < min_len and text not in allow_short:
                continue
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score({"text": text, "confidence": confidence})
            
            cleaned.append({
                "text": text, 
                "confidence": confidence, 
                "bbox_xyxy": t.get("bbox_xyxy"),
                "quality_score": quality_score,
            })
            if len(cleaned) >= max_tokens:
                break

        # 품질 기반 정렬 또는 위치 기반 정렬
        if enable_quality_sort:
            # 품질 점수로 정렬 (높은 순)
            cleaned.sort(key=lambda t: t.get("quality_score", 0.0), reverse=True)
        else:
            # 기존 위치 기반 정렬
            cleaned.sort(key=_token_sort_key)

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

            # Re-apply canonical ordering after dedupe.
            cleaned.sort(key=_token_sort_key)

        out = {
            "page_id": page_id,
            "tokens": cleaned,
            "raw_count": len(raw.get("tokens", [])),
            "clean_count": len(cleaned),
            "deduped_count": int(deduped_count),
        }
        write_json(clean_path, out)
        return out
