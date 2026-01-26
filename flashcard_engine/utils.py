from __future__ import annotations

import json
import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_filename_token(text: str, max_len: int = 50) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "token"


def slugify(text: str, max_len: int = 50, *, add_hash: bool = True) -> str:
    """Safe filename slug.

    Notes:
    - Keeps output ASCII-safe for Windows paths.
    - Optionally appends a short sha1 suffix to reduce collisions (recommended).
    """
    base = safe_filename_token(text, max_len=max_len)
    if not add_hash:
        return base
    h = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    # keep total length bounded
    if len(base) + 1 + len(h) > max_len:
        base = base[: max(1, max_len - 1 - len(h))]
    return f"{base}_{h}"


def stable_card_id(source_ref: str, page_id: str, text: str, bbox_xyxy: Any) -> str:
    """Stable id: sha1(source_ref + page_id + text + bbox_xyxy).

    bbox_xyxy is normalized into 'x0,y0,x1,y1' string when possible.
    """
    bbox_str = ""
    try:
        if bbox_xyxy is not None:
            x0, y0, x1, y1 = bbox_xyxy
            bbox_str = f"{int(x0)},{int(y0)},{int(x1)},{int(y1)}"
    except Exception:
        bbox_str = ""

    payload = f"{source_ref}|{page_id}|{text}|{bbox_str}".encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def clamp_bbox_xyxy(bbox_xyxy: Any, *, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Clamp bbox to image bounds. Returns None if bbox is invalid."""
    try:
        x0, y0, x1, y1 = bbox_xyxy
        x0 = clamp_int(x0, 0, max(0, w - 1))
        y0 = clamp_int(y0, 0, max(0, h - 1))
        x1 = clamp_int(x1, 1, max(1, w))
        y1 = clamp_int(y1, 1, max(1, h))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return x0, y0, x1, y1
    except Exception:
        return None


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_job_relative_path(job_dir: str | Path, rel_path: str | Path, *, field: str = "path") -> Path:
    """Validate that rel_path is a safe, job-relative path and return its absolute Path.

    Security rules:
    - Reject absolute/rooted/drive paths
    - Reject any '..' path segments
    - After resolving, the path must remain within job_dir
    """
    job_dir_p = Path(job_dir)
    base = job_dir_p.resolve()

    if isinstance(rel_path, Path):
        rel_str = str(rel_path)
    else:
        rel_str = str(rel_path or "")

    rel_str = rel_str.strip().replace("\\", "/")
    if not rel_str:
        raise ValueError(f"unsafe_{field}: empty")

    p = Path(rel_str)

    # Reject absolute paths and drive/UNC paths on Windows.
    if p.is_absolute() or p.drive:
        raise ValueError(f"unsafe_{field}: absolute_or_drive_path: {rel_str}")

    parts = [part for part in p.parts if part not in (".", "")]
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe_{field}: parent_traversal: {rel_str}")

    abs_p = (base / p).resolve()
    if abs_p != base and base not in abs_p.parents:
        raise ValueError(f"unsafe_{field}: escapes_job_dir: {rel_str}")
    return abs_p


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def is_numeric_only(token: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+", token.strip()))
