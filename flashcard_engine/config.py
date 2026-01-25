from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json


@dataclass(frozen=True)
class EngineConfig:
    cleanup: dict[str, Any]
    segment: dict[str, Any]
    layout: dict[str, Any]
    confidence: dict[str, Any]
    crop: dict[str, Any]
    review: dict[str, Any]


def load_config(config_path: str | Path) -> EngineConfig:
    data = load_json(config_path)
    return EngineConfig(
        cleanup=data.get("cleanup", {}),
        segment=data.get("segment", {}),
        layout=data.get("layout", {}),
        confidence=data.get("confidence", {}),
        crop=data.get("crop", {}),
        review=data.get("review", {}),
    )
