from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import append_jsonl, ensure_dir, utc_now_iso, write_json


@dataclass
class JobPaths:
    job_dir: Path
    input_dir: Path
    pages_dir: Path
    crops_dir: Path
    stage_ocr_dir: Path
    stage_layout_dir: Path
    stage_segment_dir: Path
    result_json: Path
    review_json: Path
    metrics_json: Path
    errors_jsonl: Path


def create_job_dirs(workspace: str | Path, job_id: str) -> JobPaths:
    ws = Path(workspace)
    job_dir = ws / "jobs" / job_id

    input_dir = job_dir / "input"
    pages_dir = job_dir / "pages"
    crops_dir = pages_dir / "crops"

    stage_dir = job_dir / "stage"
    stage_ocr_dir = stage_dir / "ocr"
    stage_layout_dir = stage_dir / "layout"
    stage_segment_dir = stage_dir / "segment"

    for p in [
        input_dir,
        pages_dir,
        crops_dir,
        stage_ocr_dir,
        stage_layout_dir,
        stage_segment_dir,
    ]:
        ensure_dir(p)

    return JobPaths(
        job_dir=job_dir,
        input_dir=input_dir,
        pages_dir=pages_dir,
        crops_dir=crops_dir,
        stage_ocr_dir=stage_ocr_dir,
        stage_layout_dir=stage_layout_dir,
        stage_segment_dir=stage_segment_dir,
        result_json=job_dir / "result.json",
        review_json=job_dir / "review_queue.json",
        metrics_json=job_dir / "metrics.json",
        errors_jsonl=job_dir / "errors.jsonl",
    )


def new_job_id() -> str:
    return str(uuid.uuid4())


def record_error(paths: JobPaths, page_id: str, stage: str, message: str) -> None:
    append_jsonl(paths.errors_jsonl, {"page_id": page_id, "stage": stage, "message": message})


def init_job_outputs(paths: JobPaths) -> None:
    # Always create output files, even if empty.
    write_json(paths.result_json, {"job": {}, "cards": []})
    write_json(paths.review_json, {"items": []})
    write_json(
        paths.metrics_json,
        {
            "created_at": utc_now_iso(),
            "finished": False,
            "completed_at": None,
            "pages_total": 0,
            "pages_processed": 0,
            "cards_total": 0,
            "review_items_total": 0,
            "segment_success_count": 0,
            "segment_fail_count": 0,
            "ocr_empty_count": 0,
            # v0.2
            "pages_single_word": 0,
            "pages_multi_word": 0,
            "multiword_tokens": 0,
            "multiword_crops_written": 0,
            "multiword_crop_failures": 0,
            "multiword_crops_gated_small": 0,
            "multiword_crops_gated_ratio": 0,
            "deduped_tokens": 0,
            "review_items": 0,
        },
    )
    # errors.jsonl: create empty file
    paths.errors_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths.errors_jsonl.touch(exist_ok=True)


def snapshot_input(paths: JobPaths, input_path: str | Path, input_type: str) -> None:
    src = Path(input_path)
    if input_type == "pdf" and src.is_file():
        shutil.copy2(src, paths.input_dir / src.name)
    else:
        # For images folder or other: store a lightweight manifest.
        write_json(paths.input_dir / "manifest.json", {"type": input_type, "path": str(src.resolve())})
