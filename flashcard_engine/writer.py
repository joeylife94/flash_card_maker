from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .job import JobPaths
from .utils import write_json


@dataclass
class JobWriter:
    paths: JobPaths

    def write_final(
        self,
        job_meta: dict[str, Any],
        cards: list[dict[str, Any]],
        review_items: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        write_json(self.paths.result_json, {"job": job_meta, "cards": cards})
        write_json(self.paths.review_json, {"items": review_items})
        write_json(self.paths.metrics_json, metrics)
