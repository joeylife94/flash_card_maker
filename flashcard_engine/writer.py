from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .job import JobPaths
from .utils import utc_now_iso, write_json


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
        now = utc_now_iso()

        # Mark completion only when final outputs are successfully written.
        metrics_out = dict(metrics)
        metrics_out["finished"] = True
        metrics_out["completed_at"] = now

        job_out = dict(job_meta)
        job_out["finished"] = True
        job_out["completed_at"] = now

        write_json(self.paths.result_json, {"job": job_out, "cards": cards})
        write_json(self.paths.review_json, {"items": review_items})
        write_json(self.paths.metrics_json, metrics_out)
