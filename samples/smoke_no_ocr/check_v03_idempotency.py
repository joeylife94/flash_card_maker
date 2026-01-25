from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(args: list[str], *, cwd: Path) -> str:
    p = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"command_failed rc={p.returncode}: {' '.join(args)}\n{out}")
    return p.stdout.strip()


def parse_kv(line: str) -> dict[str, int]:
    # e.g. "feedback_items=1 applied=1 skipped_unknown_card=0 skipped_already_applied=0"
    parts = line.strip().split()
    out: dict[str, int] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if v.isdigit():
            out[k] = int(v)
    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    workspace = repo / "workspace" / "smoke_v03_idempotency"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # 1) regenerate deterministic image
    run_cmd([sys.executable, str(repo / "samples" / "smoke_no_ocr" / "generate_image.py")], cwd=repo)

    # 2) run pipeline (mocked OCR)
    job_dir = run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "run",
            "--input",
            str(repo / "samples" / "smoke_no_ocr" / "pages"),
            "--type",
            "images",
            "--lang",
            "en",
            "--workspace",
            str(workspace),
            "--source",
            "smoke_no_ocr",
            "--min-confidence",
            "0.99",
            "--segmenter",
            "off",
            "--use-mocked-ocr",
            str(repo / "samples" / "smoke_no_ocr" / "stage" / "ocr"),
        ],
        cwd=repo,
    )

    job_path = Path(job_dir)
    if not job_path.exists():
        raise RuntimeError(f"job_dir_not_found: {job_path}")

    # 3) validate before review
    run_cmd([sys.executable, "-m", "flashcard_engine.cli", "validate", "--job-dir", str(job_path)], cwd=repo)

    # 4) apply-review with edit feedback (expect applied=1)
    out1 = run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "apply-review",
            "--job-dir",
            str(job_path),
            "--feedback",
            str(repo / "samples" / "smoke_no_ocr" / "review_feedback.edit_only.json"),
        ],
        cwd=repo,
    )
    stats1 = parse_kv(out1)
    if stats1.get("applied") != 1:
        raise RuntimeError(f"expected applied=1 on first apply-review, got: {out1}")

    # 5) apply-review again with same feedback (expect applied=0)
    out2 = run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "apply-review",
            "--job-dir",
            str(job_path),
            "--feedback",
            str(repo / "samples" / "smoke_no_ocr" / "review_feedback.edit_only.json"),
        ],
        cwd=repo,
    )
    stats2 = parse_kv(out2)
    if stats2.get("applied") != 0:
        raise RuntimeError(f"expected applied=0 on second apply-review, got: {out2}")

    # 6) export + validate
    csv_out = workspace / "deck.csv"
    run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "export",
            "--job-dir",
            str(job_path),
            "--format",
            "csv",
            "--out",
            str(csv_out),
        ],
        cwd=repo,
    )

    run_cmd([sys.executable, "-m", "flashcard_engine.cli", "validate", "--job-dir", str(job_path)], cwd=repo)

    # sanity: csv has header + at least 1 row
    lines = csv_out.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"csv_too_small lines={len(lines)}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
