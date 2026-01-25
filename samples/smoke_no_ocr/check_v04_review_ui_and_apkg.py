from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
import json
from pathlib import Path


def run_cmd(args: list[str], *, cwd: Path) -> str:
    p = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"command_failed rc={p.returncode}: {' '.join(args)}\n{out}")
    return p.stdout.strip()


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    workspace = repo / "workspace" / "smoke_v04_review_ui_apkg"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # 1) deterministic input image
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

    # v0.4.1 determinism: cards must carry explicit ordering fields.
    result = json.loads((job_path / "result.json").read_text(encoding="utf-8"))
    cards = result.get("cards", [])
    if not isinstance(cards, list) or not cards:
        raise RuntimeError("cards_missing_or_empty")
    for c in cards:
        if not isinstance(c, dict):
            raise RuntimeError("card_not_object")
        if "source_page_id" not in c:
            raise RuntimeError("card_missing_source_page_id")
        if "token_index" not in c:
            raise RuntimeError("card_missing_token_index")
    cards_sorted = sorted(cards, key=lambda c: (str(c.get("source_page_id") or ""), int(c.get("token_index") or 0)))
    print("ORDER_PROOF(first5)", [(c.get("source_page_id"), c.get("token_index"), c.get("word")) for c in cards_sorted[:5]])

    # 3) validate before review
    run_cmd([sys.executable, "-m", "flashcard_engine.cli", "validate", "--job-dir", str(job_path)], cwd=repo)

    # 4) generate review-ui (static html + stub feedback file)
    out_ui = run_cmd([sys.executable, "-m", "flashcard_engine.cli", "review-ui", "--job-dir", str(job_path)], cwd=repo)
    review_html = job_path / "review.html"
    feedback_json = job_path / "review_feedback.json"
    if not review_html.exists():
        raise RuntimeError(f"review_html_missing: {review_html}\n{out_ui}")
    if not feedback_json.exists():
        raise RuntimeError(f"feedback_stub_missing: {feedback_json}\n{out_ui}")

    # 5) simulate human actions by writing compatible feedback
    shutil.copyfile(repo / "samples" / "smoke_no_ocr" / "review_feedback.example.json", feedback_json)

    # 6) apply-review
    run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "apply-review",
            "--job-dir",
            str(job_path),
            "--feedback",
            str(feedback_json),
        ],
        cwd=repo,
    )

    # 7) export apkg
    apkg_out = workspace / "smoke_no_ocr.apkg"
    run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "export",
            "--job-dir",
            str(job_path),
            "--format",
            "apkg",
            "--out",
            str(apkg_out),
            "--deck-name",
            "smoke_no_ocr",
            "--tags",
            "smoke",
        ],
        cwd=repo,
    )

    if not apkg_out.exists() or apkg_out.stat().st_size <= 0:
        raise RuntimeError(f"apkg_missing_or_empty: {apkg_out}")

    # Basic sanity: apkg is a zip with collection
    with zipfile.ZipFile(apkg_out, "r") as z:
        names = set(z.namelist())
        if "collection.anki2" not in names:
            raise RuntimeError(f"apkg_missing_collection: {sorted(list(names))[:20]}")

    # 8) validate after review/export
    run_cmd([sys.executable, "-m", "flashcard_engine.cli", "validate", "--job-dir", str(job_path)], cwd=repo)
    run_cmd(
        [
            sys.executable,
            "-m",
            "flashcard_engine.cli",
            "validate",
            "--job-dir",
            str(job_path),
            "--apkg",
            str(apkg_out),
        ],
        cwd=repo,
    )

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
