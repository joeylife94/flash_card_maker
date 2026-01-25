from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config
from .job import create_job_dirs, init_job_outputs, new_job_id, snapshot_input
from .pipeline import EnginePipeline, RunOptions
from .validator import validate_apkg, validate_job_dir
from .exporter import export_csv
from .exporters.apkg import export_apkg
from .review import apply_review_feedback
from .review_ui import generate_review_ui


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flashcard_engine")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run flashcard material production job")
    run.add_argument("--input", required=True, help="Input path (pdf file or images folder)")
    run.add_argument("--type", required=True, choices=["pdf", "images"], help="Input type")
    run.add_argument("--lang", required=True, help="OCR language (PaddleOCR lang code, e.g. en)")
    run.add_argument("--workspace", default="./workspace", help="Workspace root")
    run.add_argument("--source", required=True, help="Source name (e.g. BookName)")
    run.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering (pdf only)")
    run.add_argument("--min-confidence", type=float, default=0.7)
    run.add_argument("--segmenter", default="off", choices=["off", "mobilesam", "fastsam"])
    run.add_argument("--segmenter-device", default="cpu", choices=["cpu", "cuda", "mps"])
    run.add_argument("--config", default=str(Path("config") / "default.json"), help="Config path")
    run.add_argument(
        "--use-mocked-ocr",
        default=None,
        help="Directory containing mocked cleaned OCR JSON (skips real OCR/cleaner when file exists)",
    )

    validate = sub.add_parser("validate", help="Validate Output Contract + referenced file paths")
    validate.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    validate.add_argument("--apkg", default=None, help="Optional: validate an exported Anki .apkg file")

    export = sub.add_parser("export", help="Export flashcards from a completed job")
    export.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    export.add_argument("--format", required=True, choices=["csv", "apkg"], help="Export format")
    export.add_argument("--out", required=True, help="Output file path")
    export.add_argument("--include-review", action="store_true", help="Include cards still in review")
    export.add_argument("--deck-name", default=None, help="Deck name (apkg only; default=source_ref)")
    export.add_argument("--tags", default=None, help="Comma-separated tags (apkg only)")

    ui = sub.add_parser("review-ui", help="Generate a static HTML review UI (no server)")
    ui.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")

    ar = sub.add_parser("apply-review", help="Apply human review feedback to a job")
    ar.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    ar.add_argument("--feedback", required=True, help="Path to review_feedback.json")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    job_id = new_job_id()
    paths = create_job_dirs(args.workspace, job_id)
    init_job_outputs(paths)
    snapshot_input(paths, args.input, args.type)

    cfg = load_config(args.config)
    opts = RunOptions(
        input_path=args.input,
        input_type=args.type,
        lang=args.lang,
        source=args.source,
        dpi=args.dpi,
        min_confidence=float(args.min_confidence),
        segmenter=args.segmenter,
        segmenter_device=args.segmenter_device,
        mocked_ocr_dir=args.use_mocked_ocr,
    )

    EnginePipeline(paths=paths, cfg=cfg, opts=opts).run(job_id=job_id)
    print(str(paths.job_dir))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ok, summary = validate_job_dir(args.job_dir)
    print(f"missing_contract_files={summary.get('missing_contract_files', 0)}")
    print(f"missing_images={summary.get('missing_images', 0)}")
    print(f"invalid_cards={summary.get('invalid_cards', 0)}")
    print(f"invalid_review_items={summary.get('invalid_review_items', 0)}")

    errors = summary.get("errors") or []

    # Optional APKG integrity validation
    if args.apkg:
        ok_apkg, apkg_summary = validate_apkg(args.job_dir, args.apkg)
        apkg_errors = apkg_summary.get("errors") or []
        for m in apkg_errors:
            errors.append(m)
    if errors:
        for m in errors:
            print(m)
        return 1

    print("OK")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    try:
        if args.format == "csv":
            stats = export_csv(job_dir=args.job_dir, out_path=args.out, include_review=bool(args.include_review))
            print(
                f"exported={stats.cards_exported} skipped_review={stats.cards_skipped_review} skipped_missing_image={stats.cards_skipped_missing_image}"
            )
            return 0

        if args.format == "apkg":
            stats = export_apkg(
                job_dir=args.job_dir,
                out_path=args.out,
                deck_name=args.deck_name,
                tags=args.tags,
            )
            print(
                f"exported={stats.cards_exported} skipped_inactive={stats.cards_skipped_inactive} skipped_missing_image={stats.cards_skipped_missing_image} deck_name={stats.deck_name}"
            )
            return 0

        raise SystemExit(2)
    except Exception as e:
        print(f"export_failed: {e}")
        return 1


def cmd_review_ui(args: argparse.Namespace) -> int:
    try:
        stats = generate_review_ui(job_dir=args.job_dir)
        print(f"wrote_review_ui={stats.html_path} wrote_feedback_stub={stats.feedback_path} items={stats.items}")
        return 0
    except Exception as e:
        print(f"review_ui_failed: {e}")
        return 1


def cmd_apply_review(args: argparse.Namespace) -> int:
    try:
        stats = apply_review_feedback(job_dir=args.job_dir, feedback_path=args.feedback)
        print(f"feedback_items={stats.feedback_items} applied={stats.applied} skipped_unknown_card={stats.skipped_unknown_card} skipped_already_applied={stats.skipped_already_applied}")
        return 0
    except Exception as e:
        print(f"apply_review_failed: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)

    if args.command == "validate":
        return cmd_validate(args)

    if args.command == "export":
        return cmd_export(args)

    if args.command == "apply-review":
        return cmd_apply_review(args)

    if args.command == "review-ui":
        return cmd_review_ui(args)

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
