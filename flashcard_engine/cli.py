from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config
from .job import create_job_dirs, init_job_outputs, new_job_id, snapshot_input
from .pipeline import EnginePipeline, RunOptions
from .utils import load_json
from .exporter import export_csv
from .review import apply_review_feedback


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

    export = sub.add_parser("export", help="Export flashcards from a completed job")
    export.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    export.add_argument("--format", required=True, choices=["csv"], help="Export format")
    export.add_argument("--out", required=True, help="Output file path")
    export.add_argument("--include-review", action="store_true", help="Include cards still in review")

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


def _assert_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path}")


def _validate_image_refs(job_dir: Path, obj: Any, errors: list[str]) -> int:
    # result.json cards
    cards = (obj or {}).get("cards", []) if isinstance(obj, dict) else []
    missing = 0
    for c in cards:
        if not isinstance(c, dict):
            continue
        rel = c.get("front_image_path")
        if not rel:
            continue
        p = job_dir / str(rel)
        if not p.exists():
            errors.append(f"missing card image: {rel}")
            missing += 1
    return missing


def _is_sha1_hex(s: Any) -> bool:
    if not isinstance(s, str) or len(s) != 40:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False


def _validate_cards_schema(obj: Any, errors: list[str]) -> int:
    invalid = 0
    cards = (obj or {}).get("cards", []) if isinstance(obj, dict) else []
    for idx, c in enumerate(cards):
        if not isinstance(c, dict):
            errors.append(f"invalid card[{idx}]: not an object")
            invalid += 1
            continue

        # Required keys must exist (may be None for bbox_xyxy).
        for k in ("card_id", "method", "bbox_xyxy", "front_image_path", "page_id", "source_ref"):
            if k not in c:
                errors.append(f"invalid card[{idx}]: missing field {k}")
                invalid += 1
                continue

        if not _is_sha1_hex(c.get("card_id")):
            errors.append(f"invalid card[{idx}]: card_id not sha1 hex")
            invalid += 1

        method = c.get("method")
        if method not in ("page", "bbox_crop", "segmenter"):
            errors.append(f"invalid card[{idx}]: method={method}")
            invalid += 1

        bbox = c.get("bbox_xyxy")
        if bbox is not None:
            ok = (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and all(isinstance(v, int) for v in bbox)
            )
            if not ok:
                errors.append(f"invalid card[{idx}]: bbox_xyxy must be [int,int,int,int] or null")
                invalid += 1

    return invalid


def _validate_review_refs(job_dir: Path, obj: Any, errors: list[str]) -> int:
    items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
    missing = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        rel = it.get("front_image_path")
        if not rel:
            continue
        p = job_dir / str(rel)
        if not p.exists():
            errors.append(f"missing review image: {rel}")
            missing += 1
    return missing


def _validate_review_schema(obj: Any, errors: list[str]) -> int:
    invalid = 0
    items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"invalid review[{idx}]: not an object")
            invalid += 1
            continue

        for k in ("card_id", "review_reason", "page_id", "source_ref"):
            if k not in it:
                errors.append(f"invalid review[{idx}]: missing field {k}")
                invalid += 1
                continue

        if not _is_sha1_hex(it.get("card_id")):
            errors.append(f"invalid review[{idx}]: card_id not sha1 hex")
            invalid += 1

        bbox = it.get("bbox_xyxy")
        if bbox is not None:
            ok = (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and all(isinstance(v, int) for v in bbox)
            )
            if not ok:
                errors.append(f"invalid review[{idx}]: bbox_xyxy must be [int,int,int,int] or null")
                invalid += 1

    return invalid


def cmd_validate(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir)
    errors: list[str] = []

    missing_contract_files = 0
    missing_images = 0
    invalid_cards = 0
    invalid_review_items = 0

    # Output Contract (must always exist)
    for f in ("result.json", "review_queue.json", "metrics.json", "errors.jsonl"):
        p = job_dir / f
        if not p.exists():
            missing_contract_files += 1
            errors.append(f"missing: {p}")

    # Validate referenced files
    try:
        result = load_json(job_dir / "result.json")
        missing_images += _validate_image_refs(job_dir, result, errors)
        invalid_cards += _validate_cards_schema(result, errors)
    except Exception as e:
        errors.append(f"failed to read result.json: {e}")
        invalid_cards += 1

    try:
        review = load_json(job_dir / "review_queue.json")
        missing_images += _validate_review_refs(job_dir, review, errors)
        invalid_review_items += _validate_review_schema(review, errors)
    except Exception as e:
        errors.append(f"failed to read review_queue.json: {e}")
        invalid_review_items += 1

    print(f"missing_contract_files={missing_contract_files}")
    print(f"missing_images={missing_images}")
    print(f"invalid_cards={invalid_cards}")
    print(f"invalid_review_items={invalid_review_items}")

    if errors:
        for m in errors:
            print(m)
        return 1

    print("OK")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    if args.format != "csv":
        raise SystemExit(2)
    try:
        stats = export_csv(job_dir=args.job_dir, out_path=args.out, include_review=bool(args.include_review))
        print(f"exported={stats.cards_exported} skipped_review={stats.cards_skipped_review} skipped_missing_image={stats.cards_skipped_missing_image}")
        return 0
    except Exception as e:
        print(f"export_failed: {e}")
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

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
