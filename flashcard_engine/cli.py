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
from .pair_extractor import apply_pair_feedback, LearningCache
from .pair_flashcard_builder import build_flashcards_from_pairs, PairFlashcardBuilder


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
    run.add_argument(
        "--debug-panels",
        action="store_true",
        default=False,
        help="Enable panel debug pack generation (annotated images, panel crops, diagnostics JSON)",
    )
    run.add_argument(
        "--mode",
        default="flashcard",
        choices=["flashcard", "pair"],
        help="Pipeline mode: 'flashcard' (default) or 'pair' for picture/text separation",
    )
    run.add_argument(
        "--sam",
        action="store_true",
        default=False,
        help="Use SAM-based picture detection (pair mode only)",
    )
    run.add_argument(
        "--timeline-dir",
        action="store_true",
        default=False,
        help="Use timeline-based job directory format (YYYY-MM-DD/HH-MM-SS__<shortid>)",
    )
    run.add_argument(
        "--no-learning",
        action="store_true",
        default=False,
        help="Disable learning cache (pair mode only)",
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

    # Pair mode feedback command
    pf = sub.add_parser("apply-pair-feedback", help="Apply human feedback to pair extraction results")
    pf.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    pf.add_argument("--feedback", required=True, help="Path to pair_feedback.json")
    pf.add_argument("--workspace", default="./workspace", help="Workspace root (for learning cache)")
    pf.add_argument(
        "--no-learning",
        action="store_true",
        default=False,
        help="Disable learning cache updates",
    )
    
    # Learning stats command
    ls = sub.add_parser("learning-stats", help="Show learning cache statistics")
    ls.add_argument("--workspace", default="./workspace", help="Workspace root")

    # Build flashcards from pairs command
    bf = sub.add_parser("build-flashcards", help="Build flashcards from pair extraction results")
    bf.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    bf.add_argument("--source", default="", help="Source name for card references")
    bf.add_argument(
        "--reverse",
        action="store_true",
        default=False,
        help="Generate reverse cards (Text → Picture) in addition to forward cards",
    )

    # Quick extract command (single image/folder → pairs → flashcards)
    qe = sub.add_parser("extract", help="Quick extract: image/folder → pairs → flashcards (all-in-one)")
    qe.add_argument("--input", required=True, help="Input path (image file or folder)")
    qe.add_argument("--workspace", default="./workspace", help="Workspace root")
    qe.add_argument("--lang", default="en", help="OCR language (e.g., 'en', 'ko', 'en,ko')")
    qe.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device for SAM")
    qe.add_argument("--export", default=None, help="Optional: export path (.apkg or .csv)")
    qe.add_argument("--deck-name", default=None, help="Deck name for Anki export")
    # Picture filtering options
    qe.add_argument("--min-edge-density", type=float, default=0.05, 
                    help="Minimum edge density for picture detection (default: 0.05, higher=stricter)")
    qe.add_argument("--max-area-ratio", type=float, default=0.15,
                    help="Maximum area ratio for pictures (default: 0.15, lower=reject large regions)")
    qe.add_argument("--strict", action="store_true", default=False,
                    help="Use strict filtering (edge_density=0.1, area_ratio=0.10)")

    # Generate interactive review HTML
    grh = sub.add_parser("generate-review-html", help="Generate interactive HTML for pair review")
    grh.add_argument("--job-dir", required=True, help="Job directory (workspace/jobs/<job_id>)")
    grh.add_argument("--output", default=None, help="Output HTML path (default: job_dir/review.html)")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    # Support timeline-based job IDs for pair mode or when explicitly requested
    use_timeline = getattr(args, 'timeline_dir', False) or getattr(args, 'mode', 'flashcard') == 'pair'
    job_id = new_job_id(use_timeline=use_timeline)
    paths = create_job_dirs(args.workspace, job_id, use_timeline=use_timeline)
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
        debug_panels=getattr(args, 'debug_panels', False),
        mode=getattr(args, 'mode', 'flashcard'),
        learning_enabled=not getattr(args, 'no_learning', False),
        use_sam=getattr(args, 'sam', False),
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

    def categorize_error(msg: str) -> str:
        s = (msg or "").strip()
        sl = s.lower()
        if sl.startswith("apkg_") or "apkg" in sl:
            return "APKG"
        if "unsafe" in sl or "path traversal" in sl:
            return "SECURITY"
        if "requires needs_review=true" in sl or "segment_failed" in sl:
            return "LIFECYCLE"
        if "determin" in sl or "canonical" in sl or ("token_index" in sl and "order" in sl):
            return "DETERMINISM"
        return "CONTRACT"

    if errors:
        order = ["CONTRACT", "SECURITY", "DETERMINISM", "APKG", "LIFECYCLE"]
        buckets: dict[str, list[str]] = {k: [] for k in order}
        for m in errors:
            cat = categorize_error(str(m))
            buckets.setdefault(cat, []).append(str(m))
        print("ERRORS")
        for cat in order:
            msgs = buckets.get(cat) or []
            if not msgs:
                continue
            print(f"[{cat}] ({len(msgs)})")
            for m in msgs:
                print(f"- {m}")
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


def cmd_apply_pair_feedback(args: argparse.Namespace) -> int:
    """Apply human feedback to pair extraction results."""
    try:
        learning_cache = None
        if not getattr(args, 'no_learning', False):
            learning_cache = LearningCache(args.workspace)
        
        stats = apply_pair_feedback(
            job_dir=args.job_dir,
            feedback_path=args.feedback,
            learning_cache=learning_cache,
        )
        
        print(f"processed={stats.get('processed', 0)} approved={stats.get('approved', 0)} rejected={stats.get('rejected', 0)} edited={stats.get('edited', 0)} learning_records={stats.get('learning_records', 0)}")
        return 0
    except Exception as e:
        print(f"apply_pair_feedback_failed: {e}")
        return 1


def cmd_learning_stats(args: argparse.Namespace) -> int:
    """Show learning cache statistics."""
    try:
        cache = LearningCache(args.workspace)
        stats = cache.get_stats()
        
        print(f"total_records={stats.get('total_records', 0)}")
        print(f"caption_corrections={stats.get('caption_corrections', 0)}")
        print(f"cached_parameters={stats.get('cached_parameters', 0)}")
        print(f"blank_threshold={stats.get('blank_threshold', 0.7):.3f}")
        return 0
    except Exception as e:
        print(f"learning_stats_failed: {e}")
        return 1


def cmd_build_flashcards(args: argparse.Namespace) -> int:
    """Build flashcards from pair extraction results."""
    try:
        from .utils import load_json
        
        builder = PairFlashcardBuilder(args.job_dir, getattr(args, 'source', ''))
        cards, stats = builder.build_from_result_pairs(
            include_reverse=getattr(args, 'reverse', False),
        )
        
        # Load existing job meta or create new one
        result_pairs_path = Path(args.job_dir) / "result_pairs.json"
        if result_pairs_path.exists():
            data = load_json(result_pairs_path)
            job_meta = {
                "job_id": data.get("job_id", ""),
                "mode": "pair_flashcard",
                "source": getattr(args, 'source', ''),
                "created_at": data.get("created_at", ""),
            }
        else:
            job_meta = {"mode": "pair_flashcard"}
        
        # Write result.json
        result_path = builder.write_result_json(cards, job_meta)
        
        print(f"pairs_total={stats.pairs_total}")
        print(f"cards_created={stats.cards_created}")
        print(f"forward_cards={stats.forward_cards}")
        print(f"reverse_cards={stats.reverse_cards}")
        print(f"cards_needing_review={stats.cards_needing_review}")
        print(f"cards_no_text={stats.cards_no_text}")
        print(f"result_path={result_path}")
        return 0
    except Exception as e:
        print(f"build_flashcards_failed: {e}")
        return 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Quick extract: image/folder → pairs → flashcards (all-in-one)."""
    try:
        from pathlib import Path
        from .sam_pair_extractor import extract_pairs_from_image, extract_pairs_from_folder, PairConfig
        
        input_path = Path(args.input)
        workspace = Path(args.workspace)
        
        # Configure filtering based on args
        config = PairConfig()
        if args.strict:
            config.min_edge_density = 0.10
            config.max_mask_area_ratio = 0.10
            print("[INFO] Using strict filtering mode")
        else:
            config.min_edge_density = args.min_edge_density
            config.max_mask_area_ratio = args.max_area_ratio
        
        # Step 1: Extract pairs
        print(f"[1/3] Extracting pairs from {input_path}...")
        print(f"  → Config: edge_density={config.min_edge_density}, max_area={config.max_mask_area_ratio}")
        
        if input_path.is_file():
            job_summary = extract_pairs_from_image(
                image_path=input_path,
                workspace=workspace,
                lang=args.lang,
                device=args.device,
                config=config,
            )
        elif input_path.is_dir():
            job_summary = extract_pairs_from_folder(
                folder_path=input_path,
                workspace=workspace,
                lang=args.lang,
                device=args.device,
                config=config,
            )
        else:
            print(f"ERROR: Input path does not exist: {input_path}")
            return 1
        
        job_id = job_summary.get("job_id", "")
        output_dir = workspace / "output" / f"job_{job_id}"
        
        print(f"  → Job ID: {job_id}")
        print(f"  → Pairs extracted: {job_summary.get('total_pairs', 0)}")
        print(f"  → Needing review: {job_summary.get('total_needing_review', 0)}")
        
        # Step 2: Build flashcards (create result_pairs.json compatible output)
        print(f"\n[2/3] Building flashcards...")
        
        # The SAM extractor already creates the output in the right format
        # We just need to report
        total_pairs = job_summary.get('total_pairs', 0)
        print(f"  → Flashcards ready: {total_pairs}")
        
        # Step 3: Export (if requested)
        if args.export:
            print(f"\n[3/3] Exporting to {args.export}...")
            export_path = Path(args.export)
            
            # Create a temporary job structure for the exporter
            # The SAM output is in workspace/output/job_<id>/
            # We need to create result.json for the standard exporter
            
            job_summary_path = output_dir / "job_summary.json"
            if job_summary_path.exists():
                from .utils import load_json, write_json
                
                summary_data = load_json(job_summary_path)
                
                # Build cards from pages data
                cards = []
                for page_data in summary_data.get("pages", []):
                    page_id = page_data.get("page_id", "")
                    for pair in page_data.get("pairs", []):
                        card = {
                            "card_id": pair.get("pair_id", ""),
                            "page_id": page_id,
                            "source_page_id": page_id,
                            "token_index": pair.get("order_index", 0),
                            "layout_type": "pair",
                            "word": pair.get("caption_text", ""),
                            "bbox_xyxy": pair.get("picture_bbox"),
                            "method": "pair_sam",
                            "front_image_path": pair.get("picture_path", ""),
                            "source_ref": page_id,
                            "confidence": pair.get("confidence", 0.0),
                            "needs_review": pair.get("needs_review", False),
                            "status": "active" if not pair.get("needs_review") else "review",
                            "created_at": summary_data.get("created_at", ""),
                            "updated_at": summary_data.get("created_at", ""),
                            "reasons": pair.get("reasons", []),
                        }
                        cards.append(card)
                
                # Write temporary result.json
                result_json = {
                    "job": {
                        "job_id": job_id,
                        "mode": "pair_sam",
                        "source": input_path.stem,
                        "created_at": summary_data.get("created_at", ""),
                    },
                    "cards": cards,
                }
                result_path = output_dir / "result.json"
                write_json(result_path, result_json)
                
                # Export
                if export_path.suffix.lower() == ".apkg":
                    stats = export_apkg(
                        job_dir=output_dir,
                        out_path=export_path,
                        deck_name=args.deck_name or input_path.stem,
                    )
                    print(f"  → Exported {stats.cards_exported} cards to {export_path}")
                elif export_path.suffix.lower() == ".csv":
                    stats = export_csv(
                        job_dir=output_dir,
                        out_path=export_path,
                    )
                    print(f"  → Exported {stats.cards_exported} cards to {export_path}")
                else:
                    print(f"  → Unknown export format: {export_path.suffix}")
        else:
            print(f"\n[3/3] Skipping export (use --export to save)")
        
        print(f"\n✅ Complete! Output directory: {output_dir}")
        return 0
        
    except Exception as e:
        import traceback
        print(f"extract_failed: {e}")
        traceback.print_exc()
        return 1


def cmd_generate_review_html(args: argparse.Namespace) -> int:
    """Generate interactive HTML for pair review."""
    try:
        from pathlib import Path
        from .review_ui_generator import generate_review_from_result_pairs
        
        job_dir = Path(args.job_dir)
        
        # Find result_pairs.json
        result_pairs_path = job_dir / "result_pairs.json"
        if not result_pairs_path.exists():
            # Try in stage/pair directory
            result_pairs_path = job_dir / "stage" / "pair" / "result_pairs.json"
        
        if not result_pairs_path.exists():
            print(f"Error: result_pairs.json not found in {job_dir}")
            return 1
        
        # Determine stage directory for images
        stage_dir = job_dir / "stage" / "pair"
        if not stage_dir.exists():
            stage_dir = job_dir
        
        # Output path
        output_html = args.output if args.output else str(job_dir / "review.html")
        
        output_path = generate_review_from_result_pairs(
            result_pairs_path=result_pairs_path,
            stage_dir=stage_dir,
            output_html_path=output_html,
        )
        
        print(f"Generated review HTML: {output_path}")
        return 0
    except Exception as e:
        import traceback
        print(f"generate_review_html_failed: {e}")
        traceback.print_exc()
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
    
    if args.command == "apply-pair-feedback":
        return cmd_apply_pair_feedback(args)
    
    if args.command == "learning-stats":
        return cmd_learning_stats(args)
    
    if args.command == "build-flashcards":
        return cmd_build_flashcards(args)
    
    if args.command == "extract":
        return cmd_extract(args)
    
    if args.command == "generate-review-html":
        return cmd_generate_review_html(args)

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
