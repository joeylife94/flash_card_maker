from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .job import create_job_dirs, init_job_outputs, new_job_id, snapshot_input
from .pipeline import EnginePipeline, RunOptions


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
    )

    EnginePipeline(paths=paths, cfg=cfg, opts=opts).run(job_id=job_id)
    print(str(paths.job_dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
