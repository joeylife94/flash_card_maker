# flashcard_engine

Local, fail-soft flashcard material-set production engine. Converts PDF or image inputs into structured flashcard assets (images + text) with OCR, layout classification, and optional segmentation.

## Project Overview

A Python CLI tool that processes PDF documents or image folders to generate flashcard materials suitable for Anki or CSV export. The engine is designed with **fail-soft** semantics: errors are logged but processing continues, ensuring output files are always generated.

**Version:** 0.1.0 (per pyproject.toml)

## Key Features

- **Input formats:** PDF files (via PyMuPDF) or image folders (PNG, JPG, etc.)
- **OCR integration:** PaddleOCR for text extraction (optional; fail-soft if unavailable)
- **Layout classification:** Automatic single-word vs multi-word page detection
- **Segmentation:** Optional MobileSAM/FastSAM support for precise image cropping
- **Export formats:** CSV and Anki `.apkg` with embedded media
- **Review workflow:** Static HTML review UI + JSON feedback system
- **Deterministic outputs:** Canonical token ordering for reproducible exports
- **Security:** Path traversal protection on all file references
- **QA Gate:** Explicit validation guarantees with repro commands

## Architecture

### High-level Pipeline

```
Input (PDF/Images)
    ↓
PageProvider (render pages)
    ↓
OCRExtractor (PaddleOCR or mocked)
    ↓
TextCleaner (normalize, dedupe, filter)
    ↓
LayoutClassifier (single_word / multi_word)
    ↓
Segmenter (optional: MobileSAM/FastSAM or bbox fallback)
    ↓
FlashcardBuilder (create cards + review items)
    ↓
JobWriter (write result.json, review_queue.json, metrics.json)
```

### Folder Structure

```
flashcard_engine/
├── __init__.py
├── cli.py              # CLI entry point (run, validate, export, review-ui, apply-review)
├── pipeline.py         # Main pipeline orchestration
├── config.py           # Configuration loading from JSON
├── job.py              # Job directory management and output initialization
├── page_provider.py    # PDF/image iteration and page rendering
├── ocr.py              # PaddleOCR wrapper (fail-soft)
├── cleaner.py          # Text normalization and filtering
├── layout.py           # Layout classification (single_word/multi_word)
├── segmenter.py        # Optional SAM-based segmentation
├── cropper.py          # Bbox-based token cropping for multi_word pages
├── builder.py          # Flashcard and review item construction
├── writer.py           # Final JSON output writer
├── exporter.py         # CSV export
├── exporters/
│   └── apkg.py         # Anki .apkg export (requires genanki)
├── review.py           # Apply review feedback to job
├── review_ui.py        # Generate static HTML review interface
├── validator.py        # Output contract and APKG validation
├── types.py            # Core data types (Page, OCRToken, etc.)
└── utils.py            # Utilities (JSON I/O, path safety, hashing)

config/
└── default.json        # Default configuration

docs/
├── output_contract.md  # Output file schemas and field definitions
├── qa-gate.md          # QA guarantees and repro commands
└── ...

samples/
└── smoke_no_ocr/       # Deterministic smoke test fixtures
```

## Quickstart

### Prerequisites

- Python 3.10+ (recommended: 3.11–3.12)
- Virtual environment recommended

### Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
# Or manually: pip install pillow>=10.0.0 pymupdf>=1.23.0 genanki==0.13.1

# (Optional) Install OCR support
pip install paddleocr paddlepaddle
```

### Minimal Run

```powershell
# From PDF
python -m flashcard_engine.cli run `
  --input .\samples\book.pdf `
  --type pdf `
  --lang en `
  --workspace .\workspace `
  --source "BookName"

# From image folder
python -m flashcard_engine.cli run `
  --input .\samples\images `
  --type images `
  --lang en `
  --workspace .\workspace `
  --source "MyDeck"
```

The CLI prints the job directory path on success (e.g., `.\workspace\jobs\<uuid>`).

## Configuration

Configuration file: [config/default.json](config/default.json)

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `cleanup` | `lowercase` | Normalize tokens to lowercase | `true` |
| `cleanup` | `remove_numeric_only` | Drop pure-number tokens | `true` |
| `cleanup` | `min_token_length` | Minimum token length | `3` |
| `cleanup` | `dedupe_enabled` | Deduplicate by exact text | `true` |
| `cleanup` | `max_tokens_per_page` | Cap tokens per page | `200` |
| `crop` | `bbox_crop_padding_px` | Padding around bbox crops | `10` |
| `crop` | `bbox_crop_min_area_px` | Minimum crop area (pixels) | `4096` |
| `segment` | `expand_scale` | Bbox expansion for segmenter | `5.0` |
| `layout` | `single_word_max_tokens` | Max tokens for single-word classification | `2` |
| `confidence` | `multi_word_default` | Default confidence for multi-word cards | `0.5` |

Override with `--config <path>` CLI flag.

## Usage

### CLI Commands

#### `run` — Execute flashcard production pipeline

```powershell
python -m flashcard_engine.cli run `
  --input <path>           # PDF file or images folder (required)
  --type <pdf|images>      # Input type (required)
  --lang <code>            # OCR language code, e.g., "en", "ko" (required)
  --workspace <dir>        # Output workspace root (default: ./workspace)
  --source <name>          # Source identifier for cards (required)
  --dpi <int>              # PDF render DPI (default: 200)
  --min-confidence <float> # Min OCR confidence threshold (default: 0.7)
  --segmenter <off|mobilesam|fastsam>  # Segmenter mode (default: off)
  --segmenter-device <cpu|cuda|mps>    # Device for segmenter (default: cpu)
  --config <path>          # Custom config JSON (default: config/default.json)
  --use-mocked-ocr <dir>   # Directory with mocked cleaned OCR JSON (for CI)
```

#### `validate` — Verify output contract and file integrity

```powershell
python -m flashcard_engine.cli validate `
  --job-dir <path>         # Job directory (required)
  --apkg <path>            # Optional: validate an exported .apkg file
```

#### `export` — Export cards to CSV or Anki .apkg

```powershell
# CSV export
python -m flashcard_engine.cli export `
  --job-dir <path>         # Job directory (required)
  --format csv             # Export format (required)
  --out <path>             # Output file path (required)
  --include-review         # Include cards in review status (optional)

# APKG export (approved cards only)
python -m flashcard_engine.cli export `
  --job-dir <path> `
  --format apkg `
  --out <path> `
  --deck-name <name>       # Deck name (default: source_ref)
  --tags <comma-separated> # Tags for cards (optional)
```

#### `review-ui` — Generate static HTML review interface

```powershell
python -m flashcard_engine.cli review-ui --job-dir <path>
```

Outputs:
- `<job_dir>/review.html` — Open in browser (no server needed)
- `<job_dir>/review_feedback.json` — Feedback file for `apply-review`

#### `apply-review` — Apply human review feedback

```powershell
python -m flashcard_engine.cli apply-review `
  --job-dir <path> `
  --feedback <path>        # JSON file with review actions
```

Feedback JSON format:
```json
[
  {"card_id": "<sha1>", "action": "approve"},
  {"card_id": "<sha1>", "action": "reject"},
  {"card_id": "<sha1>", "action": "edit", "edited_text": "corrected text"}
]
```

### Typical Workflow

```
run → validate → review-ui → apply-review → export → validate
```

## Development

### Running Smoke Tests

Deterministic smoke tests use mocked OCR to avoid external dependencies:

```powershell
# v0.3 idempotency check (apply-review, CSV export)
python .\samples\smoke_no_ocr\check_v03_idempotency.py

# v0.4 review UI + APKG check
python .\samples\smoke_no_ocr\check_v04_review_ui_and_apkg.py
```

### Manual Smoke Test (Copy/Paste)

```powershell
# 1) Generate deterministic input image
python .\samples\smoke_no_ocr\generate_image.py

# 2) Run pipeline with mocked OCR
python -m flashcard_engine.cli run `
  --input .\samples\smoke_no_ocr\pages `
  --type images `
  --lang en `
  --workspace .\workspace `
  --source "smoke_no_ocr" `
  --min-confidence 0.99 `
  --segmenter off `
  --use-mocked-ocr .\samples\smoke_no_ocr\stage\ocr

# 3) Validate
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>

# 4) Generate review UI
python -m flashcard_engine.cli review-ui --job-dir .\workspace\jobs\<job_id>

# 5) Apply sample feedback
python -m flashcard_engine.cli apply-review `
  --job-dir .\workspace\jobs\<job_id> `
  --feedback .\samples\smoke_no_ocr\review_feedback.example.json

# 6) Export APKG
python -m flashcard_engine.cli export `
  --job-dir .\workspace\jobs\<job_id> `
  --format apkg `
  --out .\workspace\smoke_no_ocr.apkg `
  --deck-name "smoke_no_ocr"

# 7) Validate again (including APKG)
python -m flashcard_engine.cli validate `
  --job-dir .\workspace\jobs\<job_id> `
  --apkg .\workspace\smoke_no_ocr.apkg
```

### QA Gate

The project enforces explicit validation guarantees. See [docs/qa-gate.md](docs/qa-gate.md) for:
- Security: path traversal protection
- Output contract: required files and schemas
- Determinism: canonical token ordering
- Lifecycle: fallback methods force review status

## Output / Artifacts

All outputs are written to `<workspace>/jobs/<job_id>/`:

| File | Description |
|------|-------------|
| `result.json` | Job metadata + all cards with fields: `card_id`, `word`, `bbox_xyxy`, `method`, `front_image_path`, `status`, etc. |
| `review_queue.json` | Items needing human review with `review_reason` |
| `metrics.json` | Processing statistics (pages, cards, crops, errors) |
| `errors.jsonl` | Per-page/stage errors (JSON Lines format) |
| `pages/page_<n>.png` | Rendered page images |
| `pages/crops/...` | Cropped token images |

See [docs/output_contract.md](docs/output_contract.md) for full schema documentation.

### Card Status Lifecycle

- `active` — Ready for export
- `review` — Needs human review (excluded from default export)
- `rejected` — Discarded (always excluded from export)

### Export Rules

- **CSV:** Exports `active` cards by default; `--include-review` adds `review` cards
- **APKG:** Exports only `active` cards; rejected cards always excluded
- **Ordering:** Deterministic by `source_page_id` → `token_index`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Pillow ABI mismatch` on Windows | Delete `.venv`, recreate with correct Python version, reinstall requirements |
| `paddleocr not found` | OCR is optional; pipeline continues with empty tokens |
| `genanki not found` | Required for APKG export; install with `pip install genanki` |
| Validation fails with `finished=false` | Job did not complete; check `errors.jsonl` for page errors |
| Path traversal error | `front_image_path` must be job-relative; absolute/`..` paths are rejected |

## Roadmap

Based on code analysis, potential future work:
- [ ] Back-side content generation (currently empty in APKG)
- [ ] Additional export formats
- [ ] Web-based review UI (currently static HTML only)
- [ ] Better segmenter integration (MobileSAM/FastSAM paths exist but may need testing)

## License

Not specified in repository.

## TODO / Unknown

- **License:** No LICENSE file found; license status unknown
- **Test suite:** No pytest/unittest files detected; only integration smoke tests
- **Linting:** `ruff` configured in pyproject.toml but no CI configuration found
- **Segmenter dependencies:** `ultralytics`, `opencv-python` mentioned as optional but installation not documented
- **PaddlePaddle installation:** Varies by OS/CUDA version; user must determine correct wheel

## Evidence Map

| README Claim | Supporting File(s) |
|--------------|-------------------|
| CLI commands (run, validate, export, review-ui, apply-review) | [flashcard_engine/cli.py](flashcard_engine/cli.py) |
| Pipeline stages (OCR, cleaner, layout, segmenter, builder) | [flashcard_engine/pipeline.py](flashcard_engine/pipeline.py) |
| Output files (result.json, review_queue.json, metrics.json, errors.jsonl) | [flashcard_engine/job.py](flashcard_engine/job.py#L64-L93) |
| Configuration schema | [config/default.json](config/default.json) |
| CSV export rules | [flashcard_engine/exporter.py](flashcard_engine/exporter.py) |
| APKG export rules | [flashcard_engine/exporters/apkg.py](flashcard_engine/exporters/apkg.py) |
| Review feedback format | [flashcard_engine/review.py](flashcard_engine/review.py) |
| Review UI generation | [flashcard_engine/review_ui.py](flashcard_engine/review_ui.py) |
| Validation logic | [flashcard_engine/validator.py](flashcard_engine/validator.py) |
| Path security (ensure_job_relative_path) | [flashcard_engine/utils.py](flashcard_engine/utils.py#L100-L125) |
| QA Gate guarantees | [docs/qa-gate.md](docs/qa-gate.md) |
| Output contract schema | [docs/output_contract.md](docs/output_contract.md) |
| Smoke test scripts | [samples/smoke_no_ocr/check_v03_idempotency.py](samples/smoke_no_ocr/check_v03_idempotency.py), [samples/smoke_no_ocr/check_v04_review_ui_and_apkg.py](samples/smoke_no_ocr/check_v04_review_ui_and_apkg.py) |
| Python version requirement | [pyproject.toml](pyproject.toml) (`requires-python = ">=3.10"`) |
| Dependencies | [requirements.txt](requirements.txt) |

## Diff Summary

| Change | Reason |
|--------|--------|
| Restructured to standard README outline | Improved navigability for new contributors |
| Added Architecture section with pipeline diagram | Documents actual code flow from `pipeline.py` |
| Added folder structure map | Helps understand module responsibilities |
| Consolidated CLI usage into single section with all flags | Complete reference derived from `cli.py` |
| Added Configuration table | Documents actual config keys from `default.json` |
| Added Troubleshooting section | Common issues observed in code comments |
| Added TODO/Unknown section | Honest about missing license, tests, etc. |
| Added Evidence Map | Verifiable claims with file references |
| Removed outdated v0.1/v0.2/v0.3 version references | README now describes current state |
| Translated Korean comments to English | Broader accessibility |
| Removed duplicate workflow sections | Consolidated into single Usage section |

---

**Last Verified:** 2026-01-30

