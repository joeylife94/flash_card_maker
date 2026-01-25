# smoke_no_ocr (OCR-free smoke sample)

Goal: verify v0.2 **bbox crop generation**, **schema fields**, and **Output Contract validation** without installing/running PaddleOCR.

## What this validates

- Multi-word page tokens produce bbox crops under:
  - `workspace/jobs/<job_id>/pages/crops/page_<page:03d>/token_<i:04d>_<slug>.png`
- Each generated card includes:
  - `card_id`, `bbox_xyxy`, `method` (expected: `bbox_crop`), and a valid `front_image_path`
- `python -m flashcard_engine.cli validate --job-dir ...` passes

This fixture is deterministic:
- `pages/page_000.png` can be regenerated via `generate_image.py`
- mocked OCR is fixed in `stage/ocr/page_000.cleaned.json`

## What this intentionally skips

- Real OCR extraction (PaddleOCR)
- Single-word segmentation (`segmenter`)

## How to run

From repo root:

```powershell
python .\samples\smoke_no_ocr\generate_image.py

python -m flashcard_engine.cli run \
  --input .\samples\smoke_no_ocr\pages \
  --type images \
  --lang en \
  --workspace .\workspace \
  --source "smoke_no_ocr" \
  --min-confidence 0.99 \
  --segmenter off \
  --use-mocked-ocr .\samples\smoke_no_ocr\stage\ocr
```

### --use-mocked-ocr behavior

- If `<dir>/page_XXX.cleaned.json` (or other supported names) exists and parses: the pipeline uses it as *cleaned OCR* for that page (skips real OCR + cleaner).
- If missing or parse fails: a warning is recorded in `errors.jsonl` and the pipeline falls back to real OCR.
- If real OCR is unavailable/fails: the run still completes fail-soft and produces the Output Contract (that page may have empty tokens).

The command prints the job directory path.

Then validate:

```powershell
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>
```

## End-to-end example (v0.3)

This run uses `--min-confidence 0.99` to force review items (`LOW_CONFIDENCE`), then applies a deterministic feedback file and exports CSV.

```powershell
python -m flashcard_engine.cli apply-review \
  --job-dir .\workspace\jobs\<job_id> \
  --feedback .\samples\smoke_no_ocr\review_feedback.example.json

python -m flashcard_engine.cli export \
  --job-dir .\workspace\jobs\<job_id> \
  --format csv \
  --out .\workspace\smoke_no_ocr.csv
```

Expected:
- `metrics.json` has `multiword_tokens = 5` and `multiword_crops_written = 5` (default config)
- `review_queue.json` is usually empty (unless you lower confidences or break bboxes)

CSV expected shape (default export excludes rejected/review):

- 1 header + 4 rows
- token order preserved: `token_0000`, `token_0001`, `token_0002`, `token_0004`

Example (paths/sha suffixes may differ if you change crop naming rules):

```csv
front_text,back_text,front_image_path,source_ref,card_id,review_reason
alpha,,pages/crops/page_001/token_0000_alpha_*.png,pages/page_000.png,ab7e194355c5a24f69e3af2906fc2e673a513933,
beta,,pages/crops/page_001/token_0001_beta_*.png,pages/page_000.png,1daf32f023b902aa77e910afefe24973e6d6130f,
gamma,,pages/crops/page_001/token_0002_gamma_*.png,pages/page_000.png,10fc9906c94ab8937e7fb30069ed31505e44c707,
epsilon,,pages/crops/page_001/token_0004_epsilon_*.png,pages/page_000.png,67cb74167c1978379d678935c63088429535fcb2,
```

## Expected outputs

- Crops written under (canonical path):
  - `workspace/jobs/<job_id>/pages/crops/page_<page:03d>/token_<i:04d>_<slug>.png`
- Default crop count: `5`
- Metrics keys to check:
  - `multiword_crops_written`
  - `multiword_crop_failures`
  - `multiword_crops_gated_small`
  - `multiword_crops_gated_ratio`
