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

Expected:
- `metrics.json` has `multiword_tokens = 5` and `multiword_crops_written = 5` (default config)
- `review_queue.json` is usually empty (unless you lower confidences or break bboxes)

## Expected outputs

- Crops written under (canonical path):
  - `workspace/jobs/<job_id>/pages/crops/page_<page:03d>/token_<i:04d>_<slug>.png`
- Default crop count: `5`
- Metrics keys to check:
  - `multiword_crops_written`
  - `multiword_crop_failures`
  - `multiword_crops_gated_small`
  - `multiword_crops_gated_ratio`
