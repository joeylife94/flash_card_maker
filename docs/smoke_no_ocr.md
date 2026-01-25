# OCR-free smoke test (CI-ready)

This repo includes a tiny deterministic smoke fixture under `samples/smoke_no_ocr/`.

Purpose:
- Verify v0.2 bbox cropping + schema + Output Contract
- Run fast (< 1s) and without PaddleOCR

Run:

```powershell
python .\samples\smoke_no_ocr\generate_image.py

python -m flashcard_engine.cli run \
  --input .\samples\smoke_no_ocr\pages \
  --type images \
  --lang en \
  --workspace .\workspace \
  --source smoke_no_ocr \
  --segmenter off \
  --use-mocked-ocr .\samples\smoke_no_ocr\stage\ocr

python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>
```
