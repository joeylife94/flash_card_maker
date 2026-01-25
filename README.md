# flash_card_maker (flashcard_engine)

MVP v0.1: 로컬에서 PDF/이미지 입력을 받아 Flashcard 제작에 필요한 **재료 세트(이미지+텍스트)** 를 생산합니다.

## Output Contract (항상 생성)
- `workspace/jobs/<job_id>/pages/page_<n>.png`
- `workspace/jobs/<job_id>/pages/crops/page_<page:03d>/token_<i:04d>_<slug>.png` (multi_word bbox crops)
- `workspace/jobs/<job_id>/pages/crops/<page_id>_<word>.png` (single_word segmenter/bbox fallback)
- `workspace/jobs/<job_id>/result.json`
- `workspace/jobs/<job_id>/review_queue.json`
- `workspace/jobs/<job_id>/metrics.json`
- `workspace/jobs/<job_id>/errors.jsonl`

자세한 스키마/필드 설명: [docs/output_contract.md](docs/output_contract.md)

## Install
권장: Python 3.11–3.12

필수(최소 실행):
- `pip install pillow pymupdf`

OCR(PaddleOCR) 사용 시:
- `pip install paddleocr`
- `paddlepaddle` 설치는 OS/환경에 따라 다릅니다. (CPU 버전 권장)

세그먼터(FastSAM/MobileSAM)는 옵션이며 설치되지 않아도 **Fail-soft** 로 동작합니다.

### Environment notes
- Windows에서 Pillow ABI mismatch가 발생할 수 있습니다 (예: Python 3.12 venv에 cp313 Pillow wheel이 섞인 경우). 이런 경우 `.venv`를 삭제 후 원하는 Python 버전으로 venv를 다시 만들고 `pip install -r requirements.txt`를 다시 실행하세요.

## Run
```powershell
python -m flashcard_engine.cli run \
  --input .\samples\book.pdf \
  --type pdf \
  --lang en \
  --workspace .\workspace \
  --source "BookName" \
  --dpi 200 \
  --min-confidence 0.7 \
  --segmenter off \
  --segmenter-device cpu
```

이미지 폴더 입력:
```powershell
python -m flashcard_engine.cli run --input .\samples\images --type images --lang en --workspace .\workspace --source "MyDeck"
```

## Typical workflow (v0.3)

```text
run -> validate -> apply-review -> export -> validate
```

## Typical workflow (v0.4)

```text
run -> validate -> review-ui (or apply-review) -> export apkg -> validate
```

Example:

```powershell
python -m flashcard_engine.cli run --input .\samples\images --type images --lang en --workspace .\workspace --source "MyDeck"

python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>

# (optional) apply human review actions
python -m flashcard_engine.cli apply-review --job-dir .\workspace\jobs\<job_id> --feedback .\review_feedback.json

# export only non-review cards by default
python -m flashcard_engine.cli export --job-dir .\workspace\jobs\<job_id> --format csv --out .\deck.csv

# v0.4: export as Anki .apkg (approved cards only; images embedded)
python -m flashcard_engine.cli export \
  --job-dir .\workspace\jobs\<job_id> \
  --format apkg \
  --out .\deck.apkg \
  --deck-name "MyDeck" \
  --tags "book,unit1"

# validate again after review/export
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>
```

주의: low-confidence/review 상태의 카드는 export 전에 검토하는 것을 권장합니다.

## Mocked OCR (for smoke/CI)
`--use-mocked-ocr <dir>`를 지정하면, 각 페이지에 대해 아래 파일 중 하나가 **존재하고 정상 파싱되는 경우** 해당 페이지는 실제 OCR/cleaner를 건너뛰고 mocked cleaned OCR을 사용합니다.

- `<dir>/<page_id>_clean.json`
- `<dir>/<page_id>.cleaned.json`
- `<dir>/page_<idx:03d>.cleaned.json` (0-based)
- `<dir>/page_<idx>.cleaned.json` (0-based)

동작 규칙:
- 파일이 없거나 파싱 실패 시: `errors.jsonl`에 경고를 기록하고 실제 OCR로 fallback합니다.
- 실제 OCR이 설치되지 않았거나 실패하더라도: fail-soft로 진행하며 Output Contract 파일들은 항상 생성됩니다(해당 페이지는 빈 tokens로 처리될 수 있음).

## Notes
- MVP는 중단 없는 배치가 최우선입니다. 페이지 단위 예외는 `errors.jsonl`에 기록하고 계속 진행합니다.
- `segmenter=off` 일 때도 single_word 페이지는 OCR bbox 기반 간단 크롭을 시도합니다(가능하면).

## Crop failure / gate behavior
- bbox crop이 게이트(min area 등)로 제외되거나 저장 실패 시: 카드 `method`는 `page`로 남고 `front_image_path`는 전체 페이지 이미지를 가리킵니다.

## Validate
Output Contract 파일과 `front_image_path` 참조가 모두 존재하는지 검증:

```powershell
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>

# (옵션) APKG 무결성 체크(Zip/collection.anki2/media count)
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id> --apkg .\workspace\smoke_no_ocr.apkg
```

## Deterministic smoke (v0.3, copy/paste)

이 섹션만 따라 하면 v0.3 기능( `apply-review`, `export`, `validate`, mocked OCR )을 다른 문서 없이 재현할 수 있습니다.

```powershell
# 0) generate deterministic input image
python .\samples\smoke_no_ocr\generate_image.py

# 1) run (mocked OCR, force review via high min-confidence)
python -m flashcard_engine.cli run \
  --input .\samples\smoke_no_ocr\pages \
  --type images \
  --lang en \
  --workspace .\workspace \
  --source "smoke_no_ocr" \
  --min-confidence 0.99 \
  --segmenter off \
  --use-mocked-ocr .\samples\smoke_no_ocr\stage\ocr

# (prints job dir, example: .\workspace\jobs\<job_id>)

# 2) validate immediately after run
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>

# 3) apply-review (edit-only idempotency check)
python -m flashcard_engine.cli apply-review \
  --job-dir .\workspace\jobs\<job_id> \
  --feedback .\samples\smoke_no_ocr\review_feedback.edit_only.json

# re-run apply-review with the same feedback (must report applied=0)
python -m flashcard_engine.cli apply-review \
  --job-dir .\workspace\jobs\<job_id> \
  --feedback .\samples\smoke_no_ocr\review_feedback.edit_only.json

# 3b) apply full deterministic feedback (approves/rejects so export has 4 rows)
python -m flashcard_engine.cli apply-review \
  --job-dir .\workspace\jobs\<job_id> \
  --feedback .\samples\smoke_no_ocr\review_feedback.example.json

# 4) export csv
python -m flashcard_engine.cli export \
  --job-dir .\workspace\jobs\<job_id> \
  --format csv \
  --out .\workspace\smoke_no_ocr.csv

# v0.4) generate review UI (static HTML, no server)
python -m flashcard_engine.cli review-ui --job-dir .\workspace\jobs\<job_id>

# v0.4) export apkg (approved cards only)
python -m flashcard_engine.cli export \
  --job-dir .\workspace\jobs\<job_id> \
  --format apkg \
  --out .\workspace\smoke_no_ocr.apkg \
  --deck-name "smoke_no_ocr" \
  --tags "smoke"

# 5) validate again
python -m flashcard_engine.cli validate --job-dir .\workspace\jobs\<job_id>
```

또는 스크립트로 한 번에 검증:

```powershell
python .\samples\smoke_no_ocr\check_v03_idempotency.py
```

Note:
- v0.4에서도 CSV export는 그대로 지원됩니다.
- `review-ui`는 브라우저에서 `review_feedback.json`을 작성(또는 다운로드)하고, 그 JSON을 `apply-review`에 전달하는 UX를 제공합니다.

Expected CSV (stable fields only):

- rejected 카드는 export에서 제외됩니다 (기본 smoke fixture에서 `delta`를 reject하도록 예시가 제공됨)
- row order는 token index 기준으로 유지됩니다: `token_0000`, `token_0001`, `token_0002`, `token_0004`

```csv
front_text,back_text,front_image_path,source_ref,card_id,review_reason
alpha,,pages/crops/page_001/token_0000_alpha_*.png,pages/page_000.png,ab7e194355c5a24f69e3af2906fc2e673a513933,
beta,,pages/crops/page_001/token_0001_beta_*.png,pages/page_000.png,1daf32f023b902aa77e910afefe24973e6d6130f,
gamma,,pages/crops/page_001/token_0002_gamma_*.png,pages/page_000.png,10fc9906c94ab8937e7fb30069ed31505e44c707,
epsilon,,pages/crops/page_001/token_0004_epsilon_*.png,pages/page_000.png,67cb74167c1978379d678935c63088429535fcb2,
```
