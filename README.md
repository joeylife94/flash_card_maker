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
권장: Python 3.10+

필수(최소 실행):
- `pip install pillow pymupdf`

OCR(PaddleOCR) 사용 시:
- `pip install paddleocr`
- `paddlepaddle` 설치는 OS/환경에 따라 다릅니다. (CPU 버전 권장)

세그먼터(FastSAM/MobileSAM)는 옵션이며 설치되지 않아도 **Fail-soft** 로 동작합니다.

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
```
