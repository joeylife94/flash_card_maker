# flash_card_maker (flashcard_engine)

MVP v0.1: 로컬에서 PDF/이미지 입력을 받아 Flashcard 제작에 필요한 **재료 세트(이미지+텍스트)** 를 생산합니다.

## Output Contract (항상 생성)
- `workspace/jobs/<job_id>/pages/page_<n>.png`
- `workspace/jobs/<job_id>/pages/crops/page_<n>_<word>.png` (가능하면)
- `workspace/jobs/<job_id>/result.json`
- `workspace/jobs/<job_id>/review_queue.json`
- `workspace/jobs/<job_id>/metrics.json`
- `workspace/jobs/<job_id>/errors.jsonl`

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

## Notes
- MVP는 중단 없는 배치가 최우선입니다. 페이지 단위 예외는 `errors.jsonl`에 기록하고 계속 진행합니다.
- `segmenter=off` 일 때도 single_word 페이지는 OCR bbox 기반 간단 크롭을 시도합니다(가능하면).
