# 테스트 결과 및 중요 사항 (2026-01-24)

## 요약 메트릭

- **STEP 1 (Images, segmenter off)**
  - Job: `workspace/jobs/ad594bc7-b0b8-46c2-848e-ce7cb71b51d1/`
  - Metrics: `pages_total=11`, `pages_processed=11`, `cards_total=243`, `ocr_empty_count=0`, `segment_success_count=0`, `errors=0`
  - 생성된 크롭: 0

- **STEP 2 (Images, mobilesam, 튜닝 적용)**
  - Job: `workspace/jobs/31c698ca-cba9-4c4d-bf14-ee66b3a6a42a/`
  - Metrics: `pages_total=11`, `pages_processed=11`, `cards_total=243`, `segment_success_count=1`, `segment_fail_count=0`, `ocr_empty_count=0`, `errors=0`
  - 생성된 크롭: 1 (`pages/crops/page_001_apple.png`)

## 샘플 카드 (STEP2의 상위 3개 예)

- `page_001`
  - layout_type: `single_word`
  - word: `apple`
  - front_image_path: `pages/crops/page_001_apple.png`
  - confidence: `0.9997`
  - needs_review: `False`

- `page_002` (예: 멀티워드)
  - word: `akt`
  - front_image_path: `pages/page_002.png`
  - confidence: `0.569`
  - needs_review: `True` (multi_word + low_confidence)

## 중요 사항 / 주의점

- **Python & PaddleOCR 버전 민감성:** PaddleOCR는 Python 버전과 `paddlepaddle` 버전에 민감합니다. 안정 동작을 위해 Python `3.11` + `paddlepaddle==2.6.2` + `paddleocr<3` 권장.
- **Pillow 바이너리 이슈:** 일부 환경에서 최신 Pillow가 문제를 일으켜 `pillow==10.4.0`로 고정함.
- **Segmenter 동작:**
  - `--segmenter off`는 분절 단계에서 크롭을 생성하지 않도록 동작하도록 수정됨.
  - MobileSAM/FastSAM 미설치 시 모델은 실패로 처리되고, 파이프라인은 bbox_fallback으로 크롭을 생성(면적이 충분할 때)
- **설정 튜닝:** `config/default.json`에서 `segment.expand_scale=5.0`, `segment.min_area_ratio=0.01`로 조정하여 area gate를 완화함.
- **Fail-soft 원칙:** 각 페이지 단계는 예외 발생 시 오류를 `errors.jsonl`에 기록하고 파이프라인은 다음 페이지로 계속 진행함.

## 재현 커맨드 (복사하여 사용)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install "paddlepaddle==2.6.2" "paddleocr<3" "pillow==10.4.0"
.\.venv\Scripts\python -m flashcard_engine.cli run --input .\Images --type images --lang en --workspace .\workspace --source "ImagesTest_SAM" --min-confidence 0.7 --segmenter mobilesam --segmenter-device cpu --config .\config\default.json
```
