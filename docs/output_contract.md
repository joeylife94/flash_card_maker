# Output Contract (v0.3)

이 문서는 `flashcard_engine` 실행 결과가 **항상 생성해야 하는 파일**과 그 스키마(주요 필드)를 정의합니다.

## 항상 생성되는 파일

Job 디렉토리: `workspace/jobs/<job_id>/`

- `result.json`
- `review_queue.json`
- `metrics.json`
- `errors.jsonl`

보조 산출물:
- `pages/page_<n>.png` (입력 페이지 렌더링)
- `pages/crops/...` (가능하면 생성되는 크롭 이미지)

## Canonical crop path (multi_word)

```text
workspace/jobs/<job_id>/pages/crops/page_<page:03d>/token_<i:04d>_<slug>.png
```

## Crop failure / gate behavior

- bbox crop이 게이트(min area 등)로 제외되거나 저장 실패 시:
  - 카드 `method`는 `page`로 남습니다.
  - 카드 `front_image_path`는 전체 페이지 이미지를 가리킵니다.
  - 해당 상황은 `reasons`/`review_queue`에 `CROP_FAILED` 또는 `CROP_GATED_SMALL` / `CROP_GATED_RATIO` 등으로 기록될 수 있습니다.

## result.json

형태:

```json
{
  "job": { "job_id": "...", "source": "...", "input": {"type": "pdf|images", "path": "..."}, "created_at": "..." },
  "cards": [
    {
      "card_id": "sha1hex",
      "page_id": "page_003",
      "source_page_id": "page_003",
      "layout_type": "single_word|multi_word|unknown",
      "word": "example",
      "bbox_xyxy": [x0, y0, x1, y1],
      "method": "page|bbox_crop|segmenter",
      "front_image_path": "pages/...png",
      "source_ref": "book.pdf#page=3",
      "confidence": 0.73,
      "needs_review": false,
      "status": "active|review|rejected",
      "created_at": "2026-01-25T00:00:00+00:00",
      "updated_at": "2026-01-25T00:00:00+00:00",
      "reasons": ["LOW_CONFIDENCE", "CROP_FAILED"]
    }
  ]
}
```

### 필드 설명

- `card_id`: 안정적인 식별자. `sha1(source_ref + page_id + text + bbox_xyxy)`.
- `bbox_xyxy`: OCR 토큰 bbox (좌상단/우하단, 이미지 픽셀 좌표).
- `method`:
  - `page`: 전체 페이지 이미지 참조
  - `bbox_crop`: OCR bbox 기반 크롭
  - `segmenter`: 세그먼터(FastSAM/MobileSAM 등) 기반 크롭
- `front_image_path`: job 디렉토리 기준 상대경로.
- `status`:
  - `active`: export 대상
  - `review`: 사람 검토 필요 (기본 export에서 제외)
  - `rejected`: 폐기 (export 제외)
- `created_at` / `updated_at`: 카드 생성/업데이트 시각 (ISO8601)
- `source_page_id`: 원본 페이지 id (기본적으로 `page_id`와 동일)

## review_queue.json

형태:

```json
{
  "items": [
    {
      "card_id": "sha1hex",
      "page_id": "page_003",
      "source_ref": "book.pdf#page=3",
      "text": "example",
      "bbox_xyxy": [x0, y0, x1, y1],
      "review_reason": "LOW_CONFIDENCE",
      "reason": "LOW_CONFIDENCE",
      "suggested_action": "manual_review",
      "front_image_path": "pages/...png"
    }
  ]
}
```

### review_reason enum

- `OCR_EMPTY`
- `LAYOUT_UNCERTAIN`
- `WORD_MISSING`
- `LOW_CONFIDENCE`
- `SEGMENT_FAILED`
- `CROP_FAILED`
- `CROP_GATED_SMALL`
- `CROP_GATED_RATIO`
- `SUSPICIOUS_BBOX`
- `HEURISTIC_WARNING`

## metrics.json

주요 필드:

- `pages_total`, `pages_processed`
- `pages_single_word`, `pages_multi_word`
- `cards_total`
- `review_items_total` (호환성) / `review_items`
- `multiword_tokens`, `multiword_crops_written`
- `multiword_crop_failures`, `multiword_crops_gated_small`, `multiword_crops_gated_ratio`
- `deduped_tokens`
- `segment_success_count`, `segment_fail_count`
- `ocr_empty_count`

## errors.jsonl

페이지/스테이지 단위 예외를 JSON Lines로 기록합니다. Fail-soft 정책으로 인해 오류가 있어도 실행은 계속됩니다.

각 라인 예시:

```json
{"page_id":"page_003","stage":"bbox_crop","message":"token_0012: ..."}
```

## Export rules (v0.3)

- `export --format csv`는 기본적으로 `needs_review == false` AND `status != rejected` 카드만 내보냅니다.
- `--include-review`를 지정하면 `review` 상태 카드도 포함할 수 있습니다.
- 이미지가 누락된 경우 해당 카드는 skip하고 `errors.jsonl`에 경고를 기록합니다.
