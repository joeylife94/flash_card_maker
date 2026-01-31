# Output Contract (v0.4.1)

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
      "token_index": 0,
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
- `token_index`: 페이지 내 토큰/카드의 순서 (v0.4.1+ deterministic ordering용)

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
- `debug_panels_enabled` (v0.5+): boolean, whether panel debug was enabled
- `panels_total` (v0.5+): total panels processed when debug enabled
- `panels_needing_review` (v0.5+): panels flagged for review
- `panels_blank` (v0.5+): panels detected as blank/meaningless

## errors.jsonl

페이지/스테이지 단위 예외를 JSON Lines로 기록합니다. Fail-soft 정책으로 인해 오류가 있어도 실행은 계속됩니다.

각 라인 예시:

```json
{"page_id":"page_003","stage":"bbox_crop","message":"token_0012: ..."}
```

---

## Panel Debug Pack (v0.5+ / EXPERIMENTAL)

Panel mode is **experimental and unstable**. The debug pack provides visibility into panel extraction failures.

### Enabling Panel Debug

```bash
python -m flashcard_engine.cli run --input ./Images --type images --lang en --source MyBook --debug-panels
```

### Generated Artifacts

When `--debug-panels` is enabled, the following artifacts are generated:

#### 1. Annotated Page Images

Path: `pages/annotated/page_XXX_panels.png`

- Original page with detected panel bounding boxes drawn
- Block index labels for each panel
- Reading order: top → bottom, left → right
- Color coding: green (active), orange (review), red (rejected), gray (blank)

#### 2. Panel Crop Images

Path: `pages/panels/page_XXX/panel_NNN.png`

- Individual crop images for each detected panel
- These are the EXACT images used as card fronts
- Job-relative paths only

#### 3. Panel Diagnostics JSON

Path: `stage/panel/page_XXX.json`

Schema:

```json
{
  "page_id": "page_001",
  "page_index": 0,
  "image_size": [1200, 1600],
  "panels_count": 12,
  "panels_needing_review": 3,
  "panels_blank": 1,
  "created_at": "2026-01-31T12:00:00+00:00",
  "panels": [
    {
      "block_index": 0,
      "bbox_xyxy": [100, 100, 300, 250],
      "blank_score": 0.15,
      "entropy": 5.8,
      "edge_density": 0.045,
      "ocr_text_raw": "vocabulary",
      "caption_text": "vocabulary",
      "reasons": [],
      "needs_review": false,
      "status": "active",
      "crop_path": "pages/panels/page_001/panel_000.png"
    }
  ]
}
```

### Panel Diagnostic Fields

| Field | Type | Description |
|-------|------|-------------|
| `block_index` | int | Reading order index (0-based) |
| `bbox_xyxy` | [x0,y0,x1,y1] | Absolute pixel coordinates |
| `blank_score` | float | 0.0 = content-rich, 1.0 = blank |
| `entropy` | float | Grayscale entropy (higher = more content) |
| `edge_density` | float | Edge pixel ratio (higher = more edges) |
| `ocr_text_raw` | string | Raw OCR output |
| `caption_text` | string | Cleaned caption text |
| `reasons` | string[] | Failure/warning codes |
| `needs_review` | bool | Whether human review is needed |
| `status` | string | active / review / rejected |
| `crop_path` | string | Job-relative path to crop image |

### Panel Reason Codes

- `BLANK_CROP`: Crop detected as visually blank/meaningless
- `OCR_EMPTY`: No OCR text extracted
- `PARTIAL_CROP`: Incomplete panel capture
- `OVERLAP`: Panel overlaps with others
- `FALLBACK_USED`: Fallback method was used
- `BBOX_INVALID`: Invalid bounding box
- `SMALL_AREA`: Panel area below threshold
- `LOW_EDGE_DENSITY`: Few structural edges detected
- `LOW_ENTROPY`: Low information content

### Blank Score Computation

The `blank_score` metric combines two signals:

1. **Grayscale Entropy**: Measures histogram-based information content
   - Low entropy (< 3.5) indicates uniform/blank regions
   
2. **Edge Density**: Measures structural content via gradient magnitude
   - Low edge density (< 0.02) indicates few visual features

Combined score: `blank_score = 1.0 - (0.6 * norm_entropy + 0.4 * norm_edge)`

Panels with `blank_score > 0.7` are automatically flagged for review.

---

## Learning Data Pipeline (v0.5+)

The system captures training data from human review actions for future model improvement.

### Enabling Learning Data

Learning data capture is **automatic** when `apply-review` is called. No additional flags needed.

### Learning Record Schema

Records are stored in `<job_dir>/learning/training_records.jsonl`:

```json
{
  "record_id": "a1b2c3d4e5f6",
  "card_id": "sha1hex...",
  "page_id": "page_001",
  "page_image_path": "pages/page_001.png",
  "predicted_bbox_xyxy": [100, 100, 300, 250],
  "predicted_text": "original_word",
  "predicted_confidence": 0.75,
  "corrected_bbox_xyxy": null,
  "corrected_text": "corrected_word",
  "action": "edit",
  "failure_reasons": ["LOW_CONFIDENCE"],
  "layout_type": "multi_word",
  "method": "bbox_crop",
  "created_at": "2026-01-31T12:00:00+00:00",
  "source_ref": "book.pdf#page=1"
}
```

### Learning Data Files

| File | Description |
|------|-------------|
| `learning/training_records.jsonl` | Append-only training records |
| `learning/learning_stats.json` | Cumulative statistics |
| `learning/exports/` | Batch exports for model training |

### Learning Statistics

```json
{
  "created_at": "2026-01-31T12:00:00+00:00",
  "updated_at": "2026-01-31T13:00:00+00:00",
  "records_written": 150,
  "approvals": 100,
  "rejections": 20,
  "edits": 30,
  "bbox_corrections": 5,
  "text_corrections": 25
}
```

### Extended Feedback Format (v0.5+)

The feedback JSON now supports bbox corrections:

```json
{
  "items": [
    {"card_id": "...", "action": "approve"},
    {"card_id": "...", "action": "reject"},
    {"card_id": "...", "action": "edit", "edited_text": "corrected"},
    {"card_id": "...", "action": "edit", "edited_text": "new", "corrected_bbox": [110, 120, 310, 260]}
  ]
}
```

---

## Export rules (CSV)

- `export --format csv`는 기본적으로 `needs_review == false` AND `status != rejected` 카드만 내보냅니다.
- `--include-review`를 지정하면 `review` 상태 카드도 포함할 수 있습니다.
- 이미지가 누락된 경우 해당 카드는 skip하고 `errors.jsonl`에 경고를 기록합니다.

## Export rules (APKG) — Anki .apkg

CLI:

```text
python -m flashcard_engine.cli export --job-dir <JOB_DIR> --format apkg --out <deck.apkg>
```

Rules:

- `status == active` 카드만 export 합니다 (approved only).
- `rejected` 카드는 항상 제외됩니다.
- row/order는 `source_page_id` -> `token_index` 순으로 deterministic 하게 유지됩니다 (v0.4.1+, path parsing 없음).
- media(이미지)는 `.apkg`에 embed 됩니다.
- 이미지가 누락된 경우 해당 카드는 skip하고 `errors.jsonl`에 경고를 기록합니다.
- export 가능한 카드가 0장이면 명확한 에러 메시지와 함께 non-zero exit 입니다.

Deck metadata:

- `--deck-name`: deck 이름 (default: 첫 카드의 `source_ref`)
- `--tags`: comma-separated tags (optional)

Model fields:

- Front: image + text
- Back: text (현재는 empty 허용)

## Review UI (v0.4.1)

CLI:

```text
python -m flashcard_engine.cli review-ui --job-dir <JOB_DIR>
```

Output:

- `<JOB_DIR>/review.html` (static HTML; no server)
- `<JOB_DIR>/review_feedback.json` (없으면 메타데이터 포함 object로 생성; apply-review는 legacy list format도 호환)

Feedback format (v0.4.1; apply-review와 호환):

```json
{
  "feedback_version": 1,
  "generated_at": "2026-01-25T00:00:00+00:00",
  "items": [
    {"card_id": "...", "action": "approve"},
    {"card_id": "...", "action": "reject"},
    {"card_id": "...", "action": "edit", "edited_text": "..."}
  ]
}
```

Legacy (v0.4) list format도 지원:

```json
[
  {"card_id": "...", "action": "approve"},
  {"card_id": "...", "action": "reject"},
  {"card_id": "...", "action": "edit", "edited_text": "..."}
]
```

## Card lifecycle (v0.4)

상태 전이 규칙:

- `review -> active`: `apply-review`의 `approve` 또는 `edit`(edit 후 approve)
- `review -> rejected`: `apply-review`의 `reject`
- `rejected` 카드는 기본 export에서 항상 제외되며, `review_queue.json`에도 다시 나타나지 않습니다.

---

## Pair Mode (v0.5+ / EXPERIMENTAL)

Pair mode separates vocabulary workbook pages into **picture crops** and **text crops**, with explicit binding between them.

### Enabling Pair Mode

```bash
python -m flashcard_engine.cli run --input ./Images --type images --lang en --source MyBook --mode pair
```

### Timeline Job Directory (R4)

Pair mode uses timeline-based job directories by default:

```
workspace/jobs/YYYY-MM-DD/HH-MM-SS__<shortid>/
```

Example: `workspace/jobs/2026-01-31/14-30-45__a1b2c3d4/`

Benefits:
- Chronological sorting by default
- Clear timestamp for each run
- Nested structure for daily organization

### Generated Artifacts

#### 1. result_pairs.json

Primary output containing all extracted pairs:

```json
{
  "schema_version": "1.0",
  "job_id": "2026-01-31/14-30-45__a1b2c3d4",
  "mode": "pair",
  "created_at": "2026-01-31T14:30:45+00:00",
  "pairs": [
    {
      "pair_id": "abc123def456",
      "page_id": "page_001",
      "item_index": 0,
      "picture_path": "pages/items/page_001/item_000/picture.png",
      "text_path": "pages/items/page_001/item_000/text.png",
      "caption_text": "APPLE",
      "bbox_item_xyxy": [20, 20, 380, 500],
      "bbox_picture_xyxy": [20, 20, 380, 350],
      "bbox_text_xyxy": [20, 350, 380, 500],
      "status": "active",
      "needs_review": false,
      "reasons": [],
      "blank_score_picture": 0.15,
      "blank_score_text": 0.25,
      "confidence": 0.85
    }
  ],
  "diagnostics": [...]
}
```

#### 2. Picture/Text Crops (R1)

For each item block detected, two separate crops are created:

```
pages/items/page_XXX/item_NNN/picture.png
pages/items/page_XXX/item_NNN/text.png
```

These are the EXACT separated image files for each vocabulary item.

#### 3. Per-Page Diagnostics

Path: `stage/pair/page_XXX.json`

Schema:

```json
{
  "page_id": "page_001",
  "page_index": 0,
  "image_size": [400, 500],
  "items_detected": 4,
  "pairs_extracted": 4,
  "pairs_needing_review": 1,
  "grid_detected": true,
  "grid_rows": 2,
  "grid_cols": 2,
  "text_ratio_used": 0.3,
  "text_position_used": "bottom",
  "learned_adjustments": ["text_ratio=0.35"],
  "created_at": "2026-01-31T14:30:45+00:00"
}
```

### Pair Fields (R2: Binding/Linking)

| Field | Type | Description |
|-------|------|-------------|
| `pair_id` | string | Stable identifier (sha1-based) |
| `page_id` | string | Source page identifier |
| `item_index` | int | Index within page (reading order) |
| `picture_path` | string | Job-relative path to picture crop |
| `text_path` | string | Job-relative path to text crop |
| `caption_text` | string | Extracted text from text region |
| `bbox_item_xyxy` | [x0,y0,x1,y1] | Full item bounding box |
| `bbox_picture_xyxy` | [x0,y0,x1,y1] | Picture region bbox |
| `bbox_text_xyxy` | [x0,y0,x1,y1] | Text region bbox |
| `status` | string | active / review / rejected |
| `needs_review` | bool | Whether human review is needed |
| `reasons` | string[] | Failure/warning codes |
| `blank_score_picture` | float | 0.0-1.0 picture blankness |
| `blank_score_text` | float | 0.0-1.0 text region blankness |
| `confidence` | float | Overall extraction confidence |

### Pair Reason Codes

- `BLANK_PICTURE`: Picture region appears blank
- `BLANK_TEXT`: Text region appears blank
- `NO_TEXT_DETECTED`: No text found via OCR
- `SMALL_ITEM`: Item block below minimum area
- `UNCERTAIN_SPLIT`: Uncertain picture/text division
- `OCR_EMPTY`: OCR returned no text
- `LEARNED_CORRECTION`: Used cached correction

### Self-Improving Learning Loop (R3)

Pair mode includes a workspace-level learning cache that improves across runs.

#### Enabling/Disabling Learning

Learning is **enabled by default**. Disable with:

```bash
python -m flashcard_engine.cli run --mode pair --no-learning ...
```

#### Learning Cache Location

```
workspace/learning/
├── adaptive_cache.json      # Learned parameters
├── caption_corrections.json # Text corrections
└── records/                 # Feedback history
    └── YYYY-MM-DDTHH-MM-SS__<job_short_id>.jsonl
```

#### Applying Pair Feedback

```bash
python -m flashcard_engine.cli apply-pair-feedback \
  --job-dir workspace/jobs/2026-01-31/14-30-45__a1b2c3d4 \
  --feedback pair_feedback.json
```

Feedback format:

```json
{
  "items": [
    {"pair_id": "abc123", "action": "approve"},
    {"pair_id": "def456", "action": "reject"},
    {
      "pair_id": "ghi789",
      "action": "edit",
      "edited_caption": "CORRECTED TEXT",
      "corrected_text_ratio": 0.35,
      "corrected_text_position": "bottom"
    }
  ]
}
```

#### Observable Improvement

After applying feedback:

1. **Caption corrections** are cached and reused in future runs
2. **Text ratio** adjustments are learned per page layout
3. **Text position** (top/bottom/left/right) can be corrected
4. **Blank threshold** adapts based on approval patterns

Check learning stats:

```bash
python -m flashcard_engine.cli learning-stats --workspace ./workspace
```

Output:

```
total_records=150
caption_corrections=45
cached_parameters=12
blank_threshold=0.700
```

### Metrics (Pair Mode)

```json
{
  "mode": "pair",
  "created_at": "2026-01-31T14:30:45+00:00",
  "pages_total": 10,
  "pages_processed": 10,
  "pairs_total": 40,
  "pairs_needing_review": 5,
  "pairs_blank_picture": 2,
  "pairs_blank_text": 3,
  "pairs_ocr_empty": 4,
  "items_detected": 40,
  "learning_enabled": true,
  "learning_records_total": 100,
  "learning_caption_corrections": 30,
  "learning_cached_parameters": 8
}
```
