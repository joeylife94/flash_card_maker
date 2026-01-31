# 🎴 Flash Card Maker - 완벽 설계 문서

> **버전**: 2.0.0  
> **최종 업데이트**: 2026-02-01  
> **목적**: PDF/이미지에서 그림과 텍스트를 분리 추출하여 플래시카드 생성

---

## 🆕 v2.0 업데이트 내용

| 기능 | 설명 |
|------|------|
| **EasyOCR 통합** | PaddleOCR에서 EasyOCR로 변경 (numpy 2.x 호환) |
| **FastSAM 통합** | 고속 세그멘테이션 (Ultralytics FastSAM-s) |
| **리뷰 UI HTML** | 인터랙티브 웹 리뷰 인터페이스 |
| **양방향 카드** | Picture→Text, Text→Picture 양방향 지원 |

---

## 📋 목차

1. [개요](#1-개요)
2. [핵심 기능](#2-핵심-기능)
3. [파이프라인 아키텍처](#3-파이프라인-아키텍처)
4. [사용법](#4-사용법)
5. [출력 구조](#5-출력-구조)
6. [모듈 설명](#6-모듈-설명)
7. [설정 옵션](#7-설정-옵션)
8. [문제 해결](#8-문제-해결)

---

## 1. 개요

### 1.1 무엇을 하는 도구인가?

**Flash Card Maker**는 어휘 학습 자료(PDF, 이미지)에서:

1. **그림(Picture)** 영역을 자동 감지하여 추출
2. **텍스트(Caption)** 영역을 OCR로 인식하여 추출
3. 그림-텍스트를 **자동 매칭**
4. **Anki 플래시카드**로 내보내기

### 1.2 왜 필요한가?

- 영어 단어장, 어휘 교재를 플래시카드로 변환
- 수동으로 그림/단어를 복사-붙여넣기하는 시간 절약
- 일관된 형식의 학습 자료 생성

### 1.3 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Fail-soft** | 에러가 발생해도 처리 계속, 항상 출력 생성 |
| **결정론적** | 같은 입력 = 같은 출력 (재현 가능) |
| **학습 가능** | 사용자 피드백으로 정확도 개선 |

---

## 2. 핵심 기능

### 2.1 지원 입력

| 형식 | 설명 |
|------|------|
| **PDF** | PyMuPDF로 페이지별 이미지 렌더링 |
| **이미지 폴더** | PNG, JPG, BMP, TIFF, WebP |

### 2.2 지원 출력

| 형식 | 설명 |
|------|------|
| **Anki (.apkg)** | genanki 사용, 미디어 임베드 |
| **CSV** | 범용 포맷, 다른 앱 호환 |

### 2.3 파이프라인 모드

| 모드 | 명령어 | 설명 |
|------|--------|------|
| **Pair 모드** | `--mode pair --sam` | 그림/텍스트 분리 추출 (권장) |
| **Flashcard 모드** | `--mode flashcard` | 단일 단어 카드 생성 (레거시) |

---

## 3. 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FLASH CARD MAKER PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐                                                           │
│  │  INPUT   │ PDF 파일 또는 이미지 폴더                                    │
│  └────┬─────┘                                                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                       │
│  │ PageProvider │ 각 페이지를 PIL Image로 변환                            │
│  └────┬─────────┘                                                       │
│       │                                                                 │
│       ▼                                                                 │
│  ┌────────────────────────────────────────────────────────┐             │
│  │              DUAL DETECTION (병렬 처리)                  │             │
│  │  ┌─────────────────┐     ┌──────────────────┐          │             │
│  │  │  TextDetector   │     │  PictureDetector │          │             │
│  │  │  (EasyOCR)      │     │  (FastSAM)       │          │             │
│  │  └────────┬────────┘     └────────┬─────────┘          │             │
│  │           │                       │                    │             │
│  │           ▼                       ▼                    │             │
│  │    텍스트 영역 bbox         그림 영역 bbox               │             │
│  │    + OCR 텍스트             + 마스크                    │             │
│  └────────────────────────────────────────────────────────┘             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                       │
│  │PairingEngine │ 그림-텍스트 매칭 (거리/방향 기반)                         │
│  └────┬─────────┘                                                       │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                       │
│  │   Cropper    │ 개별 그림/텍스트 이미지 저장                              │
│  └────┬─────────┘                                                       │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                       │
│  │   Exporter   │ Anki (.apkg) 또는 CSV로 내보내기                        │
│  └──────────────┘                                                       │
│                                                                         │
│  OUTPUT:                                                                │
│  ┌────────────────────────────────────────┐                             │
│  │  📁 workspace/output/job_<id>/         │                             │
│  │  ├── 📁 page_01/                       │                             │
│  │  │   ├── 📁 pair_001/                  │                             │
│  │  │   │   ├── 🖼️ image.png (그림)       │                             │
│  │  │   │   ├── 📝 text.png (텍스트)      │                             │
│  │  │   │   └── 📋 meta.json              │                             │
│  │  │   └── 📁 pair_002/                  │                             │
│  │  ├── 📄 job_summary.json               │                             │
│  │  └── 📄 result.json                    │                             │
│  └────────────────────────────────────────┘                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 사용법

### 4.1 설치

```powershell
# 1. 가상 환경 생성
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치:
pip install pillow pymupdf genanki numpy

# 3. OCR 설치 (EasyOCR - 권장)
pip install easyocr

# 4. FastSAM 설치 (권장)
pip install ultralytics opencv-python
```

### 4.2 빠른 시작 (Quick Extract)

**가장 간단한 사용법** - 이미지 폴더에서 바로 Anki 카드 생성:

```powershell
python -m flashcard_engine extract --input ./my_images --export ./flashcards.apkg
```

### 4.3 단계별 사용법

#### Step 1: Pair 추출 실행

```powershell
# 이미지 폴더에서 그림/텍스트 추출
python -m flashcard_engine run `
    --input ./Images `
    --type images `
    --lang en `
    --source "VocabularyBook" `
    --mode pair `
    --sam `
    --workspace ./workspace
```

#### Step 2: 결과 확인

```powershell
# Job 디렉토리 확인 (출력에 표시됨)
# workspace/jobs/2026-01-31/12-30-45__abc12345/
```

#### Step 3: 플래시카드 빌드

```powershell
# 기본 (Picture → Text)
python -m flashcard_engine build-flashcards `
    --job-dir ./workspace/jobs/2026-01-31/12-30-45__abc12345 `
    --source "VocabularyBook"

# 양방향 카드 (Picture → Text, Text → Picture)
python -m flashcard_engine build-flashcards `
    --job-dir ./workspace/jobs/2026-01-31/12-30-45__abc12345 `
    --source "VocabularyBook" `
    --reverse
```

#### Step 4: 리뷰 UI 생성 (선택)

```powershell
# 인터랙티브 HTML 리뷰 인터페이스 생성
python -m flashcard_engine generate-review-html `
    --job-dir ./workspace/jobs/2026-01-31/12-30-45__abc12345
```

#### Step 5: Anki로 내보내기

```powershell
python -m flashcard_engine export `
    --job-dir ./workspace/jobs/2026-01-31/12-30-45__abc12345 `
    --format apkg `
    --out ./my_flashcards.apkg `
    --deck-name "English Vocabulary"
```

### 4.4 PDF 처리

```powershell
python -m flashcard_engine run `
    --input ./textbook.pdf `
    --type pdf `
    --lang en `
    --source "Textbook" `
    --mode pair `
    --sam `
    --dpi 200
```

---

## 5. 출력 구조

### 5.1 Job 디렉토리

```
workspace/output/job_<id>/
├── page_01/
│   ├── pair_001/
│   │   ├── image.png       # 추출된 그림
│   │   ├── text.png        # 추출된 텍스트 영역
│   │   └── meta.json       # 메타데이터
│   ├── pair_002/
│   │   └── ...
│   └── summary.json        # 페이지 요약
├── page_02/
│   └── ...
├── job_summary.json        # 전체 Job 요약
└── result.json             # 플래시카드 데이터
```

### 5.2 meta.json 구조

```json
{
  "pair_id": "abc123def456",
  "order_index": 0,
  "picture_bbox": [20, 20, 380, 280],
  "text_bbox": [20, 300, 380, 380],
  "caption_text": "Apple",
  "has_text": true,
  "needs_review": false,
  "reasons": [],
  "confidence": 0.92
}
```

### 5.3 result.json 구조

```json
{
  "job": {
    "job_id": "abc12345",
    "mode": "pair_sam",
    "source": "VocabularyBook",
    "created_at": "2026-01-31T12:30:45+00:00"
  },
  "cards": [
    {
      "card_id": "pair_001",
      "page_id": "page_01",
      "word": "Apple",
      "front_image_path": "page_01/pair_001/image.png",
      "status": "active",
      "confidence": 0.92
    }
  ]
}
```

---

## 6. 모듈 설명

### 6.1 핵심 모듈

| 모듈 | 파일 | 역할 |
|------|------|------|
| **CLI** | `cli.py` | 명령줄 인터페이스 |
| **Pipeline** | `pipeline.py` | 전체 파이프라인 조율 |
| **SAM Extractor** | `sam_pair_extractor.py` | FastSAM/EasyOCR 기반 추출 |
| **Pair Extractor** | `pair_extractor.py` | 그리드 기반 추출 (레거시) |
| **Flashcard Builder** | `pair_flashcard_builder.py` | Pair → Flashcard 변환 (양방향 지원) |

### 6.2 지원 모듈

| 모듈 | 파일 | 역할 |
|------|------|------|
| **Page Provider** | `page_provider.py` | PDF/이미지 로딩 |
| **OCR** | `ocr.py` | EasyOCR 래퍼 |
| **Review UI** | `review_ui_generator.py` | HTML 리뷰 인터페이스 생성 |
| **Exporter** | `exporters/apkg.py` | Anki 내보내기 |
| **Config** | `config.py` | 설정 로딩 |
| **Learning** | `learning.py` | 학습 캐시 |

### 6.3 데이터 흐름

```
1. Input → PageProvider → PIL.Image 리스트
2. Image → TextDetector → TextBlock 리스트 (bbox + OCR text)
3. Image → PictureDetector → PictureCandidate 리스트 (bbox + mask)
4. (Pictures, TextBlocks) → PairingEngine → Matched Pairs
5. Pairs → Cropper → image.png + text.png
6. Pairs → FlashcardBuilder → result.json
7. result.json → ApkgExporter → .apkg 파일
```

---

## 7. 설정 옵션

### 7.1 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | flashcard | `pair` 또는 `flashcard` |
| `--sam` | false | SAM 기반 그림 검출 사용 |
| `--lang` | en | OCR 언어 (en, ch, ko 등) |
| `--device` | cpu | SAM 디바이스 (cpu, cuda, mps) |
| `--dpi` | 200 | PDF 렌더링 해상도 |

### 7.2 config/default.json

```json
{
  "cleanup": {
    "lowercase": true,
    "min_token_length": 3,
    "dedupe_enabled": true
  },
  "crop": {
    "bbox_crop_padding_px": 10
  },
  "segment": {
    "min_area_ratio": 0.01
  }
}
```

### 7.3 SAM 설정 (PairConfig)

```python
@dataclass
class PairConfig:
    # 마스크 필터링
    min_mask_area_ratio: float = 0.02   # 너무 작은 마스크 제외
    max_mask_area_ratio: float = 0.85   # 배경 마스크 제외
    text_iou_threshold: float = 0.5     # 텍스트와 겹치면 제외
    
    # 매칭
    max_pairing_distance_px: int = 500  # 최대 매칭 거리
    search_direction: str = "below"     # 텍스트 검색 방향
```

---

## 8. 문제 해결

### 8.1 OCR이 작동하지 않음

```powershell
# EasyOCR 설치 (권장)
pip install easyocr

# 또는 PaddleOCR (legacy)
# pip install paddlepaddle paddleocr

# GPU 사용 시 (EasyOCR)
# gpu=True 옵션 활성화
```

### 8.2 FastSAM이 작동하지 않음

```powershell
# Ultralytics 설치
pip install ultralytics opencv-python

# 모델 다운로드 확인
# FastSAM-s.pt가 프로젝트 루트에 있어야 함
# 없으면 자동 다운로드됨
```

### 8.3 그림이 잘 검출되지 않음

- `min_mask_area_ratio` 낮추기 (작은 그림 검출)
- `max_mask_area_ratio` 높이기 (큰 그림 검출)
- `--device cuda` 사용 (더 정확한 검출)

### 8.4 텍스트 매칭이 틀림

- `search_direction` 변경 (below, above, right, left, nearest)
- `max_pairing_distance_px` 조정
- 피드백 적용으로 학습 개선

### 8.5 Anki 내보내기 실패

```powershell
# genanki 설치
pip install genanki

# 이미지 경로 확인
# result.json의 front_image_path가 올바른지 확인
```

---

## 📝 Quick Reference Card

```
# 1. 빠른 추출 (All-in-one)
python -m flashcard_engine extract --input ./images --export ./cards.apkg

# 2. 단계별 실행
python -m flashcard_engine run --input ./images --type images --lang en --source "Book" --mode pair --sam
python -m flashcard_engine build-flashcards --job-dir <job_dir> --source "Book" --reverse
python -m flashcard_engine export --job-dir <job_dir> --format apkg --out ./cards.apkg

# 3. 리뷰 UI 생성
python -m flashcard_engine generate-review-html --job-dir <job_dir>

# 4. PDF 처리
python -m flashcard_engine run --input ./book.pdf --type pdf --lang en --source "Book" --mode pair --sam

# 5. 검증
python -m flashcard_engine validate --job-dir <job_dir>
```

---

## 🔧 CLI 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `run` | 파이프라인 실행 (추출) |
| `validate` | Job 검증 |
| `export` | Anki/CSV 내보내기 |
| `review-ui` | 리뷰 UI 서버 시작 |
| `apply-review` | 리뷰 피드백 적용 |
| `apply-pair-feedback` | Pair 피드백 적용 |
| `learning-stats` | 학습 통계 표시 |
| `build-flashcards` | 플래시카드 빌드 (`--reverse` 옵션) |
| `extract` | 빠른 추출 (All-in-one) |
| `generate-review-html` | HTML 리뷰 인터페이스 생성 |

---

**🎉 이제 Flash Card Maker v2.0을 사용하여 효율적으로 학습 자료를 만들 수 있습니다!**
