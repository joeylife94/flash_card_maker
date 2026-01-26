# QA Gate: Hardening & Validation Guarantees

This project intentionally implements a **QA Gate**: a set of explicit, testable guarantees that must remain true over time.

The QA Gate exists because this engine is **fail-soft by design** (it tries to produce outputs even when OCR/cropping/segmentation degrade). The gate ensures that “keep going” never becomes “silently lie”.

## Checklist

### Security

- **`front_image_path` is job-relative; path traversal is blocked**
  - **Rationale:** `front_image_path` is treated as untrusted. Export/validate must never read files outside the job directory.
  - **Enforced by:** job-relative normalization and rejection
    - `flashcard_engine/utils.py` (`ensure_job_relative_path`)
    - `flashcard_engine/validator.py` (image ref validation)
    - `flashcard_engine/exporter.py` / `flashcard_engine/exporters/apkg.py` (export hard-fail on unsafe paths)

### Output Contract

- **Required contract artifacts exist per job**
  - `result.json`, `review_queue.json`, `metrics.json`, `errors.jsonl`
  - **Rationale:** downstream tooling assumes these files exist even under fail-soft behavior.
  - **Enforced by:** `flashcard_engine/validator.py` (`validate_job_dir` contract checks)

- **Cards have required fields and types**
  - Required keys include (non-exhaustive): `card_id`, `method`, `bbox_xyxy`, `front_image_path`, `page_id`, `source_ref`, `source_page_id`, `token_index`
  - **Rationale:** consumers depend on stable, typed fields and ordering keys.
  - **Enforced by:** `flashcard_engine/validator.py` (`_validate_cards_schema`)

- **Review items have required fields and types**
  - Required keys include: `card_id`, `review_reason`, `page_id`, `source_ref`, `front_image_path` (and optional `bbox_xyxy`)
  - **Rationale:** review tooling must remain interoperable.
  - **Enforced by:** `flashcard_engine/validator.py` (`_validate_review_schema`)

### Determinism

- **Canonical token ordering before `token_index` assignment**
  - **Rationale:** deterministic ordering is required for reproducible CSV/APKG exports and stable regression tests.
  - **Enforced by:** canonical ordering logic during build
    - `flashcard_engine/builder.py` (token ordering + `token_index` assignment)

### Validation Truthfulness

- **Placeholder jobs FAIL validation**
  - A job that only has empty “contract files” is *not* considered complete.
  - **Rationale:** prevents “green but meaningless” outputs.
  - **Enforced by:** completion markers in metrics + required job metadata
    - `flashcard_engine/job.py` (`init_job_outputs` writes `finished=false` and missing `completed_at`)
    - `flashcard_engine/validator.py` (`metrics.json: finished==true` and non-empty `completed_at`, plus job fields)

- **Completion marker required**
  - `metrics.json.finished` must be `true` and `metrics.json.completed_at` must be a non-empty string.
  - **Rationale:** the QA gate distinguishes “ran and completed” from “created placeholders”.
  - **Enforced by:** `flashcard_engine/validator.py` (`validate_job_dir`)

### APKG Validation

- **Media is validated via DB references + `media` mapping (shared images allowed)**
  - Reads referenced media filenames from `collection.anki2` (SQLite `notes.flds`), then ensures:
    - each referenced filename exists in the `media` mapping values, and
    - at least one mapped blob exists in the zip
  - **Rationale:** avoids false failures when multiple notes share the same image filename.
  - **Enforced by:** `flashcard_engine/validator.py` (`validate_apkg`)

### Lifecycle Truth + Export Honesty

- **Fallback crops force `needs_review=true`**
  - `method in {page_fallback, bbox_fallback}` implies degraded quality.
  - **Rationale:** fail-soft is allowed, but degradation must be surfaced.
  - **Enforced by:**
    - `flashcard_engine/builder.py` (sets fallback `method`, `needs_review`, and stable reason codes)
    - `flashcard_engine/validator.py` (fails if fallback method exists but `needs_review=false`)

- **Degradation reasons are never hidden in exports**
  - CSV always emits `review_reason` when reasons exist (even for exported/active cards).
  - **Rationale:** exports must not hide quality degradations that impact downstream study.
  - **Enforced by:** `flashcard_engine/exporter.py` (CSV `review_reason` population)

## QA Gate Repro Commands

All commands assume you run from the repo root.

### 1) Placeholder job FAIL

```powershell
# Create a placeholder job (contract files only, not finished)
.\.venv\Scripts\python.exe -c "from flashcard_engine.job import create_job_dirs, init_job_outputs; p=create_job_dirs('workspace','qa_placeholder'); init_job_outputs(p); print(p.job_dir)"

# Must FAIL (unfinished metrics + missing job metadata)
.\.venv\Scripts\python.exe -m flashcard_engine.cli validate --job-dir .\workspace\jobs\qa_placeholder
```

### 2) Path traversal export FAIL

```powershell
# Create a valid job first (deterministic smoke)
.\.venv\Scripts\python.exe .\samples\smoke_no_ocr\check_v03_idempotency.py

# Pick the newest job directory from the smoke workspace
$JOB_DIR = Get-ChildItem .\workspace\smoke_v03_idempotency\jobs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | % FullName

# Copy it so we don't mutate the original
Copy-Item -Recurse -Force $JOB_DIR .\workspace\jobs\qa_path_traversal

# Tamper: inject a traversal path into the first card
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; p=Path('workspace/jobs/qa_path_traversal/result.json'); o=json.loads(p.read_text(encoding='utf-8')); o['cards'][0]['front_image_path']='..\\..\\README.md'; p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding='utf-8')"

# Export must FAIL (unsafe front_image_path)
.\.venv\Scripts\python.exe -m flashcard_engine.cli export --job-dir .\workspace\jobs\qa_path_traversal --format csv --out .\workspace\qa_path_traversal.csv --include-review
```

### 3) Shared-image APKG PASS

```powershell
# Use the repro APKG that contains shared media
# (Any valid job-dir works; reuse a fresh v0.4 smoke job so job validation is also green.)
.\.venv\Scripts\python.exe .\samples\smoke_no_ocr\check_v04_review_ui_and_apkg.py
$JOB_DIR = Get-ChildItem .\workspace\smoke_v04_review_ui_apkg\jobs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | % FullName

.\.venv\Scripts\python.exe -m flashcard_engine.cli validate --job-dir $JOB_DIR --apkg .\workspace\repro\shared_dedup.apkg
```

### 4) Tampered APKG FAIL

```powershell
# Same job-dir as above, but a tampered apkg must FAIL
.\.venv\Scripts\python.exe -m flashcard_engine.cli validate --job-dir $JOB_DIR --apkg .\workspace\repro\shared_dedup_tampered.apkg
```

### 5) Fallback => needs_review enforced

```powershell
# Run a job designed to force page_fallback (mocked OCR without usable bboxes)
.\.venv\Scripts\python.exe -m flashcard_engine.cli run --input .\workspace\repro\images --type images --lang en --workspace .\workspace\qa_fallback --source qa_fallback --segmenter off --use-mocked-ocr .\workspace\repro\mocked
$JOB_DIR = Get-ChildItem .\workspace\qa_fallback\jobs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | % FullName

# Validate should PASS, and cards should be marked needs_review=true due to fallback
.\.venv\Scripts\python.exe -m flashcard_engine.cli validate --job-dir $JOB_DIR

# Optional: print a small proof summary
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; r=json.loads((Path(r'$JOB_DIR')/'result.json').read_text(encoding='utf-8')); cards=r.get('cards',[]); print('cards=',len(cards)); print('methods=',sorted({c.get('method') for c in cards})); print('needs_review_true=',sum(bool(c.get('needs_review')) for c in cards)); print('reasons_sample=',sorted({(c.get('reasons') or [''])[0] for c in cards})[:5])"
```

## High-risk regression areas

- **Token ordering / `token_index` assignment** (`flashcard_engine/builder.py`)
  - A small change can silently break determinism and reorder exports.

- **Job-relative path enforcement** (`flashcard_engine/utils.py`, validator/exporters)
  - Any new usage of `front_image_path` must go through job-relative checks.

- **APKG media validation logic** (`flashcard_engine/validator.py`)
  - Must continue validating *referenced* media via SQLite + mapping (not by counts).

- **Exporter media handling / dedup** (`flashcard_engine/exporters/apkg.py`)
  - Easy to regress shared media behavior or introduce false-fail validation.

- **Validator schema strictness & completion markers** (`flashcard_engine/validator.py`)
  - Relaxing required fields or finished/completed markers re-enables placeholder “green” jobs.
