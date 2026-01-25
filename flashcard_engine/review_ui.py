from __future__ import annotations

import json
from datetime import datetime, timezone
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json, write_json


_TOKEN_RE = re.compile(r"token_(\d{4})_")


@dataclass
class ReviewUiStats:
    items: int
    html_path: Path
    feedback_path: Path


def _token_index_from_path(front_image_path: str | None) -> int:
    if not front_image_path:
        return 10**9
    m = _TOKEN_RE.search(front_image_path)
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except Exception:
        return 10**9


def _page_sort_key(page_id: str) -> int:
    try:
        if page_id.startswith("page_"):
            return int(page_id.split("_")[-1])
    except Exception:
        pass
    return 10**9


def _normalize_review_item(it: dict[str, Any]) -> dict[str, Any] | None:
    card_id = str(it.get("card_id") or "").strip()
    if not card_id:
        return None

    page_id = str(it.get("page_id") or "")
    text = str(it.get("text") or "")
    front_image_path = str(it.get("front_image_path") or "")
    source_ref = str(it.get("source_ref") or "")
    token_index = int(it.get("token_index") or 0)

    return {
        "card_id": card_id,
        "page_id": page_id,
      "token_index": token_index,
        "text": text,
        "front_image_path": front_image_path,
        "source_ref": source_ref,
        "review_reason": str(it.get("review_reason") or it.get("reason") or ""),
    }


def generate_review_ui(*, job_dir: str | Path) -> ReviewUiStats:
    """Generate a static, no-server review UI in the job directory.

    Writes:
    - <job_dir>/review.html
    - <job_dir>/review_feedback.json (created if missing; UI overwrites it)

    UI behavior:
    - Pure HTML + minimal JS
    - Approve/Reject/Edit buttons
    - Idempotent actions keyed by card_id (no duplicates)
    - Attempts to write review_feedback.json via File System Access API (Chromium)
      and provides a download fallback for other browsers.
    """

    job_dir = Path(job_dir)
    review_path = job_dir / "review_queue.json"
    feedback_path = job_dir / "review_feedback.json"
    html_path = job_dir / "review.html"

    review = load_json(review_path)
    items_raw = review.get("items", []) if isinstance(review, dict) else []

    items: list[dict[str, Any]] = []
    for it in items_raw:
        if not isinstance(it, dict):
            continue
        norm = _normalize_review_item(it)
        if norm is None:
            continue
        items.append(norm)

    # v0.4.1 deterministic ordering: page_id ASC, token_index ASC (no path parsing)
    items.sort(key=lambda x: (str(x.get("page_id") or ""), int(x.get("token_index") or 0)))

    now_iso = datetime.now(timezone.utc).isoformat()

    # Ensure feedback file exists (v0.4.1 object wrapper with metadata).
    feedback_changed = False
    if feedback_path.exists():
      try:
        feedback = load_json(feedback_path)
        if isinstance(feedback, list):
          feedback = {"feedback_version": 1, "generated_at": now_iso, "items": feedback}
          feedback_changed = True
        elif isinstance(feedback, dict):
          feedback.setdefault("feedback_version", 1)
          feedback.setdefault("generated_at", now_iso)
          feedback.setdefault("items", [])
        else:
          feedback = {"feedback_version": 1, "generated_at": now_iso, "items": []}
          feedback_changed = True
      except Exception:
        feedback = {"feedback_version": 1, "generated_at": now_iso, "items": []}
        feedback_changed = True
    else:
      feedback = {"feedback_version": 1, "generated_at": now_iso, "items": []}
      feedback_changed = True

    if feedback_changed:
      write_json(feedback_path, feedback)

    # Embed data into the HTML to avoid file:// fetch/CORS issues.
    items_json = json.dumps(items, ensure_ascii=False)
    feedback_json = json.dumps(feedback, ensure_ascii=False)

    html_template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>flashcard_engine review</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: ui-sans-serif, system-ui, Segoe UI, Arial; margin: 16px; }
    .topbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .pill { padding: 4px 10px; border: 1px solid #8884; border-radius: 999px; }
    .layout { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; max-width: 980px; }
    img { max-width: 100%; height: auto; border: 1px solid #8884; border-radius: 8px; }
    textarea { width: 100%; min-height: 80px; font-size: 16px; }
    button { padding: 8px 12px; border-radius: 8px; border: 1px solid #8886; background: #8882; cursor: pointer; }
    button.primary { background: #2a6df4; color: white; border-color: #2a6df4; }
    button.danger { background: #d33; color: white; border-color: #d33; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .muted { opacity: 0.7; }
    .status { font-weight: 600; }
    .cardbox { padding: 12px; border: 1px solid #8884; border-radius: 12px; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; padding: 1px 6px; border: 1px solid #8884; border-radius: 6px; }
  </style>
</head>
<body>
  <div class=\"topbar\">
    <span class=\"pill\">flashcard_engine review-ui</span>
    <span class=\"pill\">items: <span id=\"itemsCount\"></span></span>
    <span class=\"pill\">index: <span id=\"indexLabel\"></span></span>
    <span class=\"pill\">status: <span id=\"statusLabel\">Not saved</span></span>
    <button id=\"connectBtn\" class=\"primary\">Connect job folder (writes review_feedback.json)</button>
    <button id=\"downloadBtn\">Download review_feedback.json</button>
    <button id=\"copyBtn\">Copy feedback JSON</button>
  </div>

  <div class=\"layout\">
    <div class=\"cardbox\">
      <div class=\"row\">
        <button id=\"prevBtn\">Prev</button>
        <button id=\"nextBtn\">Next</button>
        <span class=\"muted\">Tip: use <span class=\"kbd\">←</span>/<span class=\"kbd\">→</span></span>
      </div>
      <hr />
      <div class=\"row\">
        <span class=\"muted\">card_id:</span> <span id=\"cardId\" class=\"kbd\"></span>
        <span class=\"muted\">page_id:</span> <span id=\"pageId\" class=\"kbd\"></span>
        <span class=\"muted\">reason:</span> <span id=\"reason\" class=\"kbd\"></span>
      </div>
      <div style=\"margin-top: 10px\">
        <img id=\"img\" alt=\"front image\" />
      </div>
      <div style=\"margin-top: 10px\">
        <div class=\"muted\">Text</div>
        <textarea id=\"textArea\" readonly></textarea>
      </div>
      <div style=\"margin-top: 10px\" class=\"row\">
        <button id=\"approveBtn\" class=\"primary\">Approve</button>
        <button id=\"rejectBtn\" class=\"danger\">Reject</button>
        <button id=\"editBtn\">Edit (approve)</button>
        <span class=\"status\">Current: <span id=\"currentAction\">(none)</span></span>
      </div>
      <div class=\"muted\" id=\"msg\" style=\"margin-top: 8px\"></div>
    </div>
  </div>

<script>
  const ITEMS = __ITEMS_JSON__;
  const EXISTING = __FEEDBACK_JSON__;

  const itemsCount = document.getElementById('itemsCount');
  const indexLabel = document.getElementById('indexLabel');
  const statusLabel = document.getElementById('statusLabel');
  const msg = document.getElementById('msg');

  const img = document.getElementById('img');
  const cardIdEl = document.getElementById('cardId');
  const pageIdEl = document.getElementById('pageId');
  const reasonEl = document.getElementById('reason');
  const textArea = document.getElementById('textArea');
  const currentAction = document.getElementById('currentAction');

  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  const approveBtn = document.getElementById('approveBtn');
  const rejectBtn = document.getElementById('rejectBtn');
  const editBtn = document.getElementById('editBtn');
  const connectBtn = document.getElementById('connectBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const copyBtn = document.getElementById('copyBtn');

  itemsCount.textContent = String(ITEMS.length);

  let idx = 0;
  const actionsById = new Map();
  const existingItems = (EXISTING && Array.isArray(EXISTING.items)) ? EXISTING.items : (Array.isArray(EXISTING) ? EXISTING : []);
  for (const a of existingItems) {
    if (a && a.card_id) actionsById.set(String(a.card_id), a);
  }

  let dirHandle = null;
  let fileHandle = null;

  function setMessage(s) {
    msg.textContent = s;
  }

  function currentItem() {
    return ITEMS[idx] || null;
  }

  function render() {
    const it = currentItem();
    if (!it) {
      indexLabel.textContent = '0/0';
      cardIdEl.textContent = '';
      pageIdEl.textContent = '';
      reasonEl.textContent = '';
      img.removeAttribute('src');
      textArea.value = '';
      currentAction.textContent = '(none)';
      setMessage('No review items.');
      return;
    }

    indexLabel.textContent = `${idx + 1}/${ITEMS.length}`;
    cardIdEl.textContent = it.card_id;
    pageIdEl.textContent = it.page_id || '';
    reasonEl.textContent = it.review_reason || '';
    img.src = it.front_image_path || '';
    textArea.value = it.text || '';

    const a = actionsById.get(String(it.card_id));
    if (!a) currentAction.textContent = '(none)';
    else if (a.action === 'edit') currentAction.textContent = `edit: ${a.edited_text || ''}`;
    else currentAction.textContent = a.action;
  }

  function actionsArray() {
    return Array.from(actionsById.values());
  }

  function feedbackObject() {
    return {
      feedback_version: 1,
      generated_at: new Date().toISOString(),
      items: actionsArray(),
    };
  }

  function setStatus(s) {
    statusLabel.textContent = s;
  }

  async function writeFeedbackIfPossible() {
    if (!fileHandle) {
      setStatus('Not saved');
      return;
    }
    const data = JSON.stringify(feedbackObject(), null, 2);
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();
    setStatus('Saved to file system');
  }

  function upsertAction(card_id, action, edited_text) {
    const entry = { card_id: String(card_id), action: action };
    if (action === 'edit') entry.edited_text = String(edited_text || '');
    actionsById.set(String(card_id), entry);
  }

  approveBtn.addEventListener('click', async () => {
    const it = currentItem();
    if (!it) return;
    upsertAction(it.card_id, 'approve');
    await writeFeedbackIfPossible();
    setMessage('Set: approve');
    render();
  });

  rejectBtn.addEventListener('click', async () => {
    const it = currentItem();
    if (!it) return;
    upsertAction(it.card_id, 'reject');
    await writeFeedbackIfPossible();
    setMessage('Set: reject');
    render();
  });

  editBtn.addEventListener('click', async () => {
    const it = currentItem();
    if (!it) return;
    const v = prompt('Edit text (this will approve):', it.text || '');
    if (v === null) return;
    const trimmed = String(v).trim();
    if (!trimmed) {
      setMessage('Edit canceled: empty text.');
      return;
    }
    upsertAction(it.card_id, 'edit', trimmed);
    await writeFeedbackIfPossible();
    setMessage('Set: edit (approve)');
    render();
  });

  prevBtn.addEventListener('click', () => {
    idx = Math.max(0, idx - 1);
    render();
  });

  nextBtn.addEventListener('click', () => {
    idx = Math.min(ITEMS.length - 1, idx + 1);
    render();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prevBtn.click();
    if (e.key === 'ArrowRight') nextBtn.click();
  });

  connectBtn.addEventListener('click', async () => {
    if (!('showDirectoryPicker' in window)) {
      setMessage('File System Access API not available in this browser. Use Download instead.');
      return;
    }
    try {
      dirHandle = await window.showDirectoryPicker({ id: 'flashcard_engine_review' });
      fileHandle = await dirHandle.getFileHandle('review_feedback.json', { create: true });
      await writeFeedbackIfPossible();
      setMessage('Connected. Actions will be written to review_feedback.json');
    } catch (err) {
      setMessage('Connect canceled.');
    }
  });

  downloadBtn.addEventListener('click', () => {
    const data = JSON.stringify(feedbackObject(), null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'review_feedback.json';
    a.click();
    URL.revokeObjectURL(url);
    setStatus('Downloaded feedback JSON');
    setMessage('Downloaded review_feedback.json');
  });

  copyBtn.addEventListener('click', async () => {
    const data = JSON.stringify(feedbackObject(), null, 2);
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(data);
        setStatus('Copied to clipboard');
        setMessage('Copied feedback JSON to clipboard');
        return;
      }
      throw new Error('clipboard not available');
    } catch (e) {
      // Fallback: show in prompt for manual copy.
      prompt('Copy feedback JSON:', data);
      setStatus('Copied to clipboard');
      setMessage('Clipboard unavailable; shown in prompt for manual copy.');
    }
  });

  render();
</script>
</body>
</html>
"""

    html = html_template.replace("__ITEMS_JSON__", items_json).replace("__FEEDBACK_JSON__", feedback_json)

    html_path.write_text(html, encoding="utf-8")

    return ReviewUiStats(items=len(items), html_path=html_path, feedback_path=feedback_path)
