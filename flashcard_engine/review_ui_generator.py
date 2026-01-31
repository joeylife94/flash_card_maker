"""Review UI Generator - Generate interactive HTML for pair review and correction.

This module creates a standalone HTML file that allows users to:
1. View all extracted picture-text pairs
2. Correct mismatched pairs via drag-and-drop
3. Edit text captions directly
4. Mark pairs for exclusion
5. Export corrected mappings as JSON

The HTML is self-contained with embedded CSS and JavaScript.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@dataclass
class ReviewPair:
    """A single pair for review."""
    pair_id: str
    order_index: int
    picture_path: str  # Path to picture image
    text_path: str | None  # Path to text image (if exists)
    caption_text: str  # OCR extracted text
    confidence: float  # Pairing confidence
    needs_review: bool  # Flagged for review
    reasons: list[str]  # Why it needs review


def image_to_base64(image_path: str | Path) -> str:
    """Convert image file to base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        return ""
    
    suffix = path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(suffix, 'image/png')
    
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{data}"


def generate_review_html(
    pairs: list[dict[str, Any]],
    output_path: str | Path,
    page_image_path: str | Path | None = None,
    title: str = "Flashcard Pair Review",
) -> Path:
    """Generate interactive HTML review page.
    
    Args:
        pairs: List of pair dictionaries with keys:
            - pair_id: Unique identifier
            - order_index: Canonical order
            - picture_path: Path to picture crop
            - text_path: Path to text crop (optional)
            - caption_text: Extracted text
            - confidence: 0-1 confidence score
            - needs_review: Boolean flag
            - reasons: List of review reasons
        output_path: Where to save the HTML file
        page_image_path: Optional path to original page image
        title: Page title
    
    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    
    # Convert images to base64 for self-contained HTML
    pairs_json = []
    for pair in pairs:
        p = dict(pair)
        if p.get('picture_path'):
            p['picture_data'] = image_to_base64(p['picture_path'])
        if p.get('text_path'):
            p['text_data'] = image_to_base64(p['text_path'])
        pairs_json.append(p)
    
    page_image_data = ""
    if page_image_path and Path(page_image_path).exists():
        page_image_data = image_to_base64(page_image_path)
    
    html_content = _generate_html_template(
        pairs_json=pairs_json,
        page_image_data=page_image_data,
        title=title,
    )
    
    output_path.write_text(html_content, encoding='utf-8')
    return output_path


def _generate_html_template(
    pairs_json: list[dict],
    page_image_data: str,
    title: str,
) -> str:
    """Generate the complete HTML template."""
    
    pairs_data = json.dumps(pairs_json, ensure_ascii=False, indent=2)
    
    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 24px;
            color: #2c3e50;
        }}
        
        .header-stats {{
            display: flex;
            gap: 20px;
            font-size: 14px;
            color: #666;
        }}
        
        .stat {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .stat-value {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        
        .btn-primary {{
            background: #3498db;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #2980b9;
        }}
        
        .btn-success {{
            background: #27ae60;
            color: white;
        }}
        
        .btn-success:hover {{
            background: #229954;
        }}
        
        .btn-danger {{
            background: #e74c3c;
            color: white;
        }}
        
        .btn-danger:hover {{
            background: #c0392b;
        }}
        
        .btn-outline {{
            background: white;
            color: #3498db;
            border: 2px solid #3498db;
        }}
        
        .btn-outline:hover {{
            background: #3498db;
            color: white;
        }}
        
        .filter-tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }}
        
        .filter-tab {{
            padding: 8px 16px;
            border: none;
            background: white;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
            color: #666;
            transition: all 0.2s;
        }}
        
        .filter-tab.active {{
            background: #3498db;
            color: white;
        }}
        
        .filter-tab:hover:not(.active) {{
            background: #ecf0f1;
        }}
        
        .pairs-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .pair-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.2s;
        }}
        
        .pair-card:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        
        .pair-card.needs-review {{
            border-left: 4px solid #e74c3c;
        }}
        
        .pair-card.excluded {{
            opacity: 0.5;
        }}
        
        .pair-card.edited {{
            border-left: 4px solid #f39c12;
        }}
        
        .card-header {{
            padding: 12px 16px;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .card-id {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .card-badges {{
            display: flex;
            gap: 5px;
        }}
        
        .badge {{
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 500;
        }}
        
        .badge-review {{
            background: #fdedec;
            color: #e74c3c;
        }}
        
        .badge-ok {{
            background: #e8f8f5;
            color: #27ae60;
        }}
        
        .badge-confidence {{
            background: #eaf2f8;
            color: #3498db;
        }}
        
        .card-images {{
            display: flex;
            gap: 10px;
            padding: 16px;
        }}
        
        .image-box {{
            flex: 1;
            min-height: 120px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border: 2px dashed #ddd;
        }}
        
        .image-box img {{
            max-width: 100%;
            max-height: 150px;
            object-fit: contain;
        }}
        
        .image-box.picture {{
            border-color: #3498db;
        }}
        
        .image-box.text {{
            border-color: #27ae60;
        }}
        
        .image-label {{
            position: absolute;
            top: 5px;
            left: 5px;
            padding: 2px 6px;
            background: rgba(0,0,0,0.5);
            color: white;
            font-size: 10px;
            border-radius: 4px;
        }}
        
        .card-caption {{
            padding: 0 16px 16px;
        }}
        
        .caption-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .caption-input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            resize: vertical;
            min-height: 60px;
        }}
        
        .caption-input:focus {{
            outline: none;
            border-color: #3498db;
        }}
        
        .card-actions {{
            padding: 12px 16px;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .card-actions button {{
            padding: 6px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        
        .btn-exclude {{
            background: #ecf0f1;
            color: #666;
        }}
        
        .btn-exclude:hover {{
            background: #e74c3c;
            color: white;
        }}
        
        .btn-approve {{
            background: #27ae60;
            color: white;
        }}
        
        .btn-approve:hover {{
            background: #229954;
        }}
        
        .reasons {{
            padding: 0 16px 10px;
        }}
        
        .reason-tag {{
            display: inline-block;
            padding: 3px 8px;
            background: #fff3cd;
            color: #856404;
            font-size: 11px;
            border-radius: 4px;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        /* Modal for export */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        
        .modal.active {{
            display: flex;
        }}
        
        .modal-content {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .modal-header h2 {{
            margin: 0;
        }}
        
        .modal-close {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #666;
        }}
        
        .json-output {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        /* Toast notifications */
        .toast {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 25px;
            background: #27ae60;
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
            z-index: 2000;
        }}
        
        .toast.show {{
            opacity: 1;
            transform: translateY(0);
        }}
        
        .toast.error {{
            background: #e74c3c;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 {title}</h1>
        <div class="header-stats">
            <div class="stat">
                <span>Total:</span>
                <span class="stat-value" id="stat-total">0</span>
            </div>
            <div class="stat">
                <span>Needs Review:</span>
                <span class="stat-value" id="stat-review">0</span>
            </div>
            <div class="stat">
                <span>Edited:</span>
                <span class="stat-value" id="stat-edited">0</span>
            </div>
            <div class="stat">
                <span>Excluded:</span>
                <span class="stat-value" id="stat-excluded">0</span>
            </div>
        </div>
    </div>
    
    <div class="controls">
        <button class="btn btn-success" onclick="exportJSON()">💾 Export JSON</button>
        <button class="btn btn-primary" onclick="downloadJSON()">⬇️ Download JSON</button>
        <button class="btn btn-outline" onclick="resetAll()">🔄 Reset All</button>
    </div>
    
    <div class="filter-tabs">
        <button class="filter-tab active" data-filter="all" onclick="filterPairs('all')">All</button>
        <button class="filter-tab" data-filter="review" onclick="filterPairs('review')">⚠️ Needs Review</button>
        <button class="filter-tab" data-filter="edited" onclick="filterPairs('edited')">✏️ Edited</button>
        <button class="filter-tab" data-filter="excluded" onclick="filterPairs('excluded')">🚫 Excluded</button>
    </div>
    
    <div class="pairs-grid" id="pairs-container"></div>
    
    <!-- Export Modal -->
    <div class="modal" id="export-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Export Review Feedback</h2>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <p>Copy the JSON below or use the Download button:</p>
            <pre class="json-output" id="json-output"></pre>
            <div style="margin-top: 15px; display: flex; gap: 10px;">
                <button class="btn btn-primary" onclick="copyJSON()">📋 Copy to Clipboard</button>
                <button class="btn btn-success" onclick="downloadJSON()">⬇️ Download File</button>
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <script>
        // Initial pair data from Python
        const initialPairs = {pairs_data};
        
        // Runtime state
        let pairs = [];
        let editedPairs = new Set();
        let excludedPairs = new Set();
        let currentFilter = 'all';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            pairs = JSON.parse(JSON.stringify(initialPairs));
            renderPairs();
            updateStats();
        }});
        
        function renderPairs() {{
            const container = document.getElementById('pairs-container');
            container.innerHTML = '';
            
            const filtered = pairs.filter(pair => {{
                if (currentFilter === 'all') return true;
                if (currentFilter === 'review') return pair.needs_review;
                if (currentFilter === 'edited') return editedPairs.has(pair.pair_id);
                if (currentFilter === 'excluded') return excludedPairs.has(pair.pair_id);
                return true;
            }});
            
            filtered.forEach((pair, idx) => {{
                const card = createPairCard(pair);
                container.appendChild(card);
            }});
        }}
        
        function createPairCard(pair) {{
            const card = document.createElement('div');
            card.className = 'pair-card';
            card.dataset.pairId = pair.pair_id;
            
            if (pair.needs_review) card.classList.add('needs-review');
            if (excludedPairs.has(pair.pair_id)) card.classList.add('excluded');
            if (editedPairs.has(pair.pair_id)) card.classList.add('edited');
            
            const confPercent = Math.round(pair.confidence * 100);
            const confClass = confPercent >= 80 ? 'badge-ok' : confPercent >= 50 ? 'badge-confidence' : 'badge-review';
            
            card.innerHTML = `
                <div class="card-header">
                    <span class="card-id">#${{pair.order_index + 1}} ${{pair.pair_id}}</span>
                    <div class="card-badges">
                        ${{pair.needs_review ? '<span class="badge badge-review">⚠️ Review</span>' : ''}}
                        <span class="badge ${{confClass}}">${{confPercent}}%</span>
                    </div>
                </div>
                
                <div class="card-images">
                    <div class="image-box picture">
                        ${{pair.picture_data ? `<img src="${{pair.picture_data}}" alt="Picture">` : '<span>No Picture</span>'}}
                    </div>
                    <div class="image-box text">
                        ${{pair.text_data ? `<img src="${{pair.text_data}}" alt="Text">` : '<span>No Text Image</span>'}}
                    </div>
                </div>
                
                ${{pair.reasons && pair.reasons.length > 0 ? `
                <div class="reasons">
                    ${{pair.reasons.map(r => `<span class="reason-tag">${{r}}</span>`).join('')}}
                </div>
                ` : ''}}
                
                <div class="card-caption">
                    <div class="caption-label">Caption Text (editable):</div>
                    <textarea class="caption-input" 
                              data-pair-id="${{pair.pair_id}}"
                              onchange="onCaptionChange('${{pair.pair_id}}', this.value)"
                    >${{pair.caption_text || ''}}</textarea>
                </div>
                
                <div class="card-actions">
                    <button class="btn-exclude" onclick="toggleExclude('${{pair.pair_id}}')">
                        ${{excludedPairs.has(pair.pair_id) ? '↩️ Restore' : '🚫 Exclude'}}
                    </button>
                    <button class="btn-approve" onclick="approveCard('${{pair.pair_id}}')">
                        ✓ Approve
                    </button>
                </div>
            `;
            
            return card;
        }}
        
        function onCaptionChange(pairId, newValue) {{
            const pair = pairs.find(p => p.pair_id === pairId);
            if (pair) {{
                pair.caption_text = newValue;
                pair.edited = true;
                editedPairs.add(pairId);
                updateStats();
                
                const card = document.querySelector(`.pair-card[data-pair-id="${{pairId}}"]`);
                if (card) card.classList.add('edited');
            }}
        }}
        
        function toggleExclude(pairId) {{
            if (excludedPairs.has(pairId)) {{
                excludedPairs.delete(pairId);
            }} else {{
                excludedPairs.add(pairId);
            }}
            renderPairs();
            updateStats();
        }}
        
        function approveCard(pairId) {{
            const pair = pairs.find(p => p.pair_id === pairId);
            if (pair) {{
                pair.needs_review = false;
                pair.approved = true;
            }}
            renderPairs();
            updateStats();
            showToast('Pair approved!');
        }}
        
        function filterPairs(filter) {{
            currentFilter = filter;
            document.querySelectorAll('.filter-tab').forEach(tab => {{
                tab.classList.toggle('active', tab.dataset.filter === filter);
            }});
            renderPairs();
        }}
        
        function updateStats() {{
            document.getElementById('stat-total').textContent = pairs.length;
            document.getElementById('stat-review').textContent = pairs.filter(p => p.needs_review).length;
            document.getElementById('stat-edited').textContent = editedPairs.size;
            document.getElementById('stat-excluded').textContent = excludedPairs.size;
        }}
        
        function getExportData() {{
            return {{
                version: "2.0",
                exported_at: new Date().toISOString(),
                total_pairs: pairs.length,
                excluded_count: excludedPairs.size,
                edited_count: editedPairs.size,
                pairs: pairs.map(p => ({{
                    pair_id: p.pair_id,
                    order_index: p.order_index,
                    caption_text: p.caption_text,
                    excluded: excludedPairs.has(p.pair_id),
                    edited: editedPairs.has(p.pair_id),
                    approved: p.approved || false,
                    original_confidence: p.confidence,
                    original_needs_review: initialPairs.find(ip => ip.pair_id === p.pair_id)?.needs_review || false,
                }}))
            }};
        }}
        
        function exportJSON() {{
            const data = getExportData();
            document.getElementById('json-output').textContent = JSON.stringify(data, null, 2);
            document.getElementById('export-modal').classList.add('active');
        }}
        
        function closeModal() {{
            document.getElementById('export-modal').classList.remove('active');
        }}
        
        function copyJSON() {{
            const data = getExportData();
            navigator.clipboard.writeText(JSON.stringify(data, null, 2))
                .then(() => showToast('Copied to clipboard!'))
                .catch(() => showToast('Copy failed', true));
        }}
        
        function downloadJSON() {{
            const data = getExportData();
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `review_feedback_${{new Date().toISOString().split('T')[0]}}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('Downloaded!');
        }}
        
        function resetAll() {{
            if (confirm('Reset all changes? This will restore original values.')) {{
                pairs = JSON.parse(JSON.stringify(initialPairs));
                editedPairs.clear();
                excludedPairs.clear();
                renderPairs();
                updateStats();
                showToast('All changes reset');
            }}
        }}
        
        function showToast(message, isError = false) {{
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast' + (isError ? ' error' : '');
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }}
        
        // Close modal on escape
        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape') closeModal();
        }});
        
        // Close modal on outside click
        document.getElementById('export-modal').addEventListener('click', e => {{
            if (e.target.id === 'export-modal') closeModal();
        }});
    </script>
</body>
</html>'''


def generate_review_from_result_pairs(
    result_pairs_path: str | Path,
    stage_dir: str | Path,
    output_html_path: str | Path,
) -> Path:
    """Generate review HTML from result_pairs.json file.
    
    Args:
        result_pairs_path: Path to result_pairs.json
        stage_dir: Path to stage directory containing cropped images
        output_html_path: Where to save the HTML file
    
    Returns:
        Path to generated HTML file
    """
    import json
    
    result_pairs_path = Path(result_pairs_path)
    stage_dir = Path(stage_dir)
    output_html_path = Path(output_html_path)
    
    with open(result_pairs_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    # Handle both 1.0 and 2.0 schema
    pages = result_data.get('pages', [result_data] if 'pairs' in result_data else [])
    
    all_pairs = []
    for page in pages:
        for pair in page.get('pairs', []):
            all_pairs.append({
                'pair_id': pair.get('pair_id', ''),
                'order_index': pair.get('order_index', len(all_pairs)),
                'picture_path': str(stage_dir / pair.get('picture_path', '')) if pair.get('picture_path') else '',
                'text_path': str(stage_dir / pair.get('text_path', '')) if pair.get('text_path') else None,
                'caption_text': pair.get('caption_text', ''),
                'confidence': pair.get('confidence', 0.5),
                'needs_review': pair.get('needs_review', False),
                'reasons': pair.get('reasons', []),
            })
    
    return generate_review_html(
        pairs=all_pairs,
        output_path=output_html_path,
        title="Flashcard Pair Review",
    )


if __name__ == "__main__":
    # Quick test with sample data
    sample_pairs = [
        {
            'pair_id': 'test_001',
            'order_index': 0,
            'picture_path': '',
            'text_path': '',
            'caption_text': 'Sample caption 1',
            'confidence': 0.95,
            'needs_review': False,
            'reasons': [],
        },
        {
            'pair_id': 'test_002',
            'order_index': 1,
            'picture_path': '',
            'text_path': '',
            'caption_text': 'Sample caption that needs review',
            'confidence': 0.45,
            'needs_review': True,
            'reasons': ['Low confidence', 'Text overlap detected'],
        },
    ]
    
    output = generate_review_html(
        pairs=sample_pairs,
        output_path='test_review.html',
        title='Test Review',
    )
    print(f"Generated: {output}")
