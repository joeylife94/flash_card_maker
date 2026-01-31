"""이미지 레이아웃 분석 - 그림-텍스트 쌍 찾기"""
from PIL import Image
import numpy as np
import easyocr

img = Image.open('./Images/20260124_103828.jpg')
img_array = np.array(img.convert('RGB'))
H, W = img_array.shape[:2]

print(f"이미지 크기: {W}x{H}")

# OCR로 텍스트 찾기
reader = easyocr.Reader(['en', 'ko'], gpu=False)
results = reader.readtext(img_array)

print(f"\n총 텍스트 블록: {len(results)}")

# 텍스트 위치 분석
texts_with_pos = []
for item in results:
    bbox, text, conf = item
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    center_y = (y0 + y1) / 2
    center_x = (x0 + x1) / 2
    texts_with_pos.append({
        'text': text,
        'bbox': (x0, y0, x1, y1),
        'center': (center_x, center_y),
        'conf': conf
    })

# Y 위치로 정렬
texts_with_pos.sort(key=lambda t: t['center'][1])

print("\n=== 텍스트 블록 (Y 순서) ===")
for i, t in enumerate(texts_with_pos[:30]):
    cx, cy = t['center']
    print(f"  [{i:2d}] y={int(cy):4d}, x={int(cx):4d}: \"{t['text'][:25]}\"")

# 행(Row) 분석 - Y 좌표 클러스터링
print("\n=== 행(Row) 분석 ===")
row_threshold = 100  # 같은 행으로 간주할 Y 간격

rows = []
current_row = [texts_with_pos[0]]
for t in texts_with_pos[1:]:
    if t['center'][1] - current_row[-1]['center'][1] < row_threshold:
        current_row.append(t)
    else:
        rows.append(current_row)
        current_row = [t]
rows.append(current_row)

print(f"총 {len(rows)} 행 발견")
for i, row in enumerate(rows[:10]):
    y_avg = sum(t['center'][1] for t in row) / len(row)
    texts = [t['text'][:15] for t in sorted(row, key=lambda x: x['center'][0])]
    print(f"  Row {i+1} (y~{int(y_avg)}): {len(row)}개 - {texts}")

# 예상되는 그리드 구조 추정
print("\n=== 그리드 구조 추정 ===")
# 각 행의 텍스트 수로 열 수 추정
col_counts = [len(row) for row in rows]
most_common_cols = max(set(col_counts), key=col_counts.count)
print(f"가장 흔한 열 수: {most_common_cols}")
print(f"추정 그리드: {len(rows)} 행 x {most_common_cols} 열")
print(f"예상 쌍 수: {len(rows) * most_common_cols // 2} (각 셀에 그림+텍스트)")
