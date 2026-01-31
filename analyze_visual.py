"""이미지 레이아웃 상세 분석"""
from PIL import Image
import numpy as np
import cv2

# 이미지 로드
img = Image.open('./Images/20260124_103828.jpg')
img_array = np.array(img.convert('RGB'))
H, W = img_array.shape[:2]

print(f"이미지 크기: {W}x{H}")
print(f"총 픽셀: {W*H:,}")

# 그레이스케일
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# 엣지 검출
edges = cv2.Canny(gray, 50, 150)
print(f"엣지 픽셀: {np.sum(edges > 0):,} ({np.sum(edges > 0) / edges.size * 100:.2f}%)")

# 이미지의 히스토그램 분석
print("\n=== 밝기 히스토그램 ===")
hist, bins = np.histogram(gray.ravel(), bins=10, range=(0, 256))
for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
    pct = count / gray.size * 100
    print(f"  {int(bin_start):3d}-{int(bins[i+1]):3d}: {pct:5.1f}% {'*' * int(pct/2)}")

# 배경색 분석 (가장 많은 색)
unique, counts = np.unique(gray.reshape(-1), return_counts=True)
top_colors = sorted(zip(counts, unique), reverse=True)[:5]
print("\n=== 가장 흔한 밝기값 ===")
for count, color in top_colors:
    pct = count / gray.size * 100
    print(f"  밝기 {color}: {pct:.1f}%")

# 가장 흔한 색이 배경이라고 가정
bg_brightness = top_colors[0][1]
print(f"\n추정 배경 밝기: {bg_brightness}")

# 배경이 아닌 영역 마스크
fg_mask = np.abs(gray.astype(float) - bg_brightness) > 30
print(f"전경 픽셀: {np.sum(fg_mask):,} ({np.sum(fg_mask) / fg_mask.size * 100:.2f}%)")

# 컨투어 찾기
fg_mask_uint8 = (fg_mask * 255).astype(np.uint8)
contours, _ = cv2.findContours(fg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"컨투어 수: {len(contours)}")

# 큰 컨투어만 필터링
min_area = (W * H) * 0.001  # 0.1%
big_contours = [c for c in contours if cv2.contourArea(c) > min_area]
print(f"큰 컨투어 (>{min_area:.0f}px): {len(big_contours)}")

# 바운딩 박스 출력
print("\n=== 주요 컨투어 바운딩 박스 ===")
bboxes = []
for c in big_contours:
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    bboxes.append((y, x, x+w, y+h, area))

bboxes.sort()  # Y 순으로 정렬
for i, (y, x0, x1, y1, area) in enumerate(bboxes[:20]):
    print(f"  [{i:2d}] ({x0:4d}, {y:4d}) - ({x1:4d}, {y1:4d}) area={area:.0f}")

# 그리드 패턴 분석
print("\n=== Y 좌표 클러스터링 (행 분석) ===")
y_coords = [b[0] for b in bboxes]
if y_coords:
    y_sorted = sorted(set(y_coords))
    print(f"고유 Y 좌표 수: {len(y_sorted)}")
    
    # 행 그룹화
    row_gap = H // 20  # 5% of height
    rows = []
    current_row = [bboxes[0]]
    
    for b in sorted(bboxes)[1:]:
        if b[0] - current_row[-1][0] < row_gap:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
    rows.append(current_row)
    
    print(f"추정 행 수: {len(rows)}")
    for i, row in enumerate(rows[:8]):
        y_avg = sum(b[0] for b in row) // len(row)
        print(f"  Row {i+1} (y~{y_avg}): {len(row)} 영역")
