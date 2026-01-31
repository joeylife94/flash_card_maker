"""이미지 레이아웃 분석 - 그리드 패턴 찾기"""
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# 첫 번째 이미지 분석
img_path = list(Path('./Images').glob('*'))[0]
print(f"분석 대상: {img_path.name}")

img = Image.open(img_path).convert('RGB')
arr = np.array(img)
H, W = arr.shape[:2]

gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
bg_brightness = 188

# 배경과 다른 영역 마스크
diff = np.abs(gray.astype(float) - bg_brightness)
fg_mask = (diff > 15).astype(np.uint8) * 255

# Y축 프로젝션 (수평선 찾기)
y_proj = np.sum(fg_mask, axis=1)
# X축 프로젝션 (수직선 찾기)  
x_proj = np.sum(fg_mask, axis=0)

# 콘텐츠가 있는 Y 영역 찾기
y_threshold = W * 10  # 10픽셀 너비 이상
y_content = y_proj > y_threshold

# 연속된 콘텐츠 영역 찾기
y_regions = []
in_region = False
start_y = 0

for y in range(H):
    if y_content[y] and not in_region:
        in_region = True
        start_y = y
    elif not y_content[y] and in_region:
        in_region = False
        if y - start_y > 100:  # 높이 100 이상인 영역만
            y_regions.append((start_y, y))

if in_region and H - start_y > 100:
    y_regions.append((start_y, H))

print(f"\n세로 방향 콘텐츠 영역: {len(y_regions)}개")
for i, (y0, y1) in enumerate(y_regions):
    print(f"  {i+1}. Y={y0} ~ {y1} (높이 {y1-y0})")

# 각 Y 영역에서 X 방향 분석
print("\n각 행의 콘텐츠 영역:")
total_boxes = 0

for row_idx, (y0, y1) in enumerate(y_regions):
    row_slice = fg_mask[y0:y1, :]
    x_proj_row = np.sum(row_slice, axis=0)
    x_threshold = (y1 - y0) * 10
    x_content = x_proj_row > x_threshold
    
    # X 방향 영역 찾기
    x_regions = []
    in_region = False
    start_x = 0
    
    for x in range(W):
        if x_content[x] and not in_region:
            in_region = True
            start_x = x
        elif not x_content[x] and in_region:
            in_region = False
            if x - start_x > 80:  # 너비 80 이상
                x_regions.append((start_x, x))
    
    if in_region and W - start_x > 80:
        x_regions.append((start_x, W))
    
    print(f"  행 {row_idx+1} (Y={y0}~{y1}): {len(x_regions)}개 영역")
    total_boxes += len(x_regions)

print(f"\n총 예상 박스: {total_boxes}개")
