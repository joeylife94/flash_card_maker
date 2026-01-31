"""Canny 에지 기반 영역 감지 - 상세 분석"""
from PIL import Image, ImageDraw
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

# Canny 에지 검출
edges = cv2.Canny(gray, 30, 100)

# 에지 팽창해서 영역 연결
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=3)

# 모폴로지 닫기 연산으로 영역 채우기
kernel_big = np.ones((15, 15), np.uint8)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_big)

# 컨투어 찾기
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

page_area = H * W
regions = []

for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    
    # 필터링 (더 낮은 임계값)
    if area < page_area * 0.001:  # 0.1% 이상
        continue
    if area > page_area * 0.1:  # 10% 이하
        continue
    if w < 80 or h < 80:
        continue
    
    # 가장자리 제외
    if x < 50 or y < 50 or x + w > W - 50 or y + h > H - 50:
        continue
    
    aspect = w / h
    if aspect < 0.3 or aspect > 3.0:
        continue
    
    regions.append({
        'bbox': (x, y, x+w, y+h),
        'area': area,
        'aspect': aspect
    })

# Y좌표 기준 정렬
regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

print(f"\n감지된 영역: {len(regions)}개")

# 그림/텍스트 분류
pictures = []
texts = []

for i, r in enumerate(regions):
    x0, y0, x1, y1 = r['bbox']
    w, h = x1-x0, y1-y0
    pct = r['area'] / page_area * 100
    
    # 그림: 면적 0.25% 이상, 종횡비 0.5~2.0
    is_pic = pct > 0.25 and 0.5 < r['aspect'] < 2.0
    typ = 'PIC' if is_pic else 'TXT'
    
    if is_pic:
        pictures.append(r)
    else:
        texts.append(r)
    
    print(f"{i+1:2d}. {typ} ({x0:4d},{y0:4d}) {w:3d}x{h:3d} ratio={pct:.2f}% asp={r['aspect']:.2f}")

print(f"\n그림: {len(pictures)}개, 텍스트: {len(texts)}개")

# 시각화
img_vis = img.copy()
draw = ImageDraw.Draw(img_vis)
for i, r in enumerate(regions):
    x0, y0, x1, y1 = r['bbox']
    pct = r['area'] / page_area * 100
    is_pic = pct > 0.25 and 0.5 < r['aspect'] < 2.0
    color = 'blue' if is_pic else 'red'
    draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
    draw.text((x0+5, y0+5), str(i+1), fill=color)

img_vis.save('./debug_canny.jpg')
print(f"\n시각화 저장: debug_canny.jpg (파랑=그림, 빨강=텍스트)")
