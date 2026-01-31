"""감지된 영역 분석 및 시각화 - 새로운 분류 기준"""
from PIL import Image, ImageDraw
import numpy as np
import cv2
from pathlib import Path

img_path = list(Path('./Images').glob('*'))[0]
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
H, W = arr.shape[:2]
page_area = H * W

gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
bg_brightness = 188

diff = np.abs(gray.astype(float) - bg_brightness)
fg_mask = (diff > 15).astype(np.uint8) * 255  # 임계값 15
kernel = np.ones((3, 3), np.uint8)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

regions = []
for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    # 더 작은 영역도 감지 (0.02%)
    if area > page_area * 0.0002 and w >= 50 and h >= 50:
        # 전체 페이지 제외 (30%)
        if area > page_area * 0.3:
            continue
        aspect = w/h
        if 0.15 < aspect < 6.0:
            regions.append({
                'bbox': (x, y, x+w, y+h), 
                'area': area, 
                'aspect': aspect, 
                'ratio': area/page_area
            })

# 면적순 정렬
regions.sort(key=lambda r: r['area'], reverse=True)

print(f'총 {len(regions)}개 영역')
print()
print('면적순 영역:')
pictures = []
texts = []

for i, r in enumerate(regions):
    x0, y0, x1, y1 = r['bbox']
    w, h = x1-x0, y1-y0
    # 새 기준: 면적 0.08% 이상, 종횡비 0.35~2.8
    is_pic = r['ratio'] > 0.0008 and 0.35 < r['aspect'] < 2.8
    typ = 'PIC' if is_pic else 'TXT'
    pct = r['ratio'] * 100
    print(f"{i+1:2d}. {typ} ({x0:4d},{y0:4d})-({x1:4d},{y1:4d}) {w:3d}x{h:3d} ratio={pct:.3f}%")
    
    if is_pic:
        pictures.append(r)
    else:
        texts.append(r)

print()
print(f'그림: {len(pictures)}개, 텍스트: {len(texts)}개')

# 시각화
draw = ImageDraw.Draw(img)
for i, r in enumerate(regions):
    x0, y0, x1, y1 = r['bbox']
    is_pic = r['ratio'] > 0.0008 and 0.35 < r['aspect'] < 2.8
    color = 'blue' if is_pic else 'red'
    draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
    draw.text((x0+5, y0+5), str(i+1), fill=color)

img.save('./debug_regions.jpg')
print()
print('시각화 저장: debug_regions.jpg')
