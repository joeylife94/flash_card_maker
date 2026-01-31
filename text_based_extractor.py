"""텍스트 기반 그림-단어 쌍 추출기

전략:
1. OCR로 텍스트 블록 위치 찾기
2. 각 텍스트 블록의 위쪽 영역에서 그림 찾기
3. 그림-텍스트 쌍 생성
"""
from PIL import Image, ImageDraw
import numpy as np
import easyocr
import cv2
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TextBlock:
    text: str
    bbox: tuple  # (x0, y0, x1, y1)
    confidence: float


@dataclass
class PicturePair:
    text: str
    text_bbox: tuple
    picture_bbox: tuple
    confidence: float


def find_picture_above_text(img_array: np.ndarray, text_bbox: tuple, 
                            search_height: int = 300, min_content_ratio: float = 0.1) -> tuple | None:
    """텍스트 위쪽에서 그림 영역 찾기
    
    Args:
        img_array: RGB 이미지 배열
        text_bbox: (x0, y0, x1, y1) 텍스트 영역
        search_height: 위쪽으로 검색할 높이
        min_content_ratio: 최소 비배경 픽셀 비율
    
    Returns:
        그림 bbox (x0, y0, x1, y1) 또는 None
    """
    x0, y0, x1, y1 = text_bbox
    text_width = x1 - x0
    text_height = y1 - y0
    
    # 텍스트 위쪽 영역 정의
    search_x0 = max(0, x0 - text_width // 2)  # 좌우로 확장
    search_x1 = min(img_array.shape[1], x1 + text_width // 2)
    search_y0 = max(0, y0 - search_height)
    search_y1 = y0 - 5  # 텍스트 바로 위
    
    if search_y1 <= search_y0:
        return None
    
    # 검색 영역 추출
    region = img_array[search_y0:search_y1, search_x0:search_x1]
    if region.size == 0:
        return None
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    
    # 배경 제거 (흰색 또는 밝은 색 제거)
    # 적응형 임계값 사용
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 비배경 픽셀 비율 확인
    content_ratio = np.sum(binary > 0) / binary.size
    if content_ratio < min_content_ratio:
        return None
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 가장 큰 컨투어의 바운딩 박스
    all_points = np.vstack(contours)
    rx, ry, rw, rh = cv2.boundingRect(all_points)
    
    # 너무 작으면 무시
    if rw < 30 or rh < 30:
        return None
    
    # 전역 좌표로 변환
    pic_x0 = search_x0 + rx
    pic_y0 = search_y0 + ry
    pic_x1 = pic_x0 + rw
    pic_y1 = pic_y0 + rh
    
    return (pic_x0, pic_y0, pic_x1, pic_y1)


def extract_pairs_from_image(image_path: str | Path, lang: str = "en,ko") -> list[PicturePair]:
    """이미지에서 그림-텍스트 쌍 추출"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    H, W = img_array.shape[:2]
    
    print(f"이미지 크기: {W}x{H}")
    
    # OCR로 텍스트 찾기
    langs = [l.strip() for l in lang.split(',')]
    reader = easyocr.Reader(langs, gpu=False)
    results = reader.readtext(img_array)
    
    print(f"텍스트 블록: {len(results)}")
    
    # 텍스트 블록 정리
    text_blocks = []
    for item in results:
        bbox_points, text, conf = item
        if len(text.strip()) < 2:  # 너무 짧은 텍스트 무시
            continue
        if conf < 0.3:  # 낮은 신뢰도 무시
            continue
            
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        
        text_blocks.append(TextBlock(text=text, bbox=bbox, confidence=conf))
    
    print(f"유효한 텍스트: {len(text_blocks)}")
    
    # 각 텍스트에 대해 위쪽 그림 찾기
    pairs = []
    used_picture_regions = []  # 중복 방지
    
    for tb in text_blocks:
        pic_bbox = find_picture_above_text(img_array, tb.bbox)
        
        if pic_bbox is None:
            continue
        
        # 중복 체크 (이미 사용된 영역과 겹치는지)
        is_duplicate = False
        for used in used_picture_regions:
            # IoU 계산
            ix0 = max(pic_bbox[0], used[0])
            iy0 = max(pic_bbox[1], used[1])
            ix1 = min(pic_bbox[2], used[2])
            iy1 = min(pic_bbox[3], used[3])
            
            if ix1 > ix0 and iy1 > iy0:
                intersection = (ix1 - ix0) * (iy1 - iy0)
                area1 = (pic_bbox[2] - pic_bbox[0]) * (pic_bbox[3] - pic_bbox[1])
                if intersection / area1 > 0.5:
                    is_duplicate = True
                    break
        
        if is_duplicate:
            continue
        
        used_picture_regions.append(pic_bbox)
        pairs.append(PicturePair(
            text=tb.text,
            text_bbox=tb.bbox,
            picture_bbox=pic_bbox,
            confidence=tb.confidence
        ))
    
    return pairs


def visualize_pairs(image_path: str | Path, pairs: list[PicturePair], output_path: str | Path):
    """추출된 쌍 시각화"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    for i, pair in enumerate(pairs):
        # 그림 영역 (파란색)
        px0, py0, px1, py1 = pair.picture_bbox
        draw.rectangle([px0, py0, px1, py1], outline="blue", width=3)
        
        # 텍스트 영역 (빨간색)
        tx0, ty0, tx1, ty1 = pair.text_bbox
        draw.rectangle([tx0, ty0, tx1, ty1], outline="red", width=2)
        
        # 연결선 (초록색)
        draw.line([(px0 + px1) // 2, py1, (tx0 + tx1) // 2, ty0], fill="green", width=2)
        
        # 번호
        draw.text((px0, py0 - 20), f"#{i+1}", fill="blue")
    
    img.save(output_path)
    print(f"시각화 저장: {output_path}")


if __name__ == "__main__":
    # 테스트
    image_path = "./Images/20260124_103828.jpg"
    
    print("=" * 50)
    print("그림-텍스트 쌍 추출 시작")
    print("=" * 50)
    
    pairs = extract_pairs_from_image(image_path, lang="en,ko")
    
    print(f"\n총 {len(pairs)} 쌍 추출됨:")
    for i, p in enumerate(pairs):
        print(f"  [{i+1}] {p.text[:20]} - 그림: {p.picture_bbox}")
    
    # 시각화
    visualize_pairs(image_path, pairs, "./test_pairs_visualization.jpg")
