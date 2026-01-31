"""통합 그림-텍스트 쌍 추출기 v2

전략:
1. 전체 이미지에서 OCR로 모든 텍스트 찾기
2. 전체 이미지에서 컨투어로 모든 그림 영역 찾기  
3. 가장 가까운 그림-텍스트 매칭
"""
from PIL import Image, ImageDraw
import numpy as np
import cv2
import easyocr
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import hashlib


@dataclass 
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    
    def distance_to(self, other: 'BBox') -> float:
        c1 = self.center
        c2 = other.center
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5


@dataclass 
class TextBlock:
    text: str
    bbox: BBox
    confidence: float


@dataclass
class PictureRegion:
    bbox: BBox
    area: int


@dataclass
class FlashcardPair:
    pair_id: str
    picture_bbox: Tuple[int, int, int, int]
    text_bbox: Tuple[int, int, int, int]
    text: str
    confidence: float


class UnifiedExtractor:
    """통합 그림-텍스트 추출기"""
    
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._ocr = None
    
    def _init_ocr(self):
        if self._ocr is None:
            langs = [l.strip() for l in self.lang.split(',')]
            self._ocr = easyocr.Reader(langs, gpu=False)
        return self._ocr
    
    def detect_all_text(self, img_array: np.ndarray) -> List[TextBlock]:
        """이미지에서 모든 텍스트 블록 감지"""
        ocr = self._init_ocr()
        results = ocr.readtext(img_array)
        
        blocks = []
        for item in results:
            bbox_points, text, conf = item
            if len(text.strip()) < 2:
                continue
            if conf < 0.3:
                continue
                
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = BBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            
            blocks.append(TextBlock(text=text.strip(), bbox=bbox, confidence=conf))
        
        return blocks
    
    def detect_all_pictures(self, img_array: np.ndarray,
                           min_area_ratio: float = 0.001,
                           max_area_ratio: float = 0.3) -> List[PictureRegion]:
        """이미지에서 모든 그림 영역 감지"""
        H, W = img_array.shape[:2]
        page_area = H * W
        min_area = page_area * min_area_ratio
        max_area = page_area * max_area_ratio
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 배경색 추정
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        bg_brightness = np.argmax(hist)
        
        # 배경과 다른 영역
        diff = np.abs(gray.astype(float) - bg_brightness)
        fg_mask = (diff > 25).astype(np.uint8) * 255
        
        # 모폴로지
        kernel = np.ones((7, 7), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 종횡비 필터
            aspect = w / h if h > 0 else 0
            if aspect < 0.25 or aspect > 4.0:
                continue
            
            regions.append(PictureRegion(
                bbox=BBox(x, y, x + w, y + h),
                area=int(area)
            ))
        
        return regions
    
    def match_pictures_to_text(self, pictures: List[PictureRegion], 
                               texts: List[TextBlock],
                               max_distance: float = 500) -> List[FlashcardPair]:
        """가장 가까운 그림-텍스트 매칭"""
        pairs = []
        used_texts = set()
        
        for pic in pictures:
            # 이 그림과 가장 가까운 텍스트 찾기
            best_text = None
            best_distance = float('inf')
            
            for i, text in enumerate(texts):
                if i in used_texts:
                    continue
                
                dist = pic.bbox.distance_to(text.bbox)
                if dist < best_distance and dist < max_distance:
                    best_distance = dist
                    best_text = (i, text)
            
            if best_text:
                idx, text = best_text
                used_texts.add(idx)
                
                pair_id = hashlib.sha1(
                    f"{pic.bbox.x0}_{pic.bbox.y0}_{text.text}".encode()
                ).hexdigest()[:12]
                
                pairs.append(FlashcardPair(
                    pair_id=pair_id,
                    picture_bbox=(pic.bbox.x0, pic.bbox.y0, pic.bbox.x1, pic.bbox.y1),
                    text_bbox=(text.bbox.x0, text.bbox.y0, text.bbox.x1, text.bbox.y1),
                    text=text.text,
                    confidence=text.confidence
                ))
        
        return pairs
    
    def extract_pairs(self, image_path: str | Path) -> List[FlashcardPair]:
        """이미지에서 그림-텍스트 쌍 추출"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        H, W = img_array.shape[:2]
        
        print(f"이미지 크기: {W}x{H}")
        
        print("[1/4] 텍스트 감지 (OCR)...")
        texts = self.detect_all_text(img_array)
        print(f"      {len(texts)} 텍스트 블록 발견")
        for t in texts[:10]:
            print(f"        - '{t.text[:20]}' ({t.bbox.x0}, {t.bbox.y0})")
        
        print("[2/4] 그림 영역 감지...")
        pictures = self.detect_all_pictures(img_array)
        print(f"      {len(pictures)} 그림 영역 발견")
        for p in pictures[:10]:
            print(f"        - ({p.bbox.x0}, {p.bbox.y0}) - ({p.bbox.x1}, {p.bbox.y1})")
        
        print("[3/4] 그림-텍스트 매칭...")
        pairs = self.match_pictures_to_text(pictures, texts)
        print(f"      {len(pairs)} 쌍 매칭됨")
        
        print("[4/4] 완료!")
        return pairs
    
    def visualize(self, image_path: str | Path, pairs: List[FlashcardPair], 
                  output_path: str | Path):
        """결과 시각화"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for i, pair in enumerate(pairs):
            # 그림 (파란색)
            draw.rectangle(pair.picture_bbox, outline="blue", width=4)
            
            # 텍스트 (빨간색)
            draw.rectangle(pair.text_bbox, outline="red", width=3)
            
            # 연결선 (초록색)
            pc = ((pair.picture_bbox[0] + pair.picture_bbox[2]) // 2,
                  (pair.picture_bbox[1] + pair.picture_bbox[3]) // 2)
            tc = ((pair.text_bbox[0] + pair.text_bbox[2]) // 2,
                  (pair.text_bbox[1] + pair.text_bbox[3]) // 2)
            draw.line([pc, tc], fill="green", width=2)
            
            # 번호
            draw.text((pair.picture_bbox[0], pair.picture_bbox[1] - 25), 
                     f"#{i+1}: {pair.text[:12]}", fill="blue")
        
        img.save(output_path)
        print(f"시각화 저장: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("통합 그림-텍스트 추출기 v2")
    print("=" * 60)
    
    # 독일어 모델 사용
    extractor = UnifiedExtractor(lang="de")
    
    image_path = "./Images/20260124_103828.jpg"
    pairs = extractor.extract_pairs(image_path)
    
    print(f"\n=== 결과: {len(pairs)} 쌍 ===")
    for i, p in enumerate(pairs):
        print(f"  [{i+1}] '{p.text}' - 그림: {p.picture_bbox}")
    
    extractor.visualize(image_path, pairs, "./test_unified_v2.jpg")
