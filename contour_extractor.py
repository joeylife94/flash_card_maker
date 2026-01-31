"""컨투어 기반 그림-텍스트 쌍 추출기

핵심 전략:
1. 배경색과 다른 영역을 컨투어로 찾기
2. 큰 컨투어만 필터링 (실제 그림)
3. 각 그림 아래쪽에서 텍스트 OCR
4. 쌍 생성
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import easyocr
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import hashlib


@dataclass 
class ContentRegion:
    """감지된 콘텐츠 영역"""
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    area: int
    is_picture: bool = True
    text: str = ""
    confidence: float = 0.0


@dataclass
class FlashcardPair:
    """그림-텍스트 쌍"""
    pair_id: str
    picture_bbox: Tuple[int, int, int, int]
    text_bbox: Optional[Tuple[int, int, int, int]]
    text: str
    confidence: float


class ContourBasedExtractor:
    """컨투어 기반 그림-텍스트 추출기"""
    
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._ocr = None
    
    def _init_ocr(self):
        if self._ocr is None:
            langs = [l.strip() for l in self.lang.split(',')]
            self._ocr = easyocr.Reader(langs, gpu=False)
        return self._ocr
    
    def detect_content_regions(self, img_array: np.ndarray, 
                                min_area_ratio: float = 0.001,
                                max_area_ratio: float = 0.5) -> List[ContentRegion]:
        """이미지에서 콘텐츠 영역 감지
        
        전략: 배경색과 다른 영역을 컨투어로 찾기
        """
        H, W = img_array.shape[:2]
        page_area = H * W
        min_area = page_area * min_area_ratio
        max_area = page_area * max_area_ratio
        
        # 그레이스케일
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 배경색 추정 (가장 흔한 밝기)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        bg_brightness = np.argmax(hist)
        
        # 배경과 다른 영역 마스크
        diff = np.abs(gray.astype(float) - bg_brightness)
        fg_mask = (diff > 25).astype(np.uint8) * 255
        
        # 모폴로지 연산으로 노이즈 제거 및 영역 연결
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 면적 필터
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 종횡비 필터 (너무 길쭉한 것 제외)
            aspect = w / h if h > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue
            
            regions.append(ContentRegion(
                bbox=(x, y, x + w, y + h),
                area=int(area),
            ))
        
        # Y 좌표로 정렬 (위에서 아래로)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        return regions
    
    def find_text_below_region(self, img_array: np.ndarray, 
                                region_bbox: Tuple[int, int, int, int],
                                search_height: int = 200) -> Tuple[str, Optional[Tuple[int, int, int, int]], float]:
        """영역 아래에서 텍스트 찾기"""
        x0, y0, x1, y1 = region_bbox
        H, W = img_array.shape[:2]
        
        # 검색 영역: 영역 아래쪽
        search_y0 = y1 + 5
        search_y1 = min(H, y1 + search_height)
        
        # X 범위는 영역과 비슷하게, 약간 확장
        margin = (x1 - x0) // 4
        search_x0 = max(0, x0 - margin)
        search_x1 = min(W, x1 + margin)
        
        if search_y1 <= search_y0:
            return "", None, 0.0
        
        # 검색 영역 자르기
        search_region = img_array[search_y0:search_y1, search_x0:search_x1]
        if search_region.size == 0:
            return "", None, 0.0
        
        # OCR
        ocr = self._init_ocr()
        results = ocr.readtext(search_region)
        
        if not results:
            return "", None, 0.0
        
        # 가장 신뢰도 높은 텍스트 선택
        best_result = max(results, key=lambda x: x[2])
        bbox_points, text, conf = best_result
        
        # 전역 좌표로 변환
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        text_bbox = (
            search_x0 + int(min(xs)),
            search_y0 + int(min(ys)),
            search_x0 + int(max(xs)),
            search_y0 + int(max(ys))
        )
        
        return text.strip(), text_bbox, float(conf)
    
    def extract_pairs(self, image_path: str | Path) -> List[FlashcardPair]:
        """이미지에서 그림-텍스트 쌍 추출"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        print(f"[1/3] 콘텐츠 영역 감지...")
        regions = self.detect_content_regions(img_array)
        print(f"      {len(regions)} 영역 발견")
        
        print(f"[2/3] 각 영역 아래 텍스트 찾기...")
        pairs = []
        
        for i, region in enumerate(regions):
            text, text_bbox, conf = self.find_text_below_region(img_array, region.bbox)
            
            if text and len(text) >= 2:  # 최소 2글자
                pair_id = hashlib.sha1(
                    f"{image_path}_{i}_{region.bbox}".encode()
                ).hexdigest()[:12]
                
                pairs.append(FlashcardPair(
                    pair_id=pair_id,
                    picture_bbox=region.bbox,
                    text_bbox=text_bbox,
                    text=text,
                    confidence=conf
                ))
                print(f"      [{i+1}] {text[:20]} (conf={conf:.2f})")
            else:
                print(f"      [{i+1}] (텍스트 없음)")
        
        print(f"[3/3] 완료! {len(pairs)} 쌍 추출")
        return pairs
    
    def visualize(self, image_path: str | Path, pairs: List[FlashcardPair], output_path: str | Path):
        """결과 시각화"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for i, pair in enumerate(pairs):
            # 그림 영역 (파란색)
            px0, py0, px1, py1 = pair.picture_bbox
            draw.rectangle([px0, py0, px1, py1], outline="blue", width=4)
            
            # 텍스트 영역 (빨간색)
            if pair.text_bbox:
                tx0, ty0, tx1, ty1 = pair.text_bbox
                draw.rectangle([tx0, ty0, tx1, ty1], outline="red", width=3)
                
                # 연결선
                draw.line([
                    (px0 + px1) // 2, py1,
                    (tx0 + tx1) // 2, ty0
                ], fill="green", width=2)
            
            # 번호와 텍스트
            draw.text((px0, py0 - 25), f"#{i+1}: {pair.text[:15]}", fill="blue")
        
        img.save(output_path)
        print(f"시각화 저장: {output_path}")


def process_all_images(images_dir: str | Path, output_dir: str | Path, lang: str = "en"):
    """폴더의 모든 이미지 처리"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = ContourBasedExtractor(lang=lang)
    
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    ])
    
    print(f"총 {len(image_files)} 이미지 파일")
    
    all_pairs = []
    for img_file in image_files:
        print(f"\n{'='*50}")
        print(f"처리 중: {img_file.name}")
        print(f"{'='*50}")
        
        pairs = extractor.extract_pairs(img_file)
        all_pairs.extend(pairs)
        
        # 시각화 저장
        vis_path = output_dir / f"vis_{img_file.stem}.jpg"
        extractor.visualize(img_file, pairs, vis_path)
    
    print(f"\n{'='*50}")
    print(f"전체 결과: {len(all_pairs)} 쌍 추출")
    print(f"{'='*50}")
    
    return all_pairs


if __name__ == "__main__":
    # 단일 이미지 테스트
    print("=" * 60)
    print("컨투어 기반 그림-텍스트 쌍 추출")
    print("=" * 60)
    
    extractor = ContourBasedExtractor(lang="en")
    
    image_path = "./Images/20260124_103828.jpg"
    pairs = extractor.extract_pairs(image_path)
    
    print(f"\n총 {len(pairs)} 쌍:")
    for i, p in enumerate(pairs):
        print(f"  [{i+1}] {p.text} - 그림: {p.picture_bbox}")
    
    # 시각화
    extractor.visualize(image_path, pairs, "./test_contour_pairs.jpg")
