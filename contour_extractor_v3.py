"""컨투어 기반 완전 자동 그림-텍스트 쌍 추출기 v3

핵심 전략:
1. 배경과 다른 모든 영역을 컨투어로 감지
2. 큰 영역 = 그림, 작은 영역 = 텍스트로 분류
3. 가까운 그림-텍스트 쌍 매칭
4. OCR은 보조 역할 (없어도 됨)
"""
from PIL import Image, ImageDraw
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import hashlib
import json


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
    def width(self) -> int:
        return self.x1 - self.x0
    
    @property
    def height(self) -> int:
        return self.y1 - self.y0
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def distance_to(self, other: 'BBox') -> float:
        c1 = self.center
        c2 = other.center
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass 
class ContentRegion:
    bbox: BBox
    contour_area: int
    is_picture: bool  # True=그림, False=텍스트
    text: str = ""  # OCR 결과 (있으면)


@dataclass
class FlashcardPair:
    pair_id: str
    picture_bbox: Tuple[int, int, int, int]
    text_bbox: Optional[Tuple[int, int, int, int]]
    text: str  # OCR 결과 또는 빈 문자열
    order_index: int


class ContourPairExtractor:
    """컨투어 기반 그림-텍스트 쌍 추출기"""
    
    def __init__(self):
        self._ocr = None
    
    def _try_init_ocr(self, lang: str = "de"):
        """OCR 초기화 시도 (실패해도 계속 진행)"""
        try:
            import easyocr
            self._ocr = easyocr.Reader([lang], gpu=False)
            return True
        except:
            return False
    
    def detect_all_regions(self, img_array: np.ndarray) -> List[ContentRegion]:
        """이미지에서 모든 콘텐츠 영역 감지 (Canny 에지 기반)"""
        H, W = img_array.shape[:2]
        page_area = H * W
        
        # 그레이스케일
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Canny 에지 검출 (더 많은 에지 감지)
        edges = cv2.Canny(gray, 30, 100)
        
        # 에지 팽창해서 영역 연결
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # 모폴로지 닫기 연산으로 영역 채우기
        kernel_big = np.ones((15, 15), np.uint8)
        fg_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_big)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 너무 작거나 너무 큰 것 제외
            if area < page_area * 0.001:  # 0.1% 이상
                continue
            if area > page_area * 0.1:  # 10% 이하
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BBox(x, y, x + w, y + h)
            
            # 최소 크기 필터 (너무 작은 노이즈 제외)
            if w < 80 or h < 80:
                continue
            
            # 가장자리 영역 제외 (페이지 번호, 마진 등)
            margin = 50
            if x < margin or y < margin:
                continue
            if x + w > W - margin or y + h > H - margin:
                continue
            
            # 종횡비 필터
            aspect = w / h if h > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue
            
            # 그림 vs 텍스트 분류
            # 기준: 면적 비율 & 종횡비
            area_ratio = area / page_area
            is_picture = area_ratio > 0.003 and aspect > 0.4 and aspect < 2.5
            
            regions.append(ContentRegion(
                bbox=bbox,
                contour_area=int(area),
                is_picture=is_picture
            ))
        
        # Y 좌표로 정렬
        regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))
        
        return regions
    
    def classify_regions(self, regions: List[ContentRegion], 
                        page_area: int) -> Tuple[List[ContentRegion], List[ContentRegion]]:
        """영역을 그림과 텍스트로 분류
        
        분류 기준:
        - 그림: 면적 0.18% 이상, 종횡비 0.5~2.0 (정사각형에 가까움)
        - 텍스트: 면적 작거나, 세로로 긴 형태 (종횡비 < 0.5)
        """
        pictures = []
        texts = []
        
        if not regions:
            return [], []
        
        for r in regions:
            aspect = r.bbox.width / r.bbox.height if r.bbox.height > 0 else 0
            area_ratio = r.contour_area / page_area
            
            # 그림 판별: 
            # - 면적 비율 0.18% 이상
            # - 종횡비 0.5 ~ 2.0 (정사각형에 가까움)
            if (area_ratio > 0.0018 and 0.5 < aspect < 2.0):
                r.is_picture = True
                pictures.append(r)
            else:
                r.is_picture = False
                texts.append(r)
        
        return pictures, texts
    
    def match_pairs(self, pictures: List[ContentRegion], 
                   texts: List[ContentRegion],
                   max_distance: float = 600) -> List[FlashcardPair]:
        """가장 가까운 그림-텍스트 매칭"""
        pairs = []
        used_texts = set()
        
        for i, pic in enumerate(pictures):
            best_text = None
            best_dist = float('inf')
            
            for j, txt in enumerate(texts):
                if j in used_texts:
                    continue
                
                # 그림 아래 또는 옆에 있는 텍스트 선호
                dist = pic.bbox.distance_to(txt.bbox)
                
                # 방향 가중치 (아래쪽 텍스트 선호)
                if txt.bbox.center[1] > pic.bbox.center[1]:  # 아래
                    dist *= 0.8
                
                if dist < best_dist and dist < max_distance:
                    best_dist = dist
                    best_text = (j, txt)
            
            if best_text:
                idx, txt = best_text
                used_texts.add(idx)
                
                pair_id = hashlib.sha1(
                    f"{pic.bbox.x0}_{pic.bbox.y0}_{txt.bbox.x0}".encode()
                ).hexdigest()[:12]
                
                pairs.append(FlashcardPair(
                    pair_id=pair_id,
                    picture_bbox=pic.bbox.to_tuple(),
                    text_bbox=txt.bbox.to_tuple(),
                    text=txt.text,
                    order_index=i
                ))
            else:
                # 텍스트 없이 그림만
                pair_id = hashlib.sha1(
                    f"{pic.bbox.x0}_{pic.bbox.y0}_nopair".encode()
                ).hexdigest()[:12]
                
                pairs.append(FlashcardPair(
                    pair_id=pair_id,
                    picture_bbox=pic.bbox.to_tuple(),
                    text_bbox=None,
                    text="",
                    order_index=i
                ))
        
        return pairs
    
    def try_ocr_for_pairs(self, img_array: np.ndarray, pairs: List[FlashcardPair], lang: str = "de"):
        """OCR로 텍스트 영역 인식 시도"""
        if not self._try_init_ocr(lang):
            print("  OCR 초기화 실패 - 텍스트는 나중에 수동 입력 필요")
            return
        
        for pair in pairs:
            if pair.text_bbox:
                x0, y0, x1, y1 = pair.text_bbox
                region = img_array[y0:y1, x0:x1]
                if region.size > 0:
                    try:
                        results = self._ocr.readtext(region)
                        if results:
                            pair.text = " ".join([r[1] for r in results])
                    except:
                        pass
    
    def extract_pairs(self, image_path: str | Path, lang: str = "de") -> List[FlashcardPair]:
        """이미지에서 그림-텍스트 쌍 추출"""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        H, W = img_array.shape[:2]
        page_area = H * W
        
        print(f"이미지: {W}x{H}")
        
        print("[1/4] 모든 영역 감지...")
        regions = self.detect_all_regions(img_array)
        print(f"      {len(regions)} 영역 발견")
        
        print("[2/4] 그림/텍스트 분류...")
        pictures, texts = self.classify_regions(regions, page_area)
        print(f"      그림: {len(pictures)}, 텍스트: {len(texts)}")
        
        print("[3/4] 쌍 매칭...")
        pairs = self.match_pairs(pictures, texts)
        print(f"      {len(pairs)} 쌍 생성")
        
        print("[4/4] OCR 시도...")
        self.try_ocr_for_pairs(img_array, pairs, lang)
        
        return pairs
    
    def visualize(self, image_path: str | Path, pairs: List[FlashcardPair], 
                  output_path: str | Path, show_all_regions: bool = False):
        """결과 시각화"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for i, pair in enumerate(pairs):
            # 그림 (파란색 두꺼운 선)
            draw.rectangle(pair.picture_bbox, outline="blue", width=5)
            
            # 텍스트 (빨간색)
            if pair.text_bbox:
                draw.rectangle(pair.text_bbox, outline="red", width=3)
                
                # 연결선 (초록색)
                pc = ((pair.picture_bbox[0] + pair.picture_bbox[2]) // 2,
                      (pair.picture_bbox[1] + pair.picture_bbox[3]) // 2)
                tc = ((pair.text_bbox[0] + pair.text_bbox[2]) // 2,
                      (pair.text_bbox[1] + pair.text_bbox[3]) // 2)
                draw.line([pc, tc], fill="lime", width=3)
            
            # 번호
            label = f"#{i+1}"
            if pair.text:
                label += f": {pair.text[:15]}"
            draw.text((pair.picture_bbox[0], pair.picture_bbox[1] - 30), 
                     label, fill="blue")
        
        img.save(output_path)
        print(f"시각화: {output_path}")
    
    def save_pairs_json(self, pairs: List[FlashcardPair], output_path: str | Path):
        """쌍 정보를 JSON으로 저장"""
        data = {
            "total_pairs": len(pairs),
            "pairs": [
                {
                    "pair_id": p.pair_id,
                    "order_index": p.order_index,
                    "picture_bbox": p.picture_bbox,
                    "text_bbox": p.text_bbox,
                    "text": p.text,
                    "needs_text_input": not p.text
                }
                for p in pairs
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON 저장: {output_path}")


def process_all_images(images_dir: str | Path, output_dir: str | Path, lang: str = "de"):
    """폴더의 모든 이미지 처리"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = ContourPairExtractor()
    
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    ])
    
    print(f"\n총 {len(image_files)} 이미지 파일\n")
    
    all_results = {}
    total_pairs = 0
    
    for img_file in image_files:
        print(f"{'='*50}")
        print(f"처리: {img_file.name}")
        print(f"{'='*50}")
        
        pairs = extractor.extract_pairs(img_file, lang)
        total_pairs += len(pairs)
        
        # 시각화
        vis_path = output_dir / f"vis_{img_file.stem}.jpg"
        extractor.visualize(img_file, pairs, vis_path)
        
        # JSON
        json_path = output_dir / f"pairs_{img_file.stem}.json"
        extractor.save_pairs_json(pairs, json_path)
        
        all_results[img_file.name] = len(pairs)
        print()
    
    # 전체 요약
    print(f"\n{'='*50}")
    print(f"전체 결과")
    print(f"{'='*50}")
    for name, count in all_results.items():
        print(f"  {name}: {count} 쌍")
    print(f"\n총 {total_pairs} 쌍 추출")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("컨투어 기반 그림-텍스트 쌍 추출기 v3")
    print("=" * 60)
    
    # 단일 이미지 테스트
    extractor = ContourPairExtractor()
    
    image_path = "./Images/20260124_103828.jpg"
    pairs = extractor.extract_pairs(image_path, lang="de")
    
    print(f"\n=== 결과: {len(pairs)} 쌍 ===")
    for i, p in enumerate(pairs):
        text_info = f"'{p.text}'" if p.text else "(텍스트 없음)"
        print(f"  [{i+1}] {text_info}")
        print(f"       그림: {p.picture_bbox}")
        if p.text_bbox:
            print(f"       텍스트영역: {p.text_bbox}")
    
    extractor.visualize(image_path, pairs, "./test_v3.jpg")
    extractor.save_pairs_json(pairs, "./test_v3_pairs.json")
    
    print("\n" + "=" * 60)
    print("전체 이미지 처리하려면:")
    print("  python contour_extractor_v3.py all")
    print("=" * 60)
    
    # 전체 처리
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        process_all_images("./Images", "./output_pairs", lang="de")
